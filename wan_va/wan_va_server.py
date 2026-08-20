# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import argparse
import os
import sys
import time
from functools import partial
from PIL import Image
from diffusers.video_processor import VideoProcessor
from diffusers.utils import export_to_video

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from einops import rearrange
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import VA_CONFIGS
from distributed.fsdp import shard_model
from distributed.util import _configure_model, init_distributed
from modules.utils import (
    WanVAEStreamingWrapper,
    load_text_encoder,
    load_tokenizer,
    load_transformer,
    load_vae,
)
from utils import (
    FlowMatchScheduler,
    data_seq_to_patch,
    get_mesh_id,
    init_logger,
    logger,
    run_async_server_mode,
    save_async,
)
from utils.attention_analysis import JointAttentionRecorder


class VA_Server:

    def __init__(self, job_config):
        self.cache_name = 'pos'
        self.job_config = job_config
        self.save_root = job_config.save_root
        self.dtype = job_config.param_dtype
        self.device = torch.device(f"cuda:{job_config.local_rank}")
        self.enable_offload = getattr(job_config, 'enable_offload', True)  # offload vae & text_encoder to save vram
        self.inference_audit = (
            bool(getattr(job_config, 'inference_audit', False))
            and getattr(job_config, 'rank', 0) == 0
        )
        self.inference_audit_layers = list(
            getattr(job_config, 'inference_audit_layers', [0]))
        self.attention_analysis_only = bool(
            getattr(job_config, 'attention_analysis_only', False)
        )
        self._audit_forward_index = 0

        self.scheduler = FlowMatchScheduler(shift=self.job_config.snr_shift,
                                            sigma_min=0.0,
                                            extra_one_step=True)
        self.action_scheduler = FlowMatchScheduler(
            shift=self.job_config.action_snr_shift,
            sigma_min=0.0,
            extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)
        self.action_scheduler.set_timesteps(1000, training=True)

        self.vae = load_vae(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'vae'),
            torch_dtype=self.dtype,
            torch_device='cpu' if self.enable_offload else self.device,
        )
        self.streaming_vae = WanVAEStreamingWrapper(self.vae)

        self.tokenizer = load_tokenizer(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'tokenizer'), )

        self.text_encoder = load_text_encoder(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'text_encoder'),
            torch_dtype=self.dtype,
            torch_device='cpu' if self.enable_offload else self.device,
        )

        # FSDP2 requires one original dtype for the wrapped parameter groups.
        # Keep checkpoint parameters in FP32; MixedPrecisionPolicy casts them to
        # the runtime dtype during forward while preserving stable FP32 modules.
        self.transformer = load_transformer(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'transformer'),
            torch_dtype=torch.float32,
            torch_device=self.device,
            attn_mode="torch"
        )
        self.attention_recorder = None
        if getattr(job_config, 'attention_analysis', False):
            attention_output = getattr(job_config, 'attention_output', None)
            if attention_output is None:
                attention_output = 'attn-exp'
            attention_layers = job_config.attention_layers
            if job_config.world_size > 1:
                attention_layers = attention_layers[
                    job_config.rank::job_config.world_size
                ]
            self.attention_recorder = JointAttentionRecorder(
                output_dir=attention_output,
                layers=attention_layers,
                summary_layers=job_config.attention_layers,
                worker_id=job_config.rank,
                write_summary=job_config.rank == 0,
                experiment_name=getattr(job_config, 'prompt', 'prompt-action'),
                chunk_tokens=job_config.attention_chunk_tokens,
                step_indices=getattr(job_config, 'attention_step_indices', None),
                vv_chunk_sparsity=getattr(
                    job_config, 'attention_vv_chunk_sparsity', False
                ),
                expected_chunks=job_config.num_chunks_to_infer,
            )
            self.transformer.set_attention_recorder(self.attention_recorder)
            vv_chunk_sparsity = getattr(
                job_config, 'attention_vv_chunk_sparsity', False
            )
            logger.info(
                f"Enable joint attention analysis for layers "
                f"{attention_layers} on rank {job_config.rank}; "
                f"steps={getattr(job_config, 'attention_step_indices', None)}; "
                f"vv_chunk_sparsity={vv_chunk_sparsity}; "
                f"output={attention_output}")
        shard_fn = shard_model
        self.transformer = _configure_model(model=self.transformer,
                                            shard_fn=shard_fn,
                                            param_dtype=self.dtype,
                                            device=self.device,
                                            eval_mode=True,
                                            )

        self.env_type = job_config.env_type
        self.streaming_vae_half = None
        if self.env_type == 'robotwin_tshape':
            vae_half = load_vae(
                os.path.join(job_config.wan22_pretrained_model_name_or_path,
                             'vae'),
                torch_dtype=self.dtype,
                torch_device='cpu' if self.enable_offload else self.device,
            )
            self.streaming_vae_half = WanVAEStreamingWrapper(vae_half)

    def _get_t5_prompt_embeds(
        self,
        prompt=None,
        num_videos_per_prompt=1,
        max_sequence_length=512,
        device=None,
        dtype=None,
    ):
        device = device or self.device
        dtype = dtype or self.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        text_encoder_device = next(self.text_encoder.parameters()).device
        prompt_embeds = self.text_encoder(text_input_ids.to(text_encoder_device),
                                          mask.to(text_encoder_device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack([
            torch.cat(
                [u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
            for u in prompt_embeds
        ],
                                    dim=0)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt,
                                           seq_len, -1)

        return prompt_embeds.to(device)

    def encode_prompt(
        self,
        prompt,
        negative_prompt=None,
        do_classifier_free_guidance=True,
        num_videos_per_prompt=1,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        max_sequence_length=226,
        device=None,
        dtype=None,
    ):
        r"""
        TODO
        """
        device = device or self.device
        dtype = dtype or self.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(
                negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(
                    negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}.")
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`.")

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
        return prompt_embeds, negative_prompt_embeds

    def normalize_latents(
        self,
        latents: torch.Tensor,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
    ) -> torch.Tensor:
        latents_mean = latents_mean.view(1, -1, 1, 1,
                                         1).to(device=latents.device)
        latents_std = latents_std.view(1, -1, 1, 1,
                                       1).to(device=latents.device)
        latents = ((latents.float() - latents_mean) * latents_std).to(latents)
        return latents

    def preprocess_action(self, action):
        action_model_input = torch.from_numpy(action)
        CA, FA, HA = action_model_input.shape  # C, F, H
        action_model_input_paded = F.pad(action_model_input,
                                         [0, 0, 0, 0, 0, 1],
                                         mode='constant',
                                         value=0)

        action_model_input = action_model_input_paded[
            self.job_config.inverse_used_action_channel_ids]

        if self.action_norm_method == 'quantiles':
            action_model_input = (action_model_input - self.actions_q01) / (
                self.actions_q99 - self.actions_q01 + 1e-6) * 2. - 1.
        else:
            raise NotImplementedError
        return action_model_input.unsqueeze(0).unsqueeze(-1)  # B, C, F, H, W

    def postprocess_action(self, action):
        action = action.cpu()  # B, C, F, H, W

        action = action[0, ..., 0]  #C, F, H
        if self.action_norm_method == 'quantiles':
            action = (action + 1) / 2 * (self.actions_q99 - self.actions_q01 +
                                         1e-6) + self.actions_q01
        else:
            raise NotImplementedError
        action = action.squeeze(0).detach().cpu().numpy()
        return action[self.job_config.used_action_channel_ids]
    
    def _repeat_input_for_cfg(self, input_dict):
        if self.use_cfg:
            input_dict['noisy_latents'] = input_dict['noisy_latents'].repeat(2, 1, 1, 1, 1)
            input_dict['text_emb'] = torch.cat([self.prompt_embeds.to(self.dtype).clone(), self.negative_prompt_embeds.to(self.dtype).clone()], dim=0)
            input_dict['grid_id'] = input_dict['grid_id'][None].repeat(2, 1, 1)
            input_dict['timesteps'] = input_dict['timesteps'][None].repeat(2, 1)
        else:
            input_dict['grid_id'] = input_dict['grid_id'][None]
            input_dict['timesteps'] = input_dict['timesteps'][None]
        return input_dict

    def _prepare_latent_input(self,
                              latent_model_input,
                              action_model_input,
                              latent_t=0,
                              action_t=0,
                              latent_cond=None,
                              action_cond=None,
                              frame_st_id=0,
                              patch_size=(1, 2, 2)):
        logger.info(f"FRAME START ID: {frame_st_id}")
        input_dict = dict()
        if latent_model_input is not None:
            input_dict['latent_res_lst'] = {
                'noisy_latents':
                latent_model_input,
                'timesteps':
                torch.ones([latent_model_input.shape[2]],
                           dtype=torch.float32,
                           device=self.device) * latent_t,
                'grid_id':
                get_mesh_id(latent_model_input.shape[-3] // patch_size[0],
                            latent_model_input.shape[-2] // patch_size[1],
                            latent_model_input.shape[-1] // patch_size[2], 0,
                            1, frame_st_id).to(self.device),
                'text_emb':
                self.prompt_embeds.to(self.dtype).clone(),
            }
            if latent_cond is not None:
                input_dict['latent_res_lst'][
                    'noisy_latents'][:, :, 0:1] = latent_cond[:, :, 0:1]
                input_dict['latent_res_lst']['timesteps'][0:1] *= 0

        if action_model_input is not None:
            input_dict['action_res_lst'] = {
                'noisy_latents':
                action_model_input,
                'timesteps':
                torch.ones([action_model_input.shape[2]],
                           dtype=torch.float32,
                           device=self.device) * action_t,
                'grid_id':
                get_mesh_id(action_model_input.shape[-3],
                            action_model_input.shape[-2],
                            action_model_input.shape[-1],
                            1,
                            1,
                            frame_st_id,
                            action=True).to(self.device),
                'text_emb':
                self.prompt_embeds.to(self.dtype).clone(),
            }

            if action_cond is not None:
                input_dict['action_res_lst'][
                    'noisy_latents'][:, :, 0:1] = action_cond[:, :, 0:1]
                input_dict['action_res_lst']['timesteps'][0:1] *= 0
            input_dict['action_res_lst']['noisy_latents'][:, ~self.
                                                          action_mask] *= 0
        return input_dict

    def _encode_obs(self, obs):
        images = obs['obs']
        if not isinstance(images, list):
            images = [images]
        if len(images) < 1:
            return None
        videos = []
        for k_i, k in enumerate(self.job_config.obs_cam_keys):
            if self.env_type == 'robotwin_tshape':
                if k_i == 0:  # camera high
                    height_i, width_i = self.height, self.width
                else:
                    height_i, width_i = self.height // 2, self.width // 2
            else:
                height_i, width_i = self.height, self.width

            history_video_k = torch.from_numpy(
                np.stack([each[k]
                          for each in images])).float().permute(3, 0, 1, 2)
            history_video_k = F.interpolate(history_video_k,
                                            size=(height_i, width_i),
                                            mode='bilinear',
                                            align_corners=False).unsqueeze(0)
            videos.append(history_video_k)

        if self.env_type == 'robotwin_tshape':
            videos_high = videos[0] / 255.0 * 2.0 - 1.0
            videos_left_and_right = torch.cat(videos[1:],
                                              dim=0) / 255.0 * 2.0 - 1.0
            vae_device = next(self.streaming_vae.vae.parameters()).device
            enc_out_high = self.streaming_vae.encode_chunk(
                videos_high.to(vae_device).to(self.dtype))
            enc_out_left_and_right = self.streaming_vae_half.encode_chunk(
                videos_left_and_right.to(vae_device).to(self.dtype))
            enc_out = torch.cat([
                torch.cat(enc_out_left_and_right.split(1, dim=0), dim=-1),
                enc_out_high
            ],
                                dim=-2)
        else:
            videos = torch.cat(videos, dim=0) / 255.0 * 2.0 - 1.0
            vae_device = next(self.streaming_vae.vae.parameters()).device
            videos_chunk = videos.to(vae_device).to(self.dtype)
            enc_out = self.streaming_vae.encode_chunk(videos_chunk)

        mu, logvar = torch.chunk(enc_out, 2, dim=1)
        latents_mean = torch.tensor(self.vae.config.latents_mean).to(mu.device)
        latents_std = torch.tensor(self.vae.config.latents_std).to(mu.device)
        mu_norm = self.normalize_latents(mu, latents_mean, 1.0 / latents_std)
        video_latent = torch.cat(mu_norm.split(1, dim=0), dim=-1)
        return video_latent.to(self.device)

    def _reset(self, prompt=None):
        logger.info('Reset.')
        self.use_cfg = (self.job_config.guidance_scale > 1) or (self.job_config.action_guidance_scale > 1)
        #### Reset all parameters
        self.frame_st_id = 0
        self.init_latent = None
        self._audit_forward_index = 0
        #### clean vae and transformer cache
        self.transformer.clear_cache(self.cache_name)
        self.streaming_vae.clear_cache()

        self.action_per_frame = self.job_config.action_per_frame
        self.height, self.width = self.job_config.height, self.job_config.width

        if self.env_type == 'robotwin_tshape':
            self.latent_height, self.latent_width = (
                (self.height // 16) * 3) // 2, self.width // 16
            self.streaming_vae_half.clear_cache()
        else:
            self.latent_height, self.latent_width = self.height // 16, self.width // 16 * len(
                self.job_config.obs_cam_keys)

        patch_size = self.job_config.patch_size
        latent_token_per_chunk = (self.job_config.frame_chunk_size *
                                  self.latent_height * self.latent_width) // (
                                      patch_size[0] * patch_size[1] *
                                      patch_size[2])
        action_token_per_chunk = self.job_config.frame_chunk_size * self.action_per_frame
        self.transformer.create_empty_cache(self.cache_name,
                                            self.job_config.attn_window,
                                            latent_token_per_chunk,
                                            action_token_per_chunk,
                                            dtype=self.dtype,
                                            device=self.device,
                                            batch_size = 2 if self.use_cfg else 1
                                            )

        self.action_mask = torch.zeros([self.job_config.action_dim]).bool()
        self.action_mask[self.job_config.used_action_channel_ids] = True

        self.actions_q01 = torch.tensor(self.job_config.norm_stat['q01'],
                                        dtype=torch.float32).reshape(-1, 1, 1)
        self.actions_q99 = torch.tensor(self.job_config.norm_stat['q99'],
                                        dtype=torch.float32).reshape(-1, 1, 1)
        self.action_norm_method = self.job_config.action_norm_method

        ##### get prompt
        if prompt is None:
            self.prompt_embeds = self.negative_prompt_embeds = None
        else:
            self.prompt_embeds, self.negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=None,
                do_classifier_free_guidance=self.job_config.guidance_scale > 1,
                num_videos_per_prompt=1,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                max_sequence_length=512,
                device=self.device,
                dtype=self.dtype,
            )

        self.exp_name = f"{prompt}_{time.strftime('%Y%m%d_%H%M%S')}" if prompt else "default"
        self.exp_save_root = os.path.join(self.save_root, 'real', self.exp_name)
        if not self.attention_analysis_only:
            os.makedirs(self.exp_save_root, exist_ok=True)
            torch.cuda.empty_cache()

    @staticmethod
    def _audit_scalar(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        return float(value)

    def _make_inference_trace(
        self,
        phase,
        step,
        frame_st_id,
        timestep,
        raw_sigma,
        sigma,
        next_sigma,
        update_cache,
        will_update,
        input_shape,
    ):
        if not self.inference_audit:
            return None
        trace = {
            'global_forward': self._audit_forward_index,
            'chunk': frame_st_id // self.job_config.frame_chunk_size,
            'phase': phase,
            'step': step,
            'timestep': self._audit_scalar(timestep),
            'raw_sigma': self._audit_scalar(raw_sigma),
            'sigma': self._audit_scalar(sigma),
            'next_sigma': self._audit_scalar(next_sigma),
            'update_cache': update_cache,
            'will_update': will_update,
        }
        self._audit_forward_index += 1
        next_sigma_text = (
            'none' if trace['next_sigma'] is None
            else f"{trace['next_sigma']:.9f}"
        )
        delta_sigma_text = (
            'none' if trace['next_sigma'] is None
            else f"{trace['next_sigma'] - trace['sigma']:.9f}"
        )
        logger.info(
            "[INFERENCE_AUDIT][SCHEDULER] "
            f"global_forward={trace['global_forward']} "
            f"chunk={trace['chunk']} phase={phase} step={step} "
            f"timestep={trace['timestep']:.6f} "
            f"raw_sigma={trace['raw_sigma']:.9f} "
            f"sigma={trace['sigma']:.9f} "
            f"next_sigma={next_sigma_text} delta_sigma={delta_sigma_text} "
            f"will_update={will_update} update_cache={update_cache} "
            f"input_x_shape={tuple(input_shape)}"
        )
        return trace

    @staticmethod
    def _log_inference_output(trace, output, output_type):
        if trace is None:
            return
        logger.info(
            "[INFERENCE_AUDIT][OUTPUT] "
            f"global_forward={trace['global_forward']} "
            f"chunk={trace['chunk']} phase={trace['phase']} "
            f"step={trace['step']} timestep={trace['timestep']:.6f} "
            f"output_type={output_type} output_shape={tuple(output.shape)} "
            f"state_update={trace['will_update']}"
        )

    def _infer(self, obs, frame_st_id=0):
        frame_chunk_size = self.job_config.frame_chunk_size
        if frame_st_id == 0:
            if self.init_latent is None:
                self.init_latent = self._encode_obs(obs)
            init_latent = self.init_latent

        latents = torch.randn(1,
                              48,
                              frame_chunk_size,
                              self.latent_height,
                              self.latent_width,
                              device=self.device,
                              dtype=self.dtype)
        actions = torch.randn(1,
                              self.job_config.action_dim,
                              frame_chunk_size,
                              self.action_per_frame,
                              1,
                              device=self.device,
                              dtype=self.dtype)

        video_inference_step = self.job_config.num_inference_steps
        action_inference_step = self.job_config.action_num_inference_steps
        video_step = self.job_config.video_exec_step

        self.scheduler.set_timesteps(video_inference_step)
        self.action_scheduler.set_timesteps(action_inference_step)
        timesteps = self.scheduler.timesteps
        raw_sigmas = self.scheduler.raw_sigmas
        sigmas = self.scheduler.sigmas
        action_timesteps = self.action_scheduler.timesteps
        action_raw_sigmas = self.action_scheduler.raw_sigmas
        action_sigmas = self.action_scheduler.sigmas

        timesteps = F.pad(timesteps, (0, 1), mode='constant', value=0)
        raw_sigmas = F.pad(raw_sigmas, (0, 1), mode='constant', value=0)
        sigmas = F.pad(sigmas, (0, 1), mode='constant', value=0)

        if video_step != -1:
            timesteps = timesteps[:video_step]
            raw_sigmas = raw_sigmas[:video_step]
            sigmas = sigmas[:video_step]

        action_timesteps = F.pad(
            action_timesteps,
            (0,
             1),  # pad 1 element at the end (right side) of the last dimension
            mode='constant',
            value=0)
        action_raw_sigmas = F.pad(
            action_raw_sigmas, (0, 1), mode='constant', value=0)
        action_sigmas = F.pad(
            action_sigmas, (0, 1), mode='constant', value=0)

        with (
                torch.no_grad(),
        ):
            # 1. Video Generation Loop
            for i, t in enumerate(tqdm(timesteps)):
                last_step = i == len(timesteps) - 1
                update_cache = 1 if last_step else 0
                will_update = not last_step or video_step != -1
                next_sigma = None
                if will_update:
                    _, next_sigma = self.scheduler.get_step_sigmas(t)
                trace = self._make_inference_trace(
                    phase='video',
                    step=i,
                    frame_st_id=frame_st_id,
                    timestep=t,
                    raw_sigma=raw_sigmas[i],
                    sigma=sigmas[i],
                    next_sigma=next_sigma,
                    update_cache=update_cache,
                    will_update=will_update,
                    input_shape=latents.shape,
                )
                latent_cond = init_latent[:, :, 0:1].to(
                    self.dtype) if frame_st_id == 0 else None
                input_dict = self._prepare_latent_input(
                    latents,
                    None,
                    t,
                    t,
                    latent_cond,
                    None,
                    frame_st_id=frame_st_id)

                video_noise_pred = self.transformer(
                    self._repeat_input_for_cfg(input_dict['latent_res_lst']),
                    update_cache=update_cache,
                    cache_name=self.cache_name,
                    action_mode=False,
                    attention_step_index=i,
                    attention_timestep=t,
                    inference_trace=trace,
                    inference_audit_layers=self.inference_audit_layers)
                self._log_inference_output(trace, video_noise_pred, 'velocity')

                if will_update:
                    video_noise_pred = data_seq_to_patch(
                        self.job_config.patch_size, video_noise_pred,
                        frame_chunk_size, self.latent_height,
                        self.latent_width, batch_size=2 if self.use_cfg else 1)
                    if self.job_config.guidance_scale > 1:
                        video_noise_pred = video_noise_pred[1:] + self.job_config.guidance_scale * (video_noise_pred[:1] - video_noise_pred[1:])
                    else:
                        video_noise_pred = video_noise_pred[:1]
                    latents = self.scheduler.step(video_noise_pred,
                                                  t,
                                                  latents,
                                                  return_dict=False)

                latents[:, :, 0:1] = latent_cond if frame_st_id == 0 else latents[:, :, 0:1]

            for i, t in enumerate(tqdm(action_timesteps)):
                last_step = i == len(action_timesteps) - 1
                update_cache = 1 if last_step else 0
                will_update = not last_step
                next_sigma = None
                if will_update:
                    _, next_sigma = self.action_scheduler.get_step_sigmas(t)
                trace = self._make_inference_trace(
                    phase='action',
                    step=i,
                    frame_st_id=frame_st_id,
                    timestep=t,
                    raw_sigma=action_raw_sigmas[i],
                    sigma=action_sigmas[i],
                    next_sigma=next_sigma,
                    update_cache=update_cache,
                    will_update=will_update,
                    input_shape=actions.shape,
                )
                action_cond = torch.zeros(
                    [
                        1, self.job_config.action_dim, 1,
                        self.action_per_frame, 1
                    ],
                    device=self.device,
                    dtype=self.dtype) if frame_st_id == 0 else None

                input_dict = self._prepare_latent_input(
                    None,
                    actions,
                    t,
                    t,
                    None,
                    action_cond,
                    frame_st_id=frame_st_id)
                action_noise_pred = self.transformer(
                    self._repeat_input_for_cfg(input_dict['action_res_lst']),
                    update_cache=update_cache,
                    cache_name=self.cache_name,
                    action_mode=True,
                    inference_trace=trace,
                    inference_audit_layers=self.inference_audit_layers)
                self._log_inference_output(trace, action_noise_pred, 'velocity')

                if will_update:
                    action_noise_pred = rearrange(action_noise_pred,
                                                  'b (f n) c -> b c f n 1',
                                                  f=frame_chunk_size)
                    if self.job_config.action_guidance_scale > 1:
                        action_noise_pred = action_noise_pred[1:] + self.job_config.action_guidance_scale * (action_noise_pred[:1] - action_noise_pred[1:])
                    else:
                        action_noise_pred = action_noise_pred[:1]
                    actions = self.action_scheduler.step(action_noise_pred,
                                                         t,
                                                         actions,
                                                         return_dict=False)

                actions[:, :, 0:1] = action_cond if frame_st_id == 0 else actions[:, :, 0:1]

        if self.attention_analysis_only:
            return None, latents

        actions[:, ~self.action_mask] *= 0
        save_async(latents, os.path.join(
            self.exp_save_root, f'latents_{frame_st_id}.pt'))
        save_async(actions, os.path.join(
            self.exp_save_root, f'actions_{frame_st_id}.pt'))

        actions = self.postprocess_action(actions)
        torch.cuda.empty_cache()
        return actions, latents

    def _compute_kv_cache(self, obs):
        ### optional async save obs for debug
        self.transformer.clear_pred_cache(self.cache_name)
        save_async(obs['obs'], os.path.join(self.exp_save_root, f'obs_data_{self.frame_st_id}.pt'))
        latent_model_input = self._encode_obs(obs)
        if self.frame_st_id == 0:
            latent_model_input = torch.cat(
                [self.init_latent, latent_model_input],
                dim=2) if latent_model_input is not None else self.init_latent

        action_model_input = self.preprocess_action(obs['state'])
        action_model_input = action_model_input.to(latent_model_input)
        logger.info(
            f"get KV cache obs: {latent_model_input.shape} {action_model_input.shape}"
        )
        input_dict = self._prepare_latent_input(latent_model_input,
                                                action_model_input,
                                                frame_st_id=self.frame_st_id)

        with (
                torch.no_grad(),
        ):
            video_trace = self._make_inference_trace(
                phase='cache_video',
                step=0,
                frame_st_id=self.frame_st_id,
                timestep=0,
                raw_sigma=0,
                sigma=0,
                next_sigma=None,
                update_cache=2,
                will_update=False,
                input_shape=input_dict['latent_res_lst']['noisy_latents'].shape,
            )
            video_output = self.transformer(
                self._repeat_input_for_cfg(input_dict['latent_res_lst']),
                update_cache=2,
                cache_name=self.cache_name,
                action_mode=False,
                inference_trace=video_trace,
                inference_audit_layers=self.inference_audit_layers)
            self._log_inference_output(video_trace, video_output, 'cache_only')

            action_trace = self._make_inference_trace(
                phase='cache_action',
                step=0,
                frame_st_id=self.frame_st_id,
                timestep=0,
                raw_sigma=0,
                sigma=0,
                next_sigma=None,
                update_cache=2,
                will_update=False,
                input_shape=input_dict['action_res_lst']['noisy_latents'].shape,
            )
            action_output = self.transformer(
                self._repeat_input_for_cfg(input_dict['action_res_lst']),
                update_cache=2,
                cache_name=self.cache_name,
                action_mode=True,
                inference_trace=action_trace,
                inference_audit_layers=self.inference_audit_layers)
            self._log_inference_output(action_trace, action_output, 'cache_only')
        torch.cuda.empty_cache()
        self.frame_st_id += latent_model_input.shape[2]

    @torch.no_grad()
    def infer(self, obs):
        reset = obs.get('reset', False)
        prompt = obs.get('prompt', None)
        compute_kv_cache = obs.get('compute_kv_cache', False)

        if reset:
            logger.info(f"******************* Reset server ******************")
            self._reset(prompt=prompt)
            return dict()
        elif compute_kv_cache:
            logger.info(
                f"################# Compute KV Cache #################")
            self._compute_kv_cache(obs)
            return dict()
        else:
            logger.info(f"################# Infer One Chunk #################")
            action, _ = self._infer(obs, frame_st_id=self.frame_st_id)
            return dict(action=action)
    
    def decode_one_video(self, latents, output_type):
        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        video = self.vae.decode(latents, return_dict=False)[0]
        video = self.video_processor.postprocess_video(video, output_type=output_type)
        return video
    
    def load_init_obs(self):
        imf_dict = {v: np.array(Image.open(os.path.join(self.job_config.input_img_path, f"{v}.png")).convert("RGB")) for v in self.job_config.obs_cam_keys}
        init_obs = {}
        init_obs['obs'] = [imf_dict]
        return init_obs
    
    @torch.no_grad()
    def generate(self):
        prompts = list(getattr(
            self.job_config, 'prompts', [self.job_config.prompt]))
        if not prompts:
            raise ValueError("At least one generation prompt is required")
        if not self.attention_analysis_only:
            self.video_processor = VideoProcessor(vae_scale_factor=1)

        init_obs = self.load_init_obs()
        cached_init_latent = None
        seen_attention_dirs = set()
        final_pred_latent = None
        for prompt_index, prompt in enumerate(prompts):
            if self.attention_recorder is not None:
                if prompt_index > 0:
                    self.attention_recorder.start_experiment(prompt)
                output_dir = self.attention_recorder.output_dir.resolve()
                if output_dir in seen_attention_dirs:
                    raise ValueError(
                        f"Prompt slug collision for attention output: {output_dir}"
                    )
                seen_attention_dirs.add(output_dir)

            if getattr(self.job_config, 'attention_analysis', False):
                attention_seed = int(getattr(
                    self.job_config, 'attention_seed', 0))
                torch.manual_seed(attention_seed)
                torch.cuda.manual_seed_all(attention_seed)
            logger.info(
                f"Generate prompt {prompt_index + 1}/{len(prompts)}: {prompt}"
            )
            self._reset(prompt)
            if cached_init_latent is None:
                self.init_latent = self._encode_obs(init_obs)
                cached_init_latent = self.init_latent.detach().clone()
            else:
                self.init_latent = cached_init_latent.clone()
            if getattr(self.job_config, 'attention_analysis', False):
                # Prompt/observation preprocessing must not shift generation noise.
                torch.manual_seed(attention_seed)
                torch.cuda.manual_seed_all(attention_seed)

            pred_latent_lst = []
            for chunk_id in range(self.job_config.num_chunks_to_infer):
                _, latents = self._infer(
                    init_obs,
                    frame_st_id=(
                        chunk_id * self.job_config.frame_chunk_size
                    ),
                )
                if not self.attention_analysis_only:
                    pred_latent_lst.append(latents)
            if pred_latent_lst:
                final_pred_latent = torch.cat(pred_latent_lst, dim=2)

            if self.attention_recorder is not None:
                summary_path = self.attention_recorder.save()
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
                if self.job_config.rank == 0:
                    logger.info(
                        f"Saved joint attention analysis to {summary_path}"
                    )
            self.transformer.clear_cache(self.cache_name)
            self.streaming_vae.clear_cache()
            if self.streaming_vae_half:
                self.streaming_vae_half.clear_cache()

        del self.transformer
        del self.streaming_vae_half
        del self.text_encoder
        torch.cuda.empty_cache()

        if self.attention_analysis_only or self.job_config.rank != 0:
            return
        if final_pred_latent is None:
            raise RuntimeError("Generation produced no video latent")

        # Move VAE to GPU for decoding.
        if self.enable_offload:
            self.vae = self.vae.to(self.device).to(self.dtype)

        decoded_video = self.decode_one_video(final_pred_latent, 'np')[0]
        export_to_video(
            decoded_video,
            os.path.join(self.save_root, "demo.mp4"),
            fps=10,
        )


def _read_prompts_file(path):
    prompts = []
    with open(path) as handle:
        for line in handle:
            prompt = line.rstrip("\r\n")
            if not prompt.strip() or prompt.lstrip().startswith("#"):
                continue
            prompts.append(prompt)
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def run(args):    
    
    config = VA_CONFIGS[args.config_name]
    port = config.port if args.port is None else args.port
    if args.save_root is not None:
        config.save_root = args.save_root
    if args.model_path is not None:
        config.wan22_pretrained_model_name_or_path = args.model_path
    if args.num_chunks_to_infer is not None:
        config.num_chunks_to_infer = args.num_chunks_to_infer
    if args.prompts_file is not None:
        config.prompts = _read_prompts_file(args.prompts_file)
    elif args.prompt is not None:
        config.prompts = [args.prompt]
    else:
        config.prompts = [config.prompt]
    config.prompt = config.prompts[0]
    config.attention_analysis = args.attention_analysis
    config.attention_analysis_only = args.attention_analysis_only
    config.attention_seed = args.attention_seed
    if args.attention_layers.strip().lower() == 'all':
        config.attention_layers = list(range(30))
    else:
        config.attention_layers = [
            int(layer) for layer in args.attention_layers.split(',') if layer.strip()
        ]
    config.attention_output = args.attention_output
    config.attention_chunk_tokens = args.attention_chunk_tokens
    config.attention_vv_chunk_sparsity = args.attention_vv_chunk_sparsity
    config.inference_audit = args.inference_audit
    if args.inference_audit_layers.strip().lower() == 'all':
        config.inference_audit_layers = list(range(30))
    else:
        config.inference_audit_layers = sorted({
            int(layer)
            for layer in args.inference_audit_layers.split(',')
            if layer.strip()
        })
    if config.inference_audit and (
        not config.inference_audit_layers
        or any(layer < 0 or layer >= 30
               for layer in config.inference_audit_layers)
    ):
        raise ValueError("inference audit layers must be in [0, 29]")
    if args.attention_timesteps is None:
        config.attention_step_indices = None
    else:
        config.attention_step_indices = sorted({
            int(step.strip())
            for step in args.attention_timesteps.split(',')
            if step.strip()
        })
    if config.attention_vv_chunk_sparsity and not config.attention_analysis:
        raise ValueError(
            "--attention-vv-chunk-sparsity requires --attention-analysis"
        )
    if (
        config.attention_vv_chunk_sparsity
        and config.num_chunks_to_infer < 2
    ):
        raise ValueError(
            "VV chunk sparsity requires at least two chunks"
        )
    if config.attention_analysis_only and not config.attention_analysis:
        raise ValueError(
            "--attention-analysis-only requires --attention-analysis"
        )
    if len(config.prompts) > 1 and not config.attention_analysis_only:
        raise ValueError(
            "Multiple prompts require --attention-analysis-only"
        )
    if config.attention_analysis:
        if config.infer_mode != 'i2va':
            raise ValueError("Attention analysis currently supports i2va mode only")
        if config.num_chunks_to_infer <= 0:
            raise ValueError("num_chunks_to_infer must be positive")
        if not config.attention_layers:
            raise ValueError("At least one attention layer is required")
        if any(layer < 0 or layer >= 30 for layer in config.attention_layers):
            raise ValueError("attention layers must be in [0, 29]")
        if config.attention_chunk_tokens <= 0:
            raise ValueError("attention_chunk_tokens must be positive")
        if config.attention_step_indices is not None:
            if not config.attention_step_indices:
                raise ValueError("At least one attention timestep is required")
            if any(step < 0 for step in config.attention_step_indices):
                raise ValueError("attention timestep indices must be non-negative")
            video_loop_steps = (
                config.num_inference_steps + 1
                if config.video_exec_step == -1 else
                min(config.video_exec_step, config.num_inference_steps + 1)
            )
            invalid_steps = [
                step for step in config.attention_step_indices
                if step >= video_loop_steps
            ]
            if invalid_steps:
                raise ValueError(
                    f"attention timestep indices {invalid_steps} exceed "
                    f"video loop length {video_loop_steps}"
                )
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    init_distributed(world_size, local_rank, rank)
    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size
    if config.attention_analysis:
        torch.manual_seed(config.attention_seed)
        torch.cuda.manual_seed_all(config.attention_seed)
    model = VA_Server(config)
    if config.infer_mode == 'i2va':
        logger.info(f"******************************USE I2AV mode******************************")
        model.generate()
    elif config.infer_mode == 'server':
        logger.info(f"******************************USE Server mode******************************")
        run_async_server_mode(model, local_rank, config.host, port)
    else:
        raise ValueError(f"Unknown infer mode: {config.infer_mode}")

def main():
    """
    TODO
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-name",
        type=str,
        required=False,
        default='robotwin',
        help="config name.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help='(start) port'
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default=None,
        help='save root'
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help='downloaded LingBot-VA model directory',
    )
    parser.add_argument(
        "--num-chunks-to-infer",
        type=int,
        default=None,
        help='override the number of generated video-action chunks',
    )
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument(
        "--prompt",
        type=str,
        default=None,
        help='override the text prompt (cross-prompt attention runs)',
    )
    prompt_group.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help='text file with one prompt per line for one-process extraction',
    )
    parser.add_argument(
        "--attention-analysis",
        action='store_true',
        help='capture video-to-video attention analysis CSVs',
    )
    parser.add_argument(
        "--attention-analysis-only",
        action='store_true',
        help='skip inference artifacts and video decode after attention export',
    )
    parser.add_argument(
        "--attention-seed",
        type=int,
        default=0,
        help='reset this RNG seed for every attention-analysis prompt',
    )
    parser.add_argument(
        "--attention-layers",
        type=str,
        default='all',
        help="'all' or comma-separated self-attention layer indices",
    )
    parser.add_argument(
        "--attention-output",
        type=str,
        default=None,
        help='attention CSV root; default is attn-exp',
    )
    parser.add_argument(
        "--attention-chunk-tokens",
        type=int,
        default=512,
        help='key-token chunk size for streaming attention extraction',
    )
    parser.add_argument(
        "--attention-vv-chunk-sparsity",
        action='store_true',
        help=(
            'write compact lXX.csv files with rows=heads and columns=chunks '
            'instead of full lXXhYY attention matrices'
        ),
    )
    parser.add_argument(
        "--attention-timesteps",
        type=str,
        default=None,
        help=(
            'comma-separated zero-based video denoising step indices; '
            'default captures the legacy final pass'
        ),
    )
    parser.add_argument(
        "--inference-audit",
        action='store_true',
        help='log scheduler, transformer, and self-attention execution facts',
    )
    parser.add_argument(
        "--inference-audit-layers",
        type=str,
        default='0,29',
        help="'all' or comma-separated self-attention layers to trace",
    )
    args = parser.parse_args()
    run(args)
    logger.info("Finish all process!!!!!!!!!!!!")


if __name__ == "__main__":
    init_logger()
    main()
