#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Run second-stage VV chunk-sparsity extraction.

Usage:
  uv run bash attn-scripts/run_vv_sparsity.sh \
    --model MODEL_DIR (--prompts FILE | --prompt TEXT) [options]

Options:
  --chunks N       Generated chunks (default: 10)
  --step LIST      Video step index or comma list (default: 24)
  --layers LIST    all or comma-separated layers (default: all)
  --output DIR     Attention output root
  --gpus N         GPU process count (default: 8)
  --seed N         Same RNG seed for every prompt (default: 0)
  --num-prompts N  Run only the first N prompts from --prompts
  --config NAME    Model config (default: robotwin_i2av)
  -h, --help       Show this help
EOF
}

model_path=${MODEL_PATH:-}
prompts_file=
prompt=
chunks=${CHUNKS:-10}
steps=${STEP:-24}
layers=${LAYERS:-all}
output_dir=${ATTENTION_OUTPUT:-}
ngpu=${NGPU:-8}
seed=${ATTENTION_SEED:-0}
num_prompts=${NUM_PROMPTS:-}
config_name=${CONFIG_NAME:-robotwin_i2av}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            model_path=$2
            shift 2
            ;;
        --prompts)
            prompts_file=$2
            shift 2
            ;;
        --prompt)
            prompt=$2
            shift 2
            ;;
        --chunks)
            chunks=$2
            shift 2
            ;;
        --step)
            steps=$2
            shift 2
            ;;
        --layers)
            layers=$2
            shift 2
            ;;
        --output)
            output_dir=$2
            shift 2
            ;;
        --gpus)
            ngpu=$2
            shift 2
            ;;
        --seed)
            seed=$2
            shift 2
            ;;
        --num-prompts)
            num_prompts=$2
            shift 2
            ;;
        --config)
            config_name=$2
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "${model_path}" ]]; then
    echo "--model or MODEL_PATH is required" >&2
    exit 2
fi
if [[ -n "${prompts_file}" && -n "${prompt}" ]] || \
   [[ -z "${prompts_file}" && -z "${prompt}" ]]; then
    echo "Choose exactly one of --prompts or --prompt" >&2
    exit 2
fi
if [[ -n "${num_prompts}" && -z "${prompts_file}" ]]; then
    echo "--num-prompts requires --prompts" >&2
    exit 2
fi
if [[ -n "${num_prompts}" && ! "${num_prompts}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--num-prompts must be a positive integer" >&2
    exit 2
fi
if [[ ! "${chunks}" =~ ^[1-9][0-9]*$ ]] || ((chunks < 2)); then
    echo "--chunks must be an integer greater than or equal to 2" >&2
    exit 2
fi
if [[ ! "${steps}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "--step must be an index or comma-separated indices" >&2
    exit 2
fi
if [[ "${layers}" != all && ! "${layers}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "--layers must be all or comma-separated indices" >&2
    exit 2
fi
if [[ ! "${ngpu}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--gpus must be a positive integer" >&2
    exit 2
fi
if [[ ! "${seed}" =~ ^-?[0-9]+$ ]]; then
    echo "--seed must be an integer" >&2
    exit 2
fi

if [[ "${model_path}" != /* ]]; then
    model_path="${PWD}/${model_path}"
fi
if [[ ! -d "${model_path}" ]]; then
    echo "Model directory does not exist: ${model_path}" >&2
    exit 2
fi
if [[ -n "${prompts_file}" && "${prompts_file}" != /* ]]; then
    prompts_file="${PWD}/${prompts_file}"
fi
if [[ -n "${prompts_file}" && ! -f "${prompts_file}" ]]; then
    echo "Prompt file does not exist: ${prompts_file}" >&2
    exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${repo_root}"

if [[ -z "${output_dir}" ]]; then
    step_tag=${steps//,/-}
    output_dir="attn-output/vv-sparsity-c${chunks}-s${step_tag}"
fi

prompts_file_for_run=${prompts_file}
temp_prompts_file=
if [[ -n "${num_prompts}" ]]; then
    temp_prompts_file=$(mktemp)
    trap 'rm -f -- "${temp_prompts_file}"' EXIT
    awk -v limit="${num_prompts}" '
        /^[[:space:]]*$/ || /^[[:space:]]*#/ { next }
        count < limit {
            print
            count++
            if (count == limit) exit
        }
    ' "${prompts_file}" > "${temp_prompts_file}"
    prompts_file_for_run=${temp_prompts_file}
fi

server_args=(
    --model-path "${model_path}"
    --num-chunks-to-infer "${chunks}"
    --attention-analysis
    --attention-analysis-only
    --attention-vv-chunk-sparsity
    --attention-layers "${layers}"
    --attention-timesteps "${steps}"
    --attention-output "${output_dir}"
    --attention-seed "${seed}"
)
if [[ -n "${prompts_file}" ]]; then
    server_args+=(--prompts-file "${prompts_file_for_run}")
else
    server_args+=(--prompt "${prompt}")
fi

echo "VV sparsity: chunks=${chunks} step=${steps} layers=${layers} gpus=${ngpu}"
if [[ -n "${num_prompts}" ]]; then
    echo "Prompts: first ${num_prompts} non-empty, non-comment entries"
fi
echo "Output: ${output_dir}"
NGPU="${ngpu}" CONFIG_NAME="${config_name}" \
bash script/run_launch_va_server_sync.sh "${server_args[@]}"
