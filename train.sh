#!/usr/bin/env bash

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$ROOT/.venv/bin/python"
GPU=0
DRY_RUN=0

usage() {
    cat <<'EOF'
Run training configs in sequence on a single GPU.

Usage:
  ./train.sh                           every configs/*.yml on GPU 0
  ./train.sh -g 1                      ... on GPU 1
  ./train.sh -g 1 -f AllLearnable4D    a single config, by bare name
  ./train.sh -f a.yml -f b.yml         repeat -f to pick a subset
  ./train.sh -n                        print the queue, then exit

Options:
  -g, --gpu N        GPU index for CUDA_VISIBLE_DEVICES (default 0)
  -f, --config PATH  config to run, repeatable. Takes a path, or a bare
                     name looked up in configs/ with .yml appended.
  -n, --dry-run      print what would run instead of running it
  -h, --help         show this message

Each run is logged to logs/<timestamp>/<config>.log. A failing run does not
stop the queue; the summary lists failures and the exit status is then 1.
EOF
    exit "${1:-0}"
}

resolve_config() {
    local arg="$1" candidate
    [[ "$arg" == /*  ]] && { printf '%s\n' "$arg"; return; }
    [[ "$arg" == */* ]] && { printf '%s\n' "$PWD/$arg"; return; }
    for candidate in "$ROOT/configs/$arg" "$ROOT/configs/$arg.yml" "$ROOT/configs/$arg.yaml"; do
        [[ -f "$candidate" ]] && { printf '%s\n' "$candidate"; return; }
    done
    printf '%s\n' "$PWD/$arg"
}

selected=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -g|--gpu)     [[ ${2:-} ]] || { echo "-g requires a value" >&2; exit 1; }
                      GPU="$2"; shift 2 ;;
        -f|--config)  [[ ${2:-} ]] || { echo "-f requires a value" >&2; exit 1; }
                      selected+=("$2"); shift 2 ;;
        -n|--dry-run) DRY_RUN=1; shift ;;
        -h|--help)    usage 0 ;;
        --)           shift; break ;;
        -*)           echo "unknown option: $1" >&2; usage 1 ;;
        *)            break ;;
    esac
done

requested=( ${selected[@]+"${selected[@]}"} "$@" )

configs=()
if (( ${#requested[@]} > 0 )); then
    for arg in "${requested[@]}"; do
        configs+=("$(resolve_config "$arg")")
    done
else
    shopt -s nullglob
    configs=("$ROOT"/configs/*.yml "$ROOT"/configs/*.yaml)
    shopt -u nullglob
fi

if (( ${#configs[@]} == 0 )); then
    echo "No config files found in $ROOT/configs" >&2
    exit 1
fi


stamp="$(date +%Y%m%d_%H%M%S)"
logdir="$ROOT/logs/$stamp"

echo "root:     $ROOT"
echo "python:   $PYTHON"
echo "gpu:      $GPU"
echo "configs:  ${#configs[@]}"
(( DRY_RUN )) || { mkdir -p "$logdir" && echo "logs:    $logdir"; }

trap 'echo; echo "interrupted, stopping queue"; exit 130' INT

elapsed_str() {
    local s=$1
    printf '%02d:%02d:%02d' $(( s / 3600 )) $(( (s % 3600) / 60 )) $(( s % 60 ))
}

passed=()
failed=()

for cfg in "${configs[@]}"; do
    name="$(basename "${cfg%.*}")"
    echo
    echo "######## $name ########"

    if (( DRY_RUN )); then
        echo "would run: CUDA_VISIBLE_DEVICES=$GPU $PYTHON main.py --train -f $cfg"
        continue
    fi

    if [[ ! -f "$cfg" ]]; then
        echo "missing config: $cfg" >&2
        failed+=("$name")
        continue
    fi

    start=$SECONDS
    (cd "$ROOT" && CUDA_VISIBLE_DEVICES="$GPU" $PYTHON main.py --train -f "$cfg") \
        2>&1 | tee "$logdir/$name.log"
    status=${PIPESTATUS[0]}
    took="$(elapsed_str $(( SECONDS - start )))"

    if (( status == 0 )); then
        echo "-- $name finished in $took"
        passed+=("$name")
    else
        echo "-- $name failed after $took (exit $status), see $logdir/$name.log"
        failed+=("$name")
    fi
done

(( DRY_RUN )) && exit 0

echo
echo "##### summary #####"
for name in ${passed[@]+"${passed[@]}"}; do echo "ok    $name"; done
for name in ${failed[@]+"${failed[@]}"}; do echo "FAIL  $name"; done

if (( ${#failed[@]} )); then
    exit 1
fi