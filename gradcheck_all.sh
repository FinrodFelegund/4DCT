#!/usr/bin/env bash

set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo ${ROOT}
PYTHON="$ROOT/.venv/bin/python"
PYTHON="${PYTHON:-python}"
echo ${PYTHON}
failed=()
GPU=2

for pkg in LearnableSpatialFilter4D BilateralFilter3D BilateralFilter4D SpatioTemporalFilter WeightedCenterFilter4D LearnableFilter4D LearnableSpatialFilter4D AllLearnable4D; do
    echo
    echo "######## $pkg ########"
    if ! (cd "$ROOT/models/$pkg" && CUDA_VISIBLE_DEVICES="$GPU" $PYTHON gradcheck.py); then
        failed+=("$pkg")
    fi 
done

echo
if (( ${#failed[@]} )); then
    echo "Failed: ${failed[*]}"
    exit 1
fi

echo "All gradchecks ran without error. Check the output for any 'False'."
