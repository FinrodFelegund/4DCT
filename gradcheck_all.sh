#!/usr/bin/env bash

set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo ${ROOT}
PYTHON="$ROOT/.venv/bin/python"
PYTHON="${PYTHON:-python}"
echo ${PYTHON}
failed=()
GPU=3
PACKAGES=(BilateralFilterND BilateralFilterNDLUT BilateralFilterNDKernel BilateralFilterNDKernelLUT BilateralFilter4DWeightedCenter)

echo "root:   $ROOT"
echo "python: $PYTHON   gpu: $GPU"
failed=()

for pkg in ${PACKAGES[@]+"${PACKAGES[@]}"}; do
    echo
    echo "######## $pkg ########"
    if [[ ! -f "$ROOT/models/$pkg/gradcheck.py" ]]; then
        echo "missing: models/$pkg/gradcheck.py"
        failed+=("$pkg")
        continue
    fi
    if ! (cd "$ROOT/models/$pkg" && CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" gradcheck.py); then
        failed+=("$pkg")
    fi
done


#echo
#echo "######## equivalence test ########"
#if ! (cd "$ROOT/models" && CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" equivalence.py); then
#    failed+=("equivalence")
#fi

echo
if (( ${#failed[@]} )); then
    echo "Failed: ${failed[*]}"
    exit 1
fi
echo "All checks ran without error. Check the output for any 'False' or 'DIFF'."