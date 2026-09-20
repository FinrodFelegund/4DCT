#!/usr/bin/env bash

set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo ${ROOT}
PYTHON="${PYTHON:-python}"
failed=()

#BilateralFilter3D BilateralFilter4D SpatioTemporalFilter WeightedCenterFilter4D LearnableSpatialFilter4D

for pkg in AllLearnable4D; do
    echo
    echo "######## $pkg ########"
    if ! (cd "$ROOT/models/$pkg" && $PYTHON gradcheck.py); then
        failed+=("$pkg")
    fi 
done

echo
if (( ${#failed[@]} )); then
    echo "Failed: ${failed[*]}"
    exit 1
fi

echo "All gradchecks ran without error. Check the output for any 'False'."
