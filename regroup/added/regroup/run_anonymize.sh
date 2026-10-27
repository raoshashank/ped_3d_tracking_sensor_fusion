#!/usr/bin/env bash
# Run regroup_anonymize.py on every ROS2 bag found under a root directory.
# A bag is identified by the presence of a metadata.yaml file anywhere in the tree.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 2 ]]; then
    cat <<'USAGE' >&2
Usage: run_anonymize.sh <bags_root> <output_root> [regroup_anonymize.py args...]

Recursively searches <bags_root> for ROS2 bags (directories containing
metadata.yaml), then runs regroup_anonymize.py on each one.
Each bag's outputs go into a mirrored subdirectory under <output_root>.

Extra arguments are forwarded verbatim to regroup_anonymize.py, e.g.:
  --img-topic /zed/image_raw/compressed --odom-topic /odom --device cuda:0
USAGE
    exit 1
fi

bags_root="$(realpath "$1")"
output_root="$(realpath "$2")"
shift 2

if [[ ! -d "$bags_root" ]]; then
    echo "ERROR: bags root does not exist or is not a directory: $bags_root" >&2
    exit 1
fi

# Find every directory that contains a metadata.yaml (= a ROS2 bag root).
mapfile -t bag_dirs < <(
    find "$bags_root" -type f -name 'metadata.yaml' -printf '%h\n' | sort -u
)

if [[ ${#bag_dirs[@]} -eq 0 ]]; then
    echo "No ROS2 bags (metadata.yaml) found under: $bags_root" >&2
    exit 1
fi

echo "Found ${#bag_dirs[@]} bag(s) under: $bags_root"
mkdir -p "$output_root"

n_ok=0
n_fail=0

for bag_dir in "${bag_dirs[@]}"; do
    # Mirror the relative path of the bag under output_root.
    if [[ "$bag_dir" == "$bags_root" ]]; then
        bag_output="$output_root"
    else
        rel_path="${bag_dir#"$bags_root"/}"
        bag_output="$output_root/$rel_path"
    fi

    mkdir -p "$bag_output"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Bag    : $bag_dir"
    echo "Output : $bag_output"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if python3 "$SCRIPT_DIR/regroup_anonymize.py" \
            "$bag_dir" "$bag_output" "$@"; then
        echo "  ✓ done"
        (( n_ok++ )) || true
    else
        echo "  ✗ FAILED (exit $?)" >&2
        (( n_fail++ )) || true
    fi
done

echo ""
echo "Finished: $n_ok succeeded, $n_fail failed."
[[ $n_fail -eq 0 ]]
