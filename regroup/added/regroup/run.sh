#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 2 ]]; then
	cat <<'USAGE' >&2
Usage: run.sh <bags_root> <output_root> [preprocess_bags.py args...]

Runs regroup/preprocess_bags.py once for every ROS2 bag directory found under
<bags_root>. Each bag is written to its own subdirectory under <output_root>.
USAGE
	exit 1
fi

bags_root="$1"
output_root="$2"
shift 2

if [[ ! -d "$bags_root" ]]; then
	echo "Bags root does not exist or is not a directory: $bags_root" >&2
	exit 1
fi

mapfile -t bag_dirs < <(
	find "$bags_root" -type f -name '*.db3' -printf '%h\n' | sort -u
)

if [[ ${#bag_dirs[@]} -eq 0 ]]; then
	echo "No ROS2 bag directories found under: $bags_root" >&2
	exit 1
fi

mkdir -p "$output_root"

for bag_dir in "${bag_dirs[@]}"; do
	if [[ "$bag_dir" == "$bags_root" ]]; then
		bag_output="$output_root"
	else
		rel_path="${bag_dir#"$bags_root"/}"
		bag_output="$output_root/$rel_path"
	fi

	mkdir -p "$bag_output"
	echo "Processing bag: $bag_dir"
	echo "Output dir: $bag_output"
	python3 "$SCRIPT_DIR/preprocess_bags.py" "$bag_dir" "$bag_output" "$@"
done
