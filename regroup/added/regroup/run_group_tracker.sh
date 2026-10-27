#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 ]]; then
	cat <<'USAGE' >&2
Usage: run_group_tracker.sh <trajectories_root> [process_videos_grouponly.py args...]

Runs process_videos_grouponly.py for every trajectory folder found under
<trajectories_root> that contains a *_zed2_zed_node_rgb_image_rect_color_compressed.avi file.
USAGE
	exit 1
fi

trajectories_root="$1"
shift

if [[ ! -d "$trajectories_root" ]]; then
	echo "Trajectories root does not exist or is not a directory: $trajectories_root" >&2
	exit 1
fi

mapfile -t video_files < <(
	find "$trajectories_root" -type f -name '*_zed2_zed_node_rgb_image_rect_color_compressed.avi' | sort
)

if [[ ${#video_files[@]} -eq 0 ]]; then
	echo "No '*_zed2_zed_node_rgb_image_rect_color_compressed.avi' files found under: $trajectories_root" >&2
	exit 1
fi

for video in "${video_files[@]}"; do
	echo "Processing: $video"
	python3 "$SCRIPT_DIR/process_videos_grouponly.py" "$video" "$@"
done
