#!/bin/bash
# GIF fallbacks small enough for GitHub (~5 MB each): a 12-14 s highlight of each MP4 (the full-length
# MP4s stay in figures/), 560 px wide, 10 fps, 64 colours, no dithering (dithering or fewer colours
# turned the label colours grey).
# usage: bash scripts/make_gifs.sh   (from the blog folder; needs ffmpeg)
set -e
cd "$(dirname "$0")/../figures"
gif() {  # in.mp4 out.gif start_s duration_s fps width colours
  ffmpeg -loglevel error -y -ss "$3" -t "$4" -i "$1" -vf "fps=$5,scale=$6:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=$7:stats_mode=full[p];[b][p]paletteuse=dither=none:diff_mode=rectangle" "$2"
  printf "%5.1f MB  %s\n" "$(echo "$(stat -c %s "$2")/1000000" | bc -l)" "$2"
}
# start times centre the highlight on each video's poster moment (videos play at 2x, the camera one at 1.5x)
gif 00_before_after.mp4             00_before_after.gif             4.7 12 10 560 64
gif 19_before_after_utown_event.mp4 19_before_after_utown_event.gif 3.1 12 10 560 64
gif 20_before_after_robosg_hall.mp4 20_before_after_robosg_hall.gif 1.8 12 10 560 64
gif 21_before_after_science.mp4     21_before_after_science.gif     17.4 12 10 560 64
gif 01_inputs_camera.mp4            01_inputs_camera.gif            0   12 7  480 64
# figure 14 (made by 14_fused_animation.py as a GIF; its MP4 is converted from that GIF)
gif 14_fused_animation.mp4          14_fused_animation.gif          5   11 8  480 64
