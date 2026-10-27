#!/bin/bash
set -e

source /opt/ros/humble/setup.bash
export PYTHONPATH="/opt/regroup:${PYTHONPATH}"

# Usage:
#   CMD = "regroup_detector"   → live ROS2 node (default)
#   CMD = "bag_replay"         → offline: read bag → run REGROUP → write video
#   CMD = anything else        → exec directly (e.g. bash, ros2 topic list …)
#
# All REGROUP parameters are configurable via environment variables.
# See docker-compose.yml for the full list.

# if [ "${1:-}" = "bag_replay" ]; then
#     : "${BAG_PATH:?BAG_PATH env var is required}"
#     : "${TOPIC:?TOPIC env var is required}"
#     : "${OUTPUT_VIDEO:?OUTPUT_VIDEO env var is required}"
#     exec python3 /opt/regroup/bag_replay.py \
#         --bag    "${BAG_PATH}" \
#         --topic  "${TOPIC}" \
#         --output "${OUTPUT_VIDEO}" \
#         --device           "${DEVICE:-cuda:0}" \
#         --yolo-model       "${YOLO_MODEL:-yolo11x.pt}" \
#         --reid-weights     "${REID_WEIGHTS:-osnet_x0_25_msmt17.pt}" \
#         --min-confidence   "${MIN_CONFIDENCE:-0.5}" \
#         --nms-max-overlap  "${NMS_MAX_OVERLAP:-0.5}" \
#         --min-det-height   "${MIN_DET_HEIGHT:-0}" \
#         --max-cosine-dist  "${MAX_COSINE_DIST:-0.4}" \
#         --nn-budget        "${NN_BUDGET:-100}" \
#         --cif              "${CIF:-4}" \
#         --gp-dist          "${GP_DIST:-20.0}" \
#         --h-ratio          "${H_RATIO:-0.8}" \
#         --cost-metric      "${COST_METRIC:-2}"
# fi

# if [ "${1:-}" = "regroup_detector" ]; then
#     exec python3 /opt/regroup/regroup_detector_node.py --ros-args \
#         -p "image_topic:=${IMAGE_TOPIC:-/camera/image_raw}" \
#         -p "device:=${DEVICE:-cuda:0}" \
#         -p "yolo_model:=${YOLO_MODEL:-yolo11x.pt}" \
#         -p "reid_weights:=${REID_WEIGHTS:-osnet_x0_25_msmt17.pt}" \
#         -p "min_confidence:=${MIN_CONFIDENCE:-0.5}" \
#         -p "nms_max_overlap:=${NMS_MAX_OVERLAP:-0.5}" \
#         -p "min_det_height:=${MIN_DET_HEIGHT:-0}" \
#         -p "max_cosine_dist:=${MAX_COSINE_DIST:-0.4}" \
#         -p "nn_budget:=${NN_BUDGET:-100}" \
#         -p "cif:=${CIF:-4}" \
#         -p "gp_dist:=${GP_DIST:-20.0}" \
#         -p "h_ratio:=${H_RATIO:-0.8}" \
#         -p "cost_metric:=${COST_METRIC:-2}"
# fi

exec "$@"
