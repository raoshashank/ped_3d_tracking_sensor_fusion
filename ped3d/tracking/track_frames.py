"""Run YOLO11x + BoT-SORT (the pedestrian tracker of regroup-hri/process_videos.py) on the 4 Hz
sampled image folders of a processed trajectory, instead of a 30 fps video.

    python track_frames.py --dataset uex --imgs-subdir imgs_zed_zed_node_rgb_image_rect_color \
        --trajs t1 t2 ... [--no-reid] [--out-name tracks_botsort.pkl]

Output per trajectory: <out_root>/<dataset>/<traj>/<out-name>, a list (one per frame) of
(boxes_tlwh, ids) in the same layout as the processed tracks.pkl / regroup_tracks_sampled 'tracks'.
"""
import argparse
import os
import pickle
import time

import cv2
import yaml
from ultralytics import YOLO

PROCESSED_ROOT = '/media/shashank/T1/processed'
REGROUP_DIR = '/home/shashank/code/packages/regroup-hri/regroup'
OUT_ROOT = '/media/shashank/ssd2/ped3d/tracking'


def make_tracker_cfg(reid, track_buffer, gmc, tag):
    cfg = yaml.safe_load(open(os.path.join(REGROUP_DIR, 'botsort.yaml')))
    cfg['with_reid'] = bool(reid)
    cfg['track_buffer'] = int(track_buffer)
    cfg['gmc_method'] = gmc
    path = os.path.join(OUT_ROOT, f'botsort_{tag}.yaml')
    os.makedirs(OUT_ROOT, exist_ok=True)
    with open(path, 'w') as f:
        yaml.safe_dump(cfg, f)
    return path


def track_folder(model, img_dir, tracker_cfg, conf, iou, device):
    files = sorted((f for f in os.listdir(img_dir) if f.endswith('.jpg')), key=lambda f: int(f[:-4]))
    log = []
    for k, f in enumerate(files):
        frame = cv2.imread(os.path.join(img_dir, f))
        res = model.track(frame, tracker=tracker_cfg, persist=True, classes=[0], device=device,
                          conf=conf, iou=iou, verbose=False)
        b = res[0].boxes
        if b.id is not None:
            ids = b.id.int().cpu().tolist()
            tlwh = b.xywh.cpu().numpy().copy()
            tlwh[:, 0] -= tlwh[:, 2] / 2
            tlwh[:, 1] -= tlwh[:, 3] / 2
            log.append((tlwh.tolist(), ids))
        else:
            log.append(([], []))
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--imgs-subdir', default='imgs')
    ap.add_argument('--trajs', nargs='*', default=None)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--no-reid', action='store_true')
    ap.add_argument('--track-buffer', type=int, default=30)
    ap.add_argument('--gmc', default='sparseOptFlow')
    ap.add_argument('--yolo', default=os.path.join(REGROUP_DIR, 'yolo11x.pt'))
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--iou', type=float, default=0.7)
    ap.add_argument('--device', default=0)
    ap.add_argument('--out-name', default=None)
    args = ap.parse_args()

    tag = ('noreid' if args.no_reid else 'reid') + f'_buf{args.track_buffer}_{args.gmc}'
    out_name = args.out_name or f'tracks_botsort_{tag}.pkl'
    root = os.path.join(PROCESSED_ROOT, args.dataset)
    trajs = args.trajs or sorted(t for t in os.listdir(root) if os.path.isdir(os.path.join(root, t, args.imgs_subdir)))
    if args.limit:
        trajs = trajs[:args.limit]
    model = YOLO(args.yolo)
    t0 = time.time()
    for k, traj in enumerate(trajs):
        out = os.path.join(OUT_ROOT, args.dataset, traj, out_name)
        if os.path.exists(out):
            continue
        # a fresh tracker state per trajectory
        cfg = make_tracker_cfg(not args.no_reid, args.track_buffer, args.gmc, tag)
        if getattr(model, 'predictor', None) is not None and getattr(model.predictor, 'trackers', None):
            for trk in model.predictor.trackers:
                trk.reset()
        log = track_folder(model, os.path.join(root, traj, args.imgs_subdir), cfg, args.conf, args.iou, args.device)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'wb') as f:
            pickle.dump(log, f)
        n_ids = len({i for _, ids in log for i in ids})
        print(f'[{k + 1}/{len(trajs)} {time.time() - t0:.0f}s] {traj}: {len(log)} frames, {n_ids} ids -> {out_name}', flush=True)


if __name__ == '__main__':
    main()
