"""Export pedestrian tracks + robot state of a whole corpus into ONE pickle for model training.

    $P3D_CONDA_ENVS/calib/bin/python export_tracks.py --dataset go2nus [--min-track-len 3]
    $P3D_CONDA_ENVS/calib/bin/python export_tracks.py --all

Output: $P3D_PED3D_OUT/tracks_<dataset>.pkl   (format_version 3, per-track layout)

    import ped3d_lib as L
    data = L.load_tracks('go2nus')
    data['dataset'], data['format_version'], data['created'], data['notes']
    entry = data['trajectories'][traj_name]
    entry = {
        'subdataset':  location group (see ped3d_lib.subdataset)
        'n_frames':    T
        'timestamps':  (T,) float64 seconds (4 Hz sampling of the processed data)
        'robot': {                       # odom/world frame of the trajectory
            'position':    (T, 2) float32  x, y
            'yaw':         (T,)   float32
            'linear_vel':  (T, 2) float32  (from traj_data.pkl)
            'angular_vel': (T,)   float32
        },
        'tracks': {track_id: {            # one tracker track as given by the regroup tracker
            'frames':    (n,)    int32    frame indices, first..last of the track span
            'pos_world': (n, 2)  float32  fused + RTS-smoothed xy (odom frame)
            'vel_world': (n, 2)  float32  smoother velocity estimate
            'pos_robot': (n, 2)  float32  same positions in the robot frame (x forward, y left)
            'pos_std':   (n,)    float32  smoother 1-sigma position uncertainty
            'detected':  (n,)    bool     True = tracked in this frame, False = interpolated
            'source':    (n,)    int8     0 none / 1 lidar only / 2 vggt only / 3 lidar+vggt
            'box':       (n, 4)  float32  image box [x, y, w, h] on detected frames, NaN otherwise
            'merged_id': int              id of the merged track (relink.py) this track belongs to
        }},
        'merged_tracks': {merged_id: {    # after duplicate merging + gap re-linking (relink.py),
            ... same fields as tracks ... # re-smoothed over the whole chain; gap frames have
            'members': (k,) int32         # detected=False; merged_id == first member's track id
        }},
        'relink': {'n_links', 'n_duplicates', 'n_ambiguous', 'links': (L, 3) [id_a, id_b, gap_s]}
    }

    dense = L.dense_tracks(entry, 'merged_tracks')  # -> ids (P,), pos_world (T,P,2), vel_world, pos_robot,
                                                   #    pos_std (T,P), detected, source, range (T,P), box (T,P,4)
    frame_t = L.frame_pedestrians(entry, t, key='merged_tracks')   # pedestrians present at frame t

Positions are the fused estimate (VGGT-Omega depth + lidar, per-dataset range correction, per-track
lidar anchoring, constant-velocity RTS smoother, see ped3d_lib). Tracks shorter than --min-track-len
detected frames are dropped before re-linking. Re-linking precision on synthetic cuts is 92-97 %,
recall 56-82 % (see relink_analysis.log); use 'tracks' for the raw tracker ids.
"""
import argparse
import os
import pickle
import time

import numpy as np

import ped3d_lib as L
import relink as R


def export_dataset(dataset, min_track_len=3):
    d = os.path.join(L.OUT_ROOT, dataset)
    files = sorted(f for f in os.listdir(d) if f.endswith('.pkl')) if os.path.isdir(d) else []
    if not files:
        print(f'{dataset}: nothing processed')
        return None
    trajs = {}
    n_det = n_tracks = n_merged = 0
    for f in files:
        with open(os.path.join(d, f), 'rb') as fh:
            res = pickle.load(fh)
        traj = res['traj']
        T = res['n']
        with open(os.path.join(L.PROCESSED_ROOT, dataset, traj, 'traj_data.pkl'), 'rb') as fh:
            td = pickle.load(fh)
        yaw = res['robot_yaw'][:T].astype(np.float32)
        rpos = res['robot_pos'][:T].astype(np.float32)
        boxes = {}
        for i, insts in enumerate(res['frames']):
            for inst in insts:
                boxes[(i, inst['id'])] = inst['box'].astype(np.float32)
        tracks = {}
        for tid, tr in res['tracks'].items():
            if int(tr['detected'].sum()) < min_track_len:
                continue
            fr = tr['frames'].astype(np.int32)
            pos = tr['fused'][:, :2].astype(np.float32)
            c, s = np.cos(yaw[fr])[:, None], np.sin(yaw[fr])[:, None]
            dxy = pos - rpos[fr]
            pos_robot = np.concatenate([c * dxy[:, :1] + s * dxy[:, 1:], -s * dxy[:, :1] + c * dxy[:, 1:]], 1)
            box = np.full((len(fr), 4), np.nan, np.float32)
            for k, i in enumerate(fr):
                b = boxes.get((int(i), int(tid)))
                if b is not None:
                    box[k] = b
            tracks[int(tid)] = dict(
                frames=fr, pos_world=pos, vel_world=tr['fused'][:, 2:4].astype(np.float32),
                pos_robot=pos_robot.astype(np.float32),
                pos_std=np.sqrt(np.maximum(tr['fused_var'], 0)).astype(np.float32),
                detected=tr['detected'].astype(bool),
                source=np.array([(1 if 'lidar' in x else 0) + (2 if 'vggt' in x else 0) for x in tr['fused_src']], np.int8),
                box=box)
            n_det += int(tr['detected'].sum())
        n_tracks += len(tracks)
        entry = dict(
            subdataset=L.subdataset(dataset, traj), n_frames=int(T),
            timestamps=np.asarray(res['timestamps'][:T], dtype=np.float64),
            robot=dict(position=rpos, yaw=yaw,
                       linear_vel=np.asarray(td['linear_vel'][:T], np.float32),
                       angular_vel=np.asarray(td['angular_vel'][:T], np.float32)),
            tracks=tracks)
        # re-linking (relink.py): duplicate merge + gap linking -> merged_tracks, merged_id per track
        merged, links, n_dup, n_amb = R.relink_entry(entry)
        merged_tracks = {}
        for m in merged:
            fr = m.frames
            n = len(fr)
            src = np.zeros(n, np.int8)
            box = np.full((n, 4), np.nan, np.float32)
            for tid in m.members:
                tr = tracks[tid]
                idx = tr['frames'] - fr[0]
                take = tr['detected'] & (src[idx] == 0)
                src[idx[take]] = tr['source'][take]
                box[idx[take]] = tr['box'][take]
                tracks[tid]['merged_id'] = int(m.tid)
            pos = m.pos.astype(np.float32)
            c, s = np.cos(yaw[fr])[:, None], np.sin(yaw[fr])[:, None]
            dxy = pos - rpos[fr]
            pos_robot = np.concatenate([c * dxy[:, :1] + s * dxy[:, 1:], -s * dxy[:, :1] + c * dxy[:, 1:]], 1)
            merged_tracks[int(m.tid)] = dict(
                frames=fr.astype(np.int32), pos_world=pos, vel_world=m.vel.astype(np.float32),
                pos_robot=pos_robot.astype(np.float32), pos_std=m.std.astype(np.float32),
                detected=m.det.astype(bool), source=src, box=box, members=np.array(m.members, np.int32))
        for tid in tracks:
            tracks[tid].setdefault('merged_id', tid)
        entry['merged_tracks'] = merged_tracks
        entry['relink'] = dict(n_links=len(links), n_duplicates=n_dup, n_ambiguous=n_amb,
                               links=np.array([(a, b, g) for a, b, g, _ in links], dtype=np.float32).reshape(-1, 3))
        trajs[traj] = entry
        n_merged += len(merged_tracks)
    out = os.path.join(L.OUT_ROOT, f'tracks_{dataset}.pkl')
    data = dict(dataset=dataset, format_version=3, layout='per_track', created=time.strftime('%Y-%m-%d %H:%M'),
                min_track_len=min_track_len, notes=__doc__, trajectories=trajs)
    with open(out, 'wb') as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    mb = os.path.getsize(out) / 1e6
    print(f'{dataset}: {len(trajs)} trajectories, {n_tracks} tracks ({n_merged} after re-linking), '
          f'{n_det} detections -> {out} ({mb:.0f} MB)')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', action='append', choices=list(L.DATASETS))
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--min-track-len', type=int, default=3)
    args = ap.parse_args()
    for ds in (list(L.DATASETS) if args.all or not args.dataset else args.dataset):
        export_dataset(ds, args.min_track_len)


if __name__ == '__main__':
    main()
