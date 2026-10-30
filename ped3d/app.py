"""Streamlit viewer for pedestrian 3D position estimates (ped3d_lib).

    cd <repo>/ped3d
    $P3D_CONDA_ENVS/calib/bin/python -m streamlit run app.py

Left: camera frame with track boxes, person mask, projected lidar (coloured by depth) and the
per-source depths; optional VGGT-Omega depth panel. Right: bird's-eye view (world/odom frame or
robot frame) with the per-source raw estimates for the current frame, smoothed fused tracks with
trails, and the robot path. Bottom: per-track depth/position time series and metrics.
"""
import os

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ped3d_lib as L

st.set_page_config(page_title='ped3d viewer', layout='wide')

PALETTE = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6',
           '#bfef45', '#fabed4', '#469990', '#dcbeff', '#9A6324', '#800000', '#aaffc3',
           '#808000', '#ffd8b1', '#000075', '#ffe119']
SRC_STYLE = {  # source -> (label, marker symbol)
    'lidar': ('lidar', 'x'),
    'lidar_g': ('lidar (VGGT-guided cluster)', 'cross'),
    'vggt': ('VGGT raw', 'circle-open'),
    'vggt_c': ('VGGT + range corr.', 'diamond-open'),
    'vggt_a': ('VGGT + corr. + track anchor', 'diamond'),
    'da': ('DepthAnything (frame-scaled)', 'triangle-up-open'),
}


# Scatter3d only supports a few symbols
SYMBOL_3D = {'lidar': 'x', 'lidar_g': 'cross', 'vggt': 'circle-open', 'vggt_c': 'diamond-open',
             'vggt_a': 'diamond', 'da': 'square-open'}


def tcolor(tid):
    return PALETTE[tid % len(PALETTE)]


def hex2bgr(h):
    return tuple(int(h[i:i + 2], 16) for i in (5, 3, 1))


@st.cache_resource(show_spinner='estimating pedestrian positions...')
def get_result(dataset, traj):
    return L.load_or_process(dataset, traj)


@st.cache_resource
def get_traj(dataset, traj):
    return L.Traj(dataset, traj)


@st.cache_data
def traj_list(dataset):
    done = set()
    d = os.path.join(L.OUT_ROOT, dataset)
    if os.path.isdir(d):
        done = {f[:-4] for f in os.listdir(d) if f.endswith('.pkl')}
    trajs = L.list_trajs(dataset)
    return trajs, done


def render_frame(T, res, i, show_mask, show_lidar, label_src):
    img = T.image(i)
    if img is None:
        return None
    ov = img.copy()
    if show_mask:
        m = T.mask(i)
        if m is not None:
            ov[m] = (0.55 * ov[m] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)
    if show_lidar:
        lid = T.lidar(i)
        if lid is not None:
            pc = (T.cam_T_lidar[:3, :3] @ lid.T).T + T.cam_T_lidar[:3, 3]
            u, v, z, _ = L.project_cam(pc, T.K, T.W, T.H)
            col = cv2.applyColorMap((np.clip(z / 12.0, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, 0, :]
            for uu, vv, c in zip(u, v, col):
                cv2.circle(ov, (int(uu), int(vv)), 1, tuple(int(x) for x in c), -1)
    for inst in res['frames'][i]:
        x, y, w, h = inst['box'].astype(int)
        c = hex2bgr(tcolor(inst['id']))
        cv2.rectangle(ov, (x, y), (x + w, y + h), c, 2)
        parts = [f"#{inst['id']}"]
        for s in label_src:
            if s == 'vggt_a':   # anchored depth is a track-level quantity
                d = inst['depth_vggt_c'] * res['tracks'][inst['id']]['anchor']
            else:
                d = inst['depth_' + s]
            if np.isfinite(d):
                parts.append(f"{SRC_STYLE[s][0].split()[0][:5]} {d:.1f}")
        if inst['n_lidar']:
            parts.append(f"n={inst['n_lidar']}")
        txt = ' '.join(parts)
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(y - 4, th + 2)
        cv2.rectangle(ov, (x, ty - th - 2), (x + tw + 2, ty + 2), (0, 0, 0), -1)
        cv2.putText(ov, txt, (x + 1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
    return cv2.cvtColor(ov, cv2.COLOR_BGR2RGB)


def depth_panel(T, i, kind):
    d = T.depth(i, kind)
    if d is None:
        return None
    dv = cv2.applyColorMap((np.clip(d / 12.0, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    dv[d <= 0] = 0
    return cv2.cvtColor(dv, cv2.COLOR_BGR2RGB)


def bev_figure(T, res, i, sources, robot_frame, trail, max_range, sel_tid, show_interp):
    fig = go.Figure()
    tf = (lambda p: T.world_to_base(i, p)) if robot_frame else (lambda p: np.asarray(p))
    # robot path + pose
    rp = tf(res['robot_pos'])
    fig.add_trace(go.Scatter(x=rp[:, 0], y=rp[:, 1], mode='lines', line=dict(color='gray', width=1),
                             name='robot path', hoverinfo='skip'))
    yaw = 0.0 if robot_frame else res['robot_yaw'][i]
    r0 = tf(res['robot_pos'][i:i + 1])[0]
    tri = np.array([[0.35, 0], [-0.2, 0.2], [-0.2, -0.2], [0.35, 0]])
    Rm = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    tri = (Rm @ tri.T).T + r0
    fig.add_trace(go.Scatter(x=tri[:, 0], y=tri[:, 1], fill='toself', mode='lines',
                             line=dict(color='black'), fillcolor='black', name='robot'))
    th = np.linspace(0, 2 * np.pi, 90)
    fig.add_trace(go.Scatter(x=r0[0] + max_range * np.cos(th), y=r0[1] + max_range * np.sin(th),
                             mode='lines', line=dict(color='lightgray', dash='dot'), name=f'{max_range:.0f} m',
                             hoverinfo='skip'))
    # tracks
    for tid, tr in res['tracks'].items():
        f0 = tr['frames'][0]
        k = i - f0
        if k < 0 or k >= len(tr['frames']):
            continue
        col = tcolor(tid)
        wide = 3 if tid == sel_tid else 1.5
        lo, hi = max(0, k - trail), min(len(tr['frames']), k + trail + 1)
        xs = tr['fused'][lo:hi, :2]
        ok = np.isfinite(xs).all(1)
        if ok.sum() >= 1:
            p = tf(xs[ok])
            interp = tr['interpolated'][lo:hi][ok]
            fig.add_trace(go.Scatter(x=p[:, 0], y=p[:, 1], mode='lines', line=dict(color=col, width=wide),
                                     name=f'#{tid} fused', legendgroup=f'#{tid}', showlegend=False,
                                     hoverinfo='skip', opacity=0.7))
            if show_interp and interp.any():
                fig.add_trace(go.Scatter(x=p[interp, 0], y=p[interp, 1], mode='markers',
                                         marker=dict(color=col, size=5, symbol='circle-open-dot'),
                                         name=f'#{tid} interpolated', legendgroup=f'#{tid}', showlegend=False,
                                         hoverinfo='skip'))
        cur = tr['fused'][k, :2]
        if np.isfinite(cur).all():
            p = tf(cur[None])[0]
            hover = f"#{tid} fused<br>src {tr['fused_src'][k] or 'interp'}<br>anchor {tr['anchor']:.3f}"
            if tr['outlier'][k]:
                hover += '<br>OUTLIER (down-weighted)'
            fig.add_trace(go.Scatter(x=[p[0]], y=[p[1]], mode='markers+text', text=[f'#{tid}'],
                                     textposition='top center', textfont=dict(size=10, color=col),
                                     marker=dict(color=col, size=11 if tid == sel_tid else 8,
                                                 line=dict(color='black', width=1)),
                                     name=f'#{tid}', legendgroup=f'#{tid}', hovertext=hover, hoverinfo='text'))
        for s in sources:
            m = tr['world'][s][k]
            if np.isfinite(m).all():
                p = tf(m[None])[0]
                fig.add_trace(go.Scatter(x=[p[0]], y=[p[1]], mode='markers',
                                         marker=dict(color=col, size=8, symbol=SRC_STYLE[s][1],
                                                     line=dict(color=col, width=1.5)),
                                         name=SRC_STYLE[s][0], legendgroup=s, showlegend=(tid == sel_tid),
                                         hovertext=f"#{tid} {SRC_STYLE[s][0]}<br>depth {tr['depth'][s][k]:.2f} m",
                                         hoverinfo='text'))
    span = max_range + 2
    if robot_frame:
        xr, yr = [-3, span], [-span, span]
        fig.update_xaxes(title='x forward [m]'); fig.update_yaxes(title='y left [m]')
    else:
        xr, yr = [r0[0] - span, r0[0] + span], [r0[1] - span, r0[1] + span]
        fig.update_xaxes(title='x world [m]'); fig.update_yaxes(title='y world [m]')
    fig.update_xaxes(range=xr)
    fig.update_yaxes(range=yr, scaleanchor='x', scaleratio=1)
    # keep zoom/pan across frames; in world frame the view follows the robot unless the user zoomed
    fig.update_layout(height=620, margin=dict(l=10, r=10, t=10, b=10), legend=dict(font=dict(size=10)),
                      uirevision=f'bev-{"robot" if robot_frame else "world"}-{max_range}')
    return fig


def track_figure(T, res, tid, i, sources):
    tr = res['tracks'][tid]
    t = tr['t'] - res['timestamps'][0]
    fig = go.Figure()
    for s in sources:
        d = tr['depth'][s]
        fig.add_trace(go.Scatter(x=t, y=d, mode='markers', marker=dict(symbol=SRC_STYLE[s][1], size=7),
                                 name=SRC_STYLE[s][0]))
    # fused: distance from the robot at each frame (comparable to camera depth)
    fr = np.full(len(t), np.nan)
    for k, f in enumerate(tr['frames']):
        p = tr['fused'][k, :2]
        if np.isfinite(p).all():
            fr[k] = np.linalg.norm(T.world_to_base(f, p[None])[0])
    fig.add_trace(go.Scatter(x=t, y=fr, mode='lines', line=dict(color='black', width=2), name='fused (range)'))
    ip = tr['interpolated']
    if ip.any():
        fig.add_trace(go.Scatter(x=t[ip], y=fr[ip], mode='markers', marker=dict(color='black', symbol='circle-open', size=8),
                                 name='interpolated'))
    if tr['outlier'].any():
        o = tr['outlier']
        fig.add_trace(go.Scatter(x=t[o], y=fr[o], mode='markers', marker=dict(color='red', symbol='x', size=10),
                                 name='down-weighted meas.'))
    fig.add_vline(x=res['timestamps'][i] - res['timestamps'][0], line=dict(color='gray', dash='dash'))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), xaxis_title='t [s]',
                      yaxis_title='depth / range from robot [m]', title=f'track #{tid}  (anchor ratio {tr["anchor"]:.3f})',
                      uirevision=f'track-{tid}')
    return fig


@st.cache_data(max_entries=64, show_spinner=False)
def frame_points(dataset, traj, i):
    """Per-instance 3D points behind each estimate (base frame) for the 3D panel."""
    T = get_traj(dataset, traj)
    return L.estimate_frame(T, i, with_da=False, keep_points=True)


def scene3d_figure(T, res, i, insts, sources, max_range, sel_tid, show_vggt_pts, show_lidar_sel,
                   cloud_pts, colour_by):
    fig = go.Figure()
    lid = T.lidar(i)
    if lid is not None:
        p = (T.base_T_lidar[:3, :3] @ lid.T).T + T.base_T_lidar[:3, 3]
        keep = (p[:, 0] > -2) & (p[:, 0] < max_range) & (np.abs(p[:, 1]) < max_range) & (p[:, 2] > -1.0) & (p[:, 2] < 3.0)
        p = p[keep]
        if len(p) > cloud_pts:
            p = p[np.random.RandomState(0).choice(len(p), cloud_pts, replace=False)]
        if colour_by == 'height':
            col = dict(color=p[:, 2], colorscale='Greys_r', cmin=-1, cmax=3, size=1.6, opacity=0.6)
        else:
            col = dict(color=np.linalg.norm(p[:, :2], axis=1), colorscale='Turbo', cmin=0, cmax=max_range, size=1.6, opacity=0.7)
        fig.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode='markers', marker=col,
                                   name='lidar cloud', hoverinfo='skip'))
    # robot + camera frustum
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(color='black', size=6, symbol='diamond'),
                               name='robot (base)'))
    c0 = T.base_T_cam[:3, 3]
    corners = []
    for u, v in ((0, 0), (T.W, 0), (T.W, T.H), (0, T.H)):
        d = np.array([(u - T.K[0, 2]) / T.K[0, 0], (v - T.K[1, 2]) / T.K[1, 1], 1.0]) * 1.5
        corners.append(T.base_T_cam[:3, :3] @ d + c0)
    fx, fy, fz = [], [], []
    for k in range(4):
        a, b = corners[k], corners[(k + 1) % 4]
        fx += [c0[0], a[0], b[0], None]; fy += [c0[1], a[1], b[1], None]; fz += [c0[2], a[2], b[2], None]
    fig.add_trace(go.Scatter3d(x=fx, y=fy, z=fz, mode='lines', line=dict(color='black', width=2), name='camera', hoverinfo='skip'))

    for inst in insts:
        tid = inst['id']
        col = tcolor(tid)
        big = tid == sel_tid
        if show_vggt_pts and 'pts_vggt' in inst:
            q = inst['pts_vggt']
            fig.add_trace(go.Scatter3d(x=q[:, 0], y=q[:, 1], z=q[:, 2], mode='markers',
                                       marker=dict(color=col, size=2.2 if big else 1.8, opacity=0.55),
                                       name=f'#{tid} VGGT body pts', legendgroup=f'#{tid}', hoverinfo='skip'))
        if show_lidar_sel:
            key = 'pts_lidar_g' if 'pts_lidar_g' in inst else 'pts_lidar'
            if key in inst:
                q = inst[key]
                fig.add_trace(go.Scatter3d(x=q[:, 0], y=q[:, 1], z=q[:, 2], mode='markers',
                                           marker=dict(color=col, size=4, symbol='diamond', line=dict(color='black', width=1)),
                                           name=f'#{tid} lidar pts used ({len(q)})', legendgroup=f'#{tid}'))
        for s in sources:
            p = inst.get('pos_' + s) if s != 'vggt_a' else None
            if s == 'vggt_a' and np.isfinite(inst['pos_vggt_c']).all():
                a = res['tracks'][tid]['anchor']
                p = inst['pos_vggt_c'] * np.array([a, a, 1.0])
            if p is None or not np.isfinite(p).all():
                continue
            fig.add_trace(go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode='markers',
                                       marker=dict(color=col, size=7 if big else 5, symbol=SYMBOL_3D[s],
                                                   line=dict(color='black', width=1)),
                                       name=SRC_STYLE[s][0], legendgroup=s, showlegend=(inst is insts[0]),
                                       hovertext=f"#{tid} {SRC_STYLE[s][0]}<br>x {p[0]:.2f} y {p[1]:.2f} z {p[2]:.2f}", hoverinfo='text'))
        # fused position as a vertical pole (floor to head height)
        tr = res['tracks'][tid]
        k = i - tr['frames'][0]
        fp = tr['fused'][k, :2] if 0 <= k < len(tr['frames']) else None
        if fp is not None and np.isfinite(fp).all():
            b = T.world_to_base(i, fp[None])[0]
            zf = float(np.nanmin([inst['pos_vggt_c'][2], inst['pos_lidar_g'][2], 0.8])) - 0.8
            fig.add_trace(go.Scatter3d(x=[b[0], b[0]], y=[b[1], b[1]], z=[zf, zf + 1.7], mode='lines+text',
                                       line=dict(color=col, width=8 if big else 5), text=['', f'#{tid}'],
                                       textposition='top center', textfont=dict(color=col, size=11),
                                       name=f'#{tid} fused', legendgroup=f'#{tid}',
                                       hovertext=f"#{tid} fused<br>x {b[0]:.2f} y {b[1]:.2f}<br>src {tr['fused_src'][k] or 'interp'}",
                                       hoverinfo='text'))
    # uirevision keeps the user's camera across frame changes (only resets when range changes)
    fig.update_layout(height=700, margin=dict(l=0, r=0, t=0, b=0), legend=dict(font=dict(size=10)),
                      uirevision=f'scene-{max_range}',
                      scene=dict(uirevision=f'scene-{max_range}',
                                 xaxis=dict(title='x fwd [m]', range=[-2, max_range]),
                                 yaxis=dict(title='y left [m]', range=[-max_range, max_range]),
                                 zaxis=dict(title='z [m]', range=[-1, 3]),
                                 aspectmode='manual',
                                 aspectratio=dict(x=1, y=2 * max_range / (max_range + 2), z=4 / (max_range + 2)),
                                 camera=dict(eye=dict(x=-1.3, y=-1.6, z=0.9), up=dict(x=0, y=0, z=1))))
    return fig


@st.cache_data(show_spinner='collecting pedestrian positions...')
def dataset_positions(dataset, n_pkls, detected_only):
    return L.collect_positions(dataset, max_range=16.0, detected_only=detected_only)


def heatmap_figure(groups, names, bin_m, max_range, log_scale):
    from plotly.subplots import make_subplots
    x_range, y_range = (-4.0, max_range), (-max_range * 0.75, max_range * 0.75)
    maps = {g: L.heatmap(groups[g]['xy'], groups[g]['n_frames'], bin_m, x_range, y_range) for g in names}
    vmax = max(m[0].max() for m in maps.values()) or 1.0
    ncol = min(3, len(names))
    nrow = int(np.ceil(len(names) / ncol))
    titles = [f"{g} — {groups[g]['n_trajs']} traj, {len(groups[g]['xy']) / max(groups[g]['n_frames'], 1):.2f} peds/frame"
              for g in names]
    fig = make_subplots(rows=nrow, cols=ncol, subplot_titles=titles, horizontal_spacing=0.06, vertical_spacing=0.10)
    th = np.linspace(0, 2 * np.pi, 80)
    for k, g in enumerate(names):
        H, xe, ye = maps[g]
        r, c = k // ncol + 1, k % ncol + 1
        Z = np.log10(np.maximum(H, vmax / 1000.0)) if log_scale else H
        fig.add_trace(go.Heatmap(z=Z.T, x=(xe[:-1] + xe[1:]) / 2, y=(ye[:-1] + ye[1:]) / 2, colorscale='Inferno',
                                 zmin=np.log10(vmax / 1000.0) if log_scale else 0, zmax=np.log10(vmax) if log_scale else vmax,
                                 showscale=(k == 0), colorbar=dict(title='log10 peds/frame/m²' if log_scale else 'peds/frame/m²'),
                                 hovertemplate='x %{x:.2f} y %{y:.2f}<br>%{z:.3f}<extra></extra>'), row=r, col=c)
        for rad in (2, 5, 10):
            fig.add_trace(go.Scatter(x=rad * np.cos(th), y=rad * np.sin(th), mode='lines', line=dict(color='white', width=0.7),
                                     opacity=0.5, showlegend=False, hoverinfo='skip'), row=r, col=c)
        fig.add_trace(go.Scatter(x=[0, 1.2], y=[0, 0], mode='lines+markers', line=dict(color='cyan', width=3),
                                 marker=dict(size=[10, 4], color='cyan'), showlegend=False, hoverinfo='skip'), row=r, col=c)
        fig.update_xaxes(title_text='x forward [m]', range=list(x_range), row=r, col=c)
        fig.update_yaxes(title_text='y left [m]', range=list(y_range), scaleanchor=f'x{k + 1 if k else ""}', scaleratio=1, row=r, col=c)
    fig.update_layout(height=380 * nrow + 40, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def metrics_table(m):
    rows = []
    for s, lab in [('vggt', 'VGGT raw'), ('vggt_c', 'VGGT + corr.'), ('vggt_a', 'VGGT + corr. + anchor'),
                   ('da', 'DepthAnything'), ('lidar_g', 'lidar (VGGT-guided)')]:
        rows.append(dict(source=lab, **{'vs lidar median [m]': m[f'agree_lidar_{s}_med'],
                                        'vs lidar p90 [m]': m[f'agree_lidar_{s}_p90'],
                                        'pairs': m[f'agree_lidar_{s}_n'],
                                        'resid. to fused median [m]': m[f'resid_{s}_med']}))
    rows.append(dict(source='lidar', **{'vs lidar median [m]': 0.0, 'vs lidar p90 [m]': 0.0, 'pairs': m['agree_lidar_vggt_n'],
                                        'resid. to fused median [m]': m['resid_lidar_med']}))
    return pd.DataFrame(rows).set_index('source')


# ----------------------------------------------------------------------------- UI
with st.sidebar:
    st.title('ped3d')
    dataset = st.selectbox('dataset', list(L.DATASETS), index=0)
    trajs, done = traj_list(dataset)
    if not trajs:
        st.error('no trajectories with VGGT depth + tracks + extrinsics')
        st.stop()
    only_done = st.checkbox('only already-processed trajectories', value=bool(done))
    opts = [t for t in trajs if t in done] if only_done and done else trajs
    traj = st.selectbox('trajectory', opts, format_func=lambda t: ('✓ ' if t in done else '· ') + t)
    res = get_result(dataset, traj)
    T = get_traj(dataset, traj)
    n = res['n']
    if 'frame' not in st.session_state:
        st.session_state.frame = 0
    st.session_state.frame = min(st.session_state.frame, n - 1)
    b1, b2 = st.columns(2)
    if b1.button('◀ prev', use_container_width=True):
        st.session_state.frame = max(0, st.session_state.frame - 1)
    if b2.button('next ▶', use_container_width=True):
        st.session_state.frame = min(n - 1, st.session_state.frame + 1)
    frame = st.slider('frame', 0, n - 1, key='frame')
    st.caption(f't = {res["timestamps"][frame] - res["timestamps"][0]:.2f} s   ({n} frames, '
               f'{len(res["tracks"])} tracks)')
    st.markdown('**sources shown**')
    sources = [s for s in SRC_STYLE if st.checkbox(SRC_STYLE[s][0], value=s in ('lidar_g', 'vggt_a'), key='src_' + s)]
    robot_frame = st.radio('BEV frame', ['robot', 'world (odom)'], horizontal=True) == 'robot'
    trail = st.slider('trail [frames each way]', 0, 40, 12)
    max_range = st.slider('range ring [m]', 5, 30, 10)
    show_interp = st.checkbox('mark interpolated frames', value=True)
    show_mask = st.checkbox('person mask overlay', value=True)
    show_lidar = st.checkbox('projected lidar', value=True)
    show_depth = st.checkbox('VGGT-Omega depth panel', value=False)
    active = [tid for tid, tr in res['tracks'].items() if tr['frames'][0] <= frame <= tr['frames'][-1]]
    sel_tid = st.selectbox('track for time series', active, format_func=lambda t: f'#{t}') if active else None
    st.markdown('**3D panel**')
    show_3d = st.checkbox('show 3D lidar view', value=True)
    show_vggt_pts = st.checkbox('VGGT body points (dense)', value=True)
    show_lidar_sel = st.checkbox('lidar points used per person', value=True)
    cloud_pts = st.select_slider('lidar cloud points', [5000, 10000, 20000, 40000], value=20000)
    colour_by = st.radio('cloud colour', ['height', 'range'], horizontal=True)

img = render_frame(T, res, frame, show_mask, show_lidar, [s for s in ('lidar_g', 'lidar', 'vggt_a') if s in sources][:2] or ['vggt_c'])
c1, c2 = st.columns([1.15, 1])
with c1:
    if img is not None:
        st.image(img, use_container_width=True)
    if show_depth:
        dp = depth_panel(T, frame, 'vggt_omega_depth')
        if dp is not None:
            st.image(dp, caption='VGGT-Omega metric depth (0-12 m)', use_container_width=True)
    if res['frames'][frame]:
        rows = []
        for inst in res['frames'][frame]:
            rows.append({'id': inst['id'], 'lidar n': inst['n_lidar'], 'lidar d': inst['depth_lidar'],
                         'lidar guided d': inst['depth_lidar_g'],
                         'vggt d': inst['depth_vggt'], 'vggt corr.': inst['depth_vggt_c'],
                         'DA d': inst['depth_da'], 'vggt frac': inst['vggt_frac'],
                         'flags': ','.join(inst['flags'])})
        st.dataframe(pd.DataFrame(rows).set_index('id').style.format(precision=2), height=min(40 + 35 * len(rows), 300))
with c2:
    st.plotly_chart(bev_figure(T, res, frame, sources, robot_frame, trail, max_range, sel_tid, show_interp),
                    use_container_width=True, key='bev')

if show_3d:
    st.markdown('**3D view (base frame)** — grey/turbo: lidar cloud; small coloured dots: VGGT body points; '
                'diamonds: lidar points used for that person; poles: fused position; markers: per-source positions')
    insts3d = frame_points(dataset, traj, frame)
    st.plotly_chart(scene3d_figure(T, res, frame, insts3d, sources, max_range, sel_tid, show_vggt_pts,
                                   show_lidar_sel, cloud_pts, colour_by), use_container_width=True, key='scene3d')

if sel_tid is not None:
    st.plotly_chart(track_figure(T, res, sel_tid, frame, sources), use_container_width=True, key='track')

with st.expander('pedestrian distribution around the robot (heatmaps per sub-dataset)', expanded=False):
    hc1, hc2, hc3, hc4 = st.columns(4)
    h_ds = hc1.selectbox('dataset', list(L.DATASETS), index=list(L.DATASETS).index(dataset), key='heat_ds')
    h_bin = hc2.select_slider('bin [m]', [0.25, 0.5, 1.0], value=0.5)
    h_range = hc3.slider('extent [m]', 6, 16, 12)
    h_log = hc4.checkbox('log colour scale', value=True)
    h_det = hc4.checkbox('detected frames only (no interpolation)', value=True)
    ds_dir = os.path.join(L.OUT_ROOT, h_ds)
    n_pkls = len([f for f in os.listdir(ds_dir) if f.endswith('.pkl')]) if os.path.isdir(ds_dir) else 0
    groups = dataset_positions(h_ds, n_pkls, h_det)
    if not groups:
        st.info(f'no processed trajectories for {h_ds} yet — run run_ped3d.py --dataset {h_ds}')
    else:
        names = ['ALL'] + sorted(g for g in groups if g != 'ALL')
        chosen = st.multiselect('groups', names, default=names[:9], key='heat_groups')
        if chosen:
            st.plotly_chart(heatmap_figure(groups, chosen, h_bin, h_range, h_log), use_container_width=True, key='heat')
        st.caption(f'{n_pkls} processed trajectories; colour = expected pedestrians per frame per m² '
                   f'(fused VGGT+lidar estimates, robot frame: x forward, y left; rings at 2/5/10 m). '
                   f'Static PNGs: ped_heatmap.py --dataset {h_ds}')

with st.expander('trajectory metrics (within 10 m; lidar with ≥10 pts as reference)', expanded=False):
    m = res['metrics']
    st.write(f"{m['n_det']} detections within {L.EVAL_RANGE:.0f} m, lidar coverage {m['cov_lidar']}/{m['n_det']}, "
             f"VGGT coverage {m['cov_vggt']}/{m['n_det']}, fused {m['cov_fused']}/{m['n_det']}; "
             f"{m['n_interp']} interpolated frames, {m['n_outlier']} down-weighted measurements")
    st.dataframe(metrics_table(m).style.format(precision=3))
    csv_path = os.path.join(L.OUT_ROOT, dataset, '_metrics.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        st.markdown(f'**dataset aggregate** ({len(df)} trajectories, {int(df.n_det.sum())} detections)')
        agg = {}
        for s in ('vggt', 'vggt_c', 'vggt_a', 'da'):
            w = df[f'agree_lidar_{s}_n']
            agg[SRC_STYLE[s][0]] = dict(**{'vs lidar median [m] (weighted)': float((df[f'agree_lidar_{s}_med'] * w).sum() / w.sum()),
                                           'vs lidar p90 [m] (weighted)': float((df[f'agree_lidar_{s}_p90'] * w).sum() / w.sum())})
        st.dataframe(pd.DataFrame(agg).T.style.format(precision=3))
