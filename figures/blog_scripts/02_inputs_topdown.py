#!/usr/bin/env python3
"""Figure 02: top-down, odometry frame: robot odometry trajectory, one lidar scan (360 deg), camera FOV."""
import numpy as np

from _camera_common import C, FIG, FRAME, draw_fov, fov_half_angle_and_yaw, lidar_in_base, plt, traj

T = traj()
k = FRAME
pb = lidar_in_base(T, k)
pb = pb[(np.linalg.norm(pb[:, :2], axis=1) < 30) & (pb[:, 2] > -0.25) & (pb[:, 2] < 1.8)]   # drop ground returns
half, cyaw, cxy = fov_half_angle_and_yaw(T)
ang = np.arctan2(pb[:, 1] - cxy[1], pb[:, 0] - cxy[0])
in_fov = np.abs(np.angle(np.exp(1j * (ang - cyaw)))) < half
w = T.base_to_world(k, pb[:, :2])
pos, yaw = T.robot_pos[:T.n], T.robot_yaw[:T.n]

fig, axs = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw=dict(width_ratios=[1.25, 1]))
for a, zoom in zip(axs, (False, True)):
    c = T.base_to_world(k, cxy[None])[0]
    draw_fov(a, c, yaw[k] + cyaw, half, 12, color="gold", alpha=0.3, label="camera FOV", zorder=0)
    a.scatter(w[~in_fov, 0], w[~in_fov, 1], s=2, c=C["lidar_out"], label="lidar, outside camera FOV")
    a.scatter(w[in_fov, 0], w[in_fov, 1], s=2, c=C["lidar_in"], label="lidar, inside camera FOV")
    a.plot(pos[:, 0], pos[:, 1], "k-", lw=1.5, label="robot odometry trajectory")
    a.annotate("", xy=pos[k] + 2.5 * np.array([np.cos(yaw[k]), np.sin(yaw[k])]), xytext=pos[k],
               arrowprops=dict(arrowstyle="-|>", color="k", lw=2.5, mutation_scale=22), zorder=5)
    a.plot(*pos[k], "ko", ms=7, zorder=5)
    a.set_aspect("equal")
    a.set_xlabel("x [m] (odometry frame)"); a.set_ylabel("y [m]")
    if zoom:
        a.set_xlim(pos[k, 0] - 12, pos[k, 0] + 14); a.set_ylim(pos[k, 1] - 12, pos[k, 1] + 12)
        a.set_title("(b) robot (arrow = heading) at t = %.1f s" % (T.timestamps[k] - T.timestamps[0]))
    else:
        a.set_title("(a) odometry + one lidar scan (points 0.1-2.2 m above ground)")
        a.legend(loc="lower left", markerscale=5)
fig.tight_layout()
fig.savefig(FIG / "02_inputs_topdown.png", bbox_inches="tight")
print("saved; scan pts", len(pb), "in FOV", in_fov.sum(), "cam half-FOV deg", round(np.degrees(half), 1))
