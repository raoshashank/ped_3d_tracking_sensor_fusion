import sys, os
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..")))  # repo root: p3d_env
import p3d_env  # noqa: E402  (P3D_* data roots: config/paths.env, see config/paths.env.example)
b=sys.argv[1]; O=f"{os.environ.get('P3D_SCRATCH', '/tmp')}/groups/{b}"
os.chdir(f"{p3d_env.P('P3D_BAGS_GO2NUS')}/{b}/slam_output/erasor2")
sys.argv=["ped_groups.py","--out-dir",O]+sys.argv[2:]
src=open(sys.argv.pop(1) if False else os.environ.get("PG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "template", "erasor2", "ped_groups.py"))).read().replace("HERE = Path(__file__).resolve().parent","HERE = Path.cwd()")
exec(compile(src,"ped_groups.py","exec"),{"__name__":"__main__"})
