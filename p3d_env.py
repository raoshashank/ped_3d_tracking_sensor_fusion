"""Machine-specific paths for this repo.

Every script takes its data roots and interpreters from P3D_* environment variables (list and meaning in
config/paths.env.example). Set them in the shell, or write them to config/paths.env (git-ignored): scripts
that live in the repo import this module, which loads that file into os.environ without overriding
variables that are already set, and exports P3D_REPO. Scripts that run_all.py copies into a bag directory
(pipeline/template/) read os.environ directly; they inherit the variables from run_all.py or from a shell
that sourced config/paths.env.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent
ENV_FILE = REPO / "config" / "paths.env"
_loaded = False


def load():
    """Load config/paths.env (KEY=VALUE lines, optional `export`, $VAR and ~ expanded) into os.environ."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            key, sep, val = line.partition("=")
            if not sep:
                continue
            val = val.split(" #")[0].strip().strip('"').strip("'")   # drop inline comments
            os.environ.setdefault(key.strip(), os.path.expandvars(os.path.expanduser(val)))
    os.environ.setdefault("P3D_REPO", str(REPO))


def P(name, default=None):
    """Value of a P3D_* variable (a path), or `default`; exits with a hint when neither is available."""
    load()
    v = os.environ.get(name) or default
    if not v:
        raise SystemExit(f"{name} is not set: export it, or copy config/paths.env.example to {ENV_FILE} and edit it")
    return v


def py(env):
    """Python interpreter of conda env `env`: P3D_PY_<ENV> if set, else $P3D_CONDA_ENVS/<env>/bin/python."""
    return os.environ.get(f"P3D_PY_{env.upper()}") or os.path.join(P("P3D_CONDA_ENVS"), env, "bin", "python")


load()
