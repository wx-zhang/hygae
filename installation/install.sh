#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_NAVIGATION=false
INSTALL_PRIMITIVE_SKILL=false

usage() {
    cat <<'EOF'
Usage: bash installation/install.sh [options]

Options:
  --with-navigation       Install AI2-THOR for the Navigation environment.
  --with-primitive-skill  Install ManiSkill/SAPIEN for Primitive Skill.
  --all-envs              Install both optional environment dependency sets.
  -h, --help              Show this help message.
EOF
}

for arg in "$@"; do
    case "$arg" in
        --with-navigation) INSTALL_NAVIGATION=true ;;
        --with-primitive-skill) INSTALL_PRIMITIVE_SKILL=true ;;
        --all-envs)
            INSTALL_NAVIGATION=true
            INSTALL_PRIMITIVE_SKILL=true
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $arg" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "No active Conda environment was detected." >&2
    echo "Create and activate the HYGAE environment first:" >&2
    echo "  conda env create -f environment.yml" >&2
    echo "  conda activate hygae" >&2
    exit 2
fi
if [[ "${CONDA_DEFAULT_ENV:-}" == "base" ]]; then
    echo "Refusing to install into the Conda base environment." >&2
    echo "Run: conda env create -f environment.yml && conda activate hygae" >&2
    exit 2
fi

export PYTHONNOUSERSITE=1

python - <<'PY'
import sys

if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"HYGAE requires Python 3.10, found {sys.version}")
PY

echo "Installing the verified HYGAE training stack in: $CONDA_PREFIX"
python -m pip install -r "$ROOT/installation/requirements.txt"

if [[ "$INSTALL_NAVIGATION" == true ]]; then
    python -m pip install -r "$ROOT/installation/requirements-navigation.txt"
fi

if [[ "$INSTALL_PRIMITIVE_SKILL" == true ]]; then
    python -m pip install -r "$ROOT/installation/requirements-primitive-skill.txt"
fi

bash "$ROOT/installation/prepare_verl.sh"
python -m pip install -e "$ROOT/third_party/verl" --no-deps --no-build-isolation
python -m pip install -e "$ROOT" --no-deps --no-build-isolation

PYTHONPATH="$ROOT/third_party/verl:$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python "$ROOT/installation/preflight.py"
