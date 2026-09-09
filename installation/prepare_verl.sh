#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_DIR="$ROOT/third_party/verl"
PATCH="$ROOT/installation/verl.patch"
VERL_URL=https://github.com/volcengine/verl.git
BASE_COMMIT=329dcfe1dd60f2d736ee55914e2a49e1887718eb
PATCH_SHA256=145dcd694151570dd8356625bf0549913c82ed95d3e6340c02baa4baf7476ea9
CORE_SHA256=7685d62b1510b5a8cace42cadc793618d2abed2c255e2d4e41df73fdaad3b203
INIT_SHA256=a64142f777f18ba0d6631d05936c5a778961f985ecdab3fe062797db568ac9a9
ACTOR_SHA256=625d2679eb9bcce0b4672602a040a7dc7ec797479cb3a6026496fb7060b1c7c5
PYPROJECT_SHA256=4412019742f12050e62b82959a81e3ac6f36835bacb1efc27942619b4310fc4a
SETUP_SHA256=93f59a8990f6bb2da1e42a56ee3dd8bdf9bfb90857e45f6bd970e98e4ba98542

actual_patch_sha256="$(sha256sum "$PATCH" | awk '{print $1}')"
if [[ "$actual_patch_sha256" != "$PATCH_SHA256" ]]; then
    echo "VERL patch hash mismatch: $actual_patch_sha256" >&2
    exit 1
fi

if [[ ! -d "$VERL_DIR/.git" ]]; then
    if [[ -e "$VERL_DIR" ]]; then
        echo "$VERL_DIR exists but is not a Git checkout; move it and retry." >&2
        exit 1
    fi
    mkdir -p "$(dirname "$VERL_DIR")"
    git clone --filter=blob:none --no-checkout "$VERL_URL" "$VERL_DIR"
    git -C "$VERL_DIR" fetch --depth 1 origin "$BASE_COMMIT"
    git -C "$VERL_DIR" checkout --detach "$BASE_COMMIT"
fi

actual_head="$(git -C "$VERL_DIR" rev-parse HEAD)"
if [[ "$actual_head" != "$BASE_COMMIT" ]]; then
    echo "Expected official VERL $BASE_COMMIT, found $actual_head" >&2
    echo "Remove or move $VERL_DIR, then run the installer again." >&2
    exit 1
fi

if git -C "$VERL_DIR" apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "VERL patch already applied"
elif git -C "$VERL_DIR" apply --check "$PATCH" 2>/dev/null; then
    git -C "$VERL_DIR" apply "$PATCH"
else
    echo "VERL has changes that conflict with installation/verl.patch" >&2
    exit 1
fi

verify_hash() {
    local expected="$1"
    local file="$2"
    local actual
    actual="$(sha256sum "$file" | awk '{print $1}')"
    [[ "$actual" == "$expected" ]] || {
        echo "Patched VERL hash mismatch for $file: $actual" >&2
        exit 1
    }
}

verify_hash "$CORE_SHA256" "$VERL_DIR/verl/trainer/ppo/core_algos.py"
verify_hash "$INIT_SHA256" "$VERL_DIR/verl/utils/value_head_init.py"
verify_hash "$ACTOR_SHA256" "$VERL_DIR/verl/workers/actor/dp_actor.py"
verify_hash "$PYPROJECT_SHA256" "$VERL_DIR/pyproject.toml"
verify_hash "$SETUP_SHA256" "$VERL_DIR/setup.py"
git -C "$VERL_DIR" diff --check
echo "Prepared official VERL $BASE_COMMIT with installation/verl.patch"
