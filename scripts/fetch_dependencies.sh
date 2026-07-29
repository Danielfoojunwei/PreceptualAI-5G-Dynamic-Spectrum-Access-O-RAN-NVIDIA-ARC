#!/bin/bash
# fetch_dependencies.sh — reproducible fetch of every external source
# dependency in third_party/manifest.json, without using git submodules.
#
# Clones each dependency into <target-dir> (default: ./oran-deps-src),
# checks out the EXACT pinned commit recorded in third_party/manifest.json,
# and applies the single local patch (A1 mediator policy_type_id decode fix).
#
# The same pins are also available as git submodules:
#     git submodule update --init
# This script exists for consumers who prefer plain clones (CI caches,
# air-gapped mirrors, etc.). Pin table below MUST stay in sync with
# third_party/manifest.json — CI compares them.
#
# Usage: scripts/fetch_dependencies.sh [target-dir]
set -euo pipefail

TARGET="${1:-oran-deps-src}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PATCH="$REPO_ROOT/deploy/xapp-e2e/patches/a1mediator-policy-resp-type.patch"

# name|url|pinned_commit  (mirror of third_party/manifest.json)
DEPS=(
  "ric-plt-lib-rmr|https://gerrit.o-ran-sc.org/r/ric-plt/lib/rmr|8b9a214906a40b338def981d5c16b4ea247176fb"
  "ric-plt-a1|https://gerrit.o-ran-sc.org/r/ric-plt/a1|09a757b4fd63198d8690d50b52bfd04552d47f1f"
  "ric-app-hw-python|https://gerrit.o-ran-sc.org/r/ric-app/hw-python|a6d00525aa2f62f2457d84731da2b7d89e0b1013"
  "sim-a1-interface|https://gerrit.o-ran-sc.org/r/sim/a1-interface|be2943f57211f62095dc5434099e331df237aafb"
  "ocudu|https://gitlab.com/ocudu/ocudu.git|f46f5804e53fede1e5c3353420ce1bcd3f4e57c6"
  "flexric|https://gitlab.eurecom.fr/mosaic5g/flexric.git|ef6d722f22191eea74089966983da1f5ec1fedd4"
)

mkdir -p "$TARGET"
TARGET="$(cd "$TARGET" && pwd)"

fetch_pin() { # dir url pin — materialise $pin in $dir, shallow when possible
    local dir="$1" url="$2" pin="$3"
    if [ ! -d "$dir/.git" ]; then
        # Try cheapest path first: shallow-fetch the exact commit (works on
        # servers with uploadpack.allowReachableSHA1InWant, e.g. GitLab).
        git init -q "$dir"
        git -C "$dir" remote add origin "$url"
        if ! git -C "$dir" fetch -q --depth 1 origin "$pin" 2>/dev/null; then
            # Server refuses SHA fetch (some gerrit configs): full fetch.
            git -C "$dir" fetch -q origin
        fi
    fi
    git -C "$dir" cat-file -e "${pin}^{commit}" 2>/dev/null \
        || git -C "$dir" fetch -q origin "$pin" \
        || git -C "$dir" fetch -q origin
    git -C "$dir" checkout -q --detach "$pin"
    echo "  pinned: $(git -C "$dir" rev-parse HEAD)"
}

for entry in "${DEPS[@]}"; do
    IFS='|' read -r name url pin <<< "$entry"
    echo "== $name ($url @ ${pin:0:12})"
    fetch_pin "$TARGET/$name" "$url" "$pin"
done

# Apply the one local patch: the A1 mediator's Consume() rejects the
# string-typed policy_type_id its own A1_POLICY_REQ produces; the patch
# accepts both encodings (see third_party/MANIFEST.md).
echo "== applying A1 mediator patch"
if git -C "$TARGET/ric-plt-a1" apply --check "$PATCH" 2>/dev/null; then
    git -C "$TARGET/ric-plt-a1" apply "$PATCH"
    echo "  applied: $(basename "$PATCH")"
elif git -C "$TARGET/ric-plt-a1" apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "  already applied: $(basename "$PATCH")"
else
    echo "  ERROR: patch does not apply cleanly" >&2
    exit 1
fi

echo
echo "All dependencies fetched to $TARGET at their pinned commits."
echo "Next: deploy/xapp-e2e/run_stack.sh builds and starts the A1 stack"
echo "(point XAPP_E2E_WORK=$TARGET to reuse these checkouts)."
