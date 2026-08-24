#!/usr/bin/env bash
# Download and unpack the versioned acolite-mp regression test fixtures from S3.
#
# Usage:
#   AWS_PROFILE=<your-sso-profile> scripts/download_fixtures.sh [--force]
#
# Env vars:
#   AWS_PROFILE          Required. AWS SSO profile with access to the fixtures bucket.
#   ACOLITE_FIXTURES_DIR Optional. Overrides the default sibling fixtures directory.
set -euo pipefail

FORCE=0
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    *) echo "Unknown argument: $arg" >&2; exit 1 ;;
  esac
done

if [[ -z "${AWS_PROFILE:-}" ]]; then
  echo "Error: AWS_PROFILE must be set to an SSO profile with access to the fixtures bucket." >&2
  echo "Example: AWS_PROFILE=my-org-profile $0" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VERSION="$(tr -d '[:space:]' < "$REPO_ROOT/FIXTURES_VERSION")"

FIXTURES_BASE_DIR="${ACOLITE_FIXTURES_DIR:-$REPO_ROOT/../acolite-mp-fixtures}"
TARGET_DIR="$FIXTURES_BASE_DIR/$VERSION"
S3_URI="s3://adias-prod-dc-data-projects/csa-disr/acolite-mp/acolite-mp-fixtures_${VERSION}.zip"

if [[ -d "$TARGET_DIR" && "$FORCE" -eq 0 ]]; then
  echo "Fixtures already present at $TARGET_DIR (use --force to re-download)."
  exit 0
fi

echo "Downloading $S3_URI (profile: $AWS_PROFILE) ..."
mkdir -p "$FIXTURES_BASE_DIR"
TMP_ZIP="$(mktemp -t acolite-mp-fixtures-XXXXXX.zip)"
trap 'rm -f "$TMP_ZIP"' EXIT

aws s3 cp "$S3_URI" "$TMP_ZIP" --profile "$AWS_PROFILE"

rm -rf "$TARGET_DIR"
mkdir -p "$TARGET_DIR"
unzip -q "$TMP_ZIP" -d "$TARGET_DIR"

# Some zip snapshots incorrectly nest fixture data one level down under an extra
# legacy date directory (e.g. 20250600) instead of directly under $VERSION;
# detect that by checking for the known sensor directories and flatten if needed.
KNOWN_SENSOR_DIRS=(s2_original s3_original landsat9_original landsat9_acolite_mp sentinel2_acolite_mp sentinel3_acolite_mp tiles_interp)
HAS_KNOWN_DIR=0
for d in "${KNOWN_SENSOR_DIRS[@]}"; do
  [[ -d "$TARGET_DIR/$d" ]] && HAS_KNOWN_DIR=1 && break
done
if [[ "$HAS_KNOWN_DIR" -eq 0 ]]; then
  NESTED_DIR="$(find "$TARGET_DIR" -mindepth 1 -maxdepth 1 -type d | head -n1)"
  if [[ -n "$NESTED_DIR" ]]; then
    echo "Flattening legacy nested fixture directory: $NESTED_DIR"
    find "$NESTED_DIR" -mindepth 1 -maxdepth 1 -exec mv -t "$TARGET_DIR" {} +
    rmdir "$NESTED_DIR"
  fi
fi

echo "Fixtures ready at $TARGET_DIR"
