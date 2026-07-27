#!/usr/bin/env bash
#
# Generate an md5 manifest of the BIA deposit tree (parallel), for verifying the
# Aspera/FTP transfer. Paths in the manifest are RELATIVE to the deposit root,
# matching the file-list Files column and the upload layout.
#
#   scripts/make_deposit_checksums.sh [ROOT] [OUT] [SUBDIR]
#     ROOT   default /mnt/data/bia_deposit/Brightfield
#     OUT    default <ROOT>/../checksums_<subdir>.md5
#     SUBDIR default "."  (e.g. "annotations" to do just annotations)
#
set -euo pipefail
ROOT="${1:-/mnt/data/bia_deposit/Brightfield}"
SUBDIR="${3:-.}"
OUT="${2:-$(dirname "$ROOT")/checksums_$(echo "$SUBDIR" | tr '/.' '__').md5}"

cd "$ROOT"
echo "hashing *.ome.tiff under $ROOT/$SUBDIR -> $OUT"
find "$SUBDIR" -name '*.ome.tiff' -type f -print0 | sort -z \
  | xargs -0 -P 8 -n 50 md5sum > "$OUT"
echo "wrote $(wc -l < "$OUT") checksums to $OUT"
