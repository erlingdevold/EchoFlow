#!/usr/bin/env bash
# Download EK80 .raw files from the NOAA WCSD public bucket for benchmarking.
# No AWS credentials required (--no-sign-request).
#
# Usage:
#   bash download_bench_data.sh [output_dir]
#   N_FILES=2 bash download_bench_data.sh   # override file count

set -euo pipefail

DEST="${1:-data/benchmark/input}"
BUCKET="s3://noaa-wcsd-pds/data/raw/Bell_M._Shimada/SH2306/EK80"
N_FILES="${N_FILES:-64}"

mkdir -p "$DEST"

# List available .raw files, write to temp file to avoid SIGPIPE issues
echo "Listing files in $BUCKET ..."
TMPLIST=$(mktemp)
trap 'rm -f "$TMPLIST"' EXIT
aws s3 ls "$BUCKET/" --no-sign-request 2>/dev/null \
    | awk '/\.raw$/ {print $4}' > "$TMPLIST"

# Take first N files
FILES=$(head -n "$N_FILES" "$TMPLIST")
COUNT=$(echo "$FILES" | wc -l | tr -d ' ')
echo "Downloading $COUNT files to $DEST ..."

DOWNLOADED=0
for f in $FILES; do
    OUTPATH="$DEST/$f"
    if [ -f "$OUTPATH" ]; then
        echo "  [skip] $f (already exists)"
    else
        echo "  [get]  $f"
        aws s3 cp --no-sign-request "$BUCKET/$f" "$OUTPATH"
    fi
    DOWNLOADED=$((DOWNLOADED + 1))
    echo "  ($DOWNLOADED / $COUNT)"
done

echo "Done. $COUNT files in $DEST"
du -sh "$DEST"
