#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models/yoloe
file='models/yoloe/test_bus.jpg'
checksum='c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63'
if [[ -f "$file" ]] && printf '%s  %s\n' "$checksum" "$file" | sha256sum -c --status; then
  exit 0
fi
curl --fail --location --retry 3 'https://raw.githubusercontent.com/ultralytics/assets/main/im/bus.jpg' -o "$file.part"
printf '%s  %s\n' "$checksum" "$file.part" | sha256sum -c -
mv "$file.part" "$file"
