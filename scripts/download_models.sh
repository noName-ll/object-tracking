#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models
base='https://raw.githubusercontent.com/HonglinChu/SiamTrackers/master/NanoTrack/models/nanotrackv2'
fetch() {
  local name="$1" checksum="$2"
  if [[ -f "models/$name" ]] && printf '%s  %s\n' "$checksum" "models/$name" | sha256sum -c --status; then
    return
  fi
  curl --fail --location --retry 3 "$base/$name" -o "models/$name.part"
  printf '%s  %s\n' "$checksum" "models/$name.part" | sha256sum -c -
  mv "models/$name.part" "models/$name"
}
fetch nanotrack_backbone_sim.onnx 530bdd0cd00f19afab79a863e71ba71e3312395a5dc9151af675082bdaaa2fc4
fetch nanotrack_head_sim.onnx 0d8c0637be849f092cc7236cae02e55c8b9455ebe37ba50601d6115db4247cd9
