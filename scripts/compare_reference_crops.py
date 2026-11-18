"""Compare reference preprocessing using the Rust CLI and a single screenshot."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

import cv2
import numpy as np

from small_target_scene import paste, iou

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("archive")
parser.add_argument("--objects", nargs="+", type=int, default=[1, 5, 20])
parser.add_argument("--methods", nargs="+", choices=("original", "upscale", "grabcut", "grabcut_upscale", "integrated", "guarded", "guarded60", "hybrid", "hybrid42", "auto"), default=["original", "upscale", "grabcut", "grabcut_upscale"])
parser.add_argument("--holdout", action="store_true")
parser.add_argument("--stress", action="store_true", help="Three absent frames, re-entry across the frame, changed lighting and motion blur")
args = parser.parse_args()
cv2.setNumThreads(1)
background = cv2.imread("models/yoloe/test_bus.jpg")
angles = [0, 5, 15, 30, 60, 90, 135, 180, None, 180, 0]


if args.stress:
    angles = [0, 5, 15, 30, 60, 90, 135, 180, None, None, None, 180, 0]


def prepare(image, method):
    if "grabcut" in method:
        cv2.setRNGSeed(7)
        h, w = image.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        models = [np.zeros((1, 65), np.float64) for _ in range(2)]
        cv2.grabCut(image, mask, (1, 1, w-2, h-2), *models, 4, cv2.GC_INIT_WITH_RECT)
        x, y, cw, ch = cv2.boundingRect((mask % 2).astype(np.uint8))
        if min(cw, ch) >= 8:
            image = image[y:y+ch, x:x+cw]
    if "upscale" in method:
        scale = 180 / max(image.shape[:2])
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return image


def run(folder, video, ref, method):
    path = str(folder / "target.png")
    cv2.imwrite(path, prepare(ref, method))
    cmd = ["target/debug/mobile-object-tracker", "--video", video, "--target", path]
    if method in ("integrated", "guarded", "guarded60", "hybrid", "hybrid42"):
        cmd.extend(["--search-backend", "small-target"])
    elif method != "auto":
        cmd.extend(["--search-backend", "features"])
    if method in ("integrated", "guarded", "guarded60"):
        cmd.append("--template-only")
    if method in ("guarded", "hybrid42"):
        cmd.extend(["--appearance-limit", "42"])
    if method == "guarded60":
        cmd.extend(["--appearance-limit", "60"])
    if method == "integrated":
        cmd.extend(["--appearance-limit", "0"])
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode:
        if any(msg in proc.stderr for msg in ("at least 12 SIFT", "cannot separate foreground", "foreground too small")):
            return [{"status": "reference_rejected", "bbox": None}] * len(angles)
        raise RuntimeError(proc.stderr)
    rows = [json.loads(line) for line in proc.stdout.splitlines()]
    assert len(rows) == len(angles)
    return rows


writer = csv.writer(sys.stdout, lineterminator="\n")
writer.writerow("method object size fill frame angle present emitted iou status ms".split())
with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(args.archive) as archive:
    folder = Path(tmp)
    for obj in args.objects:
        video = str(folder / "frames.avi")
        out = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"MJPG"), 24, (320, 240))
        truth = []
        for i, angle in enumerate(angles):
            frame = cv2.resize(background[500+i*4:900+i*4, 100+i*7:640+i*7], (320, 240))
            if args.holdout:
                frame = cv2.flip(frame, 1)
            paste(archive, frame, 2, 0, 58, 245, 130)
            x = 170-9*i if args.holdout else 20+12*i
            side = 54 + i*2 if args.holdout else 70
            if args.stress and i >= 11:
                x = 22
            truth.append(None if angle is None else paste(archive, frame, obj, angle, side, x, 45+i*3))
            if args.stress:
                frame = cv2.convertScaleAbs(frame, alpha=0.86, beta=7)
                if i in (2, 4, 6):
                    kernel = np.zeros((7, 7), np.float32)
                    kernel[3] = 1/7
                    frame = cv2.filter2D(frame, -1, kernel)
            out.write(frame)
        out.release()
        for size in (40, 90):
            for fill in (0.95, 0.60):
                ref = cv2.resize(background[250:500, 30:280] if args.holdout else background[:250, :250], (size, size))
                side = round(size*fill)
                offset = (size-side)//2
                paste(archive, ref, obj, 0, side, offset//2 if args.holdout else offset, offset)
                for method in args.methods:
                    for i, row in enumerate(run(folder, video, ref, method)):
                        box = row["bbox"]
                        pred = [box[k] for k in ("x", "y", "width", "height")] if box else None
                        writer.writerow([method, obj, size, fill, i, angles[i], int(truth[i] is not None),
                                         int(box is not None), round(iou(pred, truth[i]), 4), row["status"],
                                         row.get("processing_ms", "")])
                    sys.stdout.flush()
