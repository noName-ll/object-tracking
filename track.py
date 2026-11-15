"""Reference-image tracking with geometric verification and global reacquisition."""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


class Matcher:
    def __init__(self, reference, backend, device):
        self.backend = backend
        if backend == 'sift':
            self.extractor = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.02)
            self.ref_points, self.ref_desc = self.extract(reference)
            if self.ref_desc is None or len(self.ref_points) < 12:
                raise ValueError('Reference has too few features; choose a clearer, textured target.')
        else:
            import torch
            from lightglue import LightGlue, SuperPoint
            from lightglue.utils import numpy_image_to_torch
            self.torch = torch
            self.convert = numpy_image_to_torch
            self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
            self.extractor = SuperPoint(max_num_keypoints=2048).eval().to(self.device)
            self.matcher = LightGlue(features='superpoint').eval().to(self.device)
            with torch.inference_mode():
                self.ref_features = self.extractor.extract(self.tensor(reference))
            self.ref_points = self.ref_features['keypoints'][0].cpu().numpy()
            if len(self.ref_points) < 12:
                raise ValueError('Reference has too few features; choose a clearer target.')

    def tensor(self, bgr):
        return self.convert(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).to(self.device)

    def extract(self, image):
        points, desc = self.extractor.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None)
        return np.array([p.pt for p in points], dtype=np.float32).reshape(-1, 2), desc

    def match(self, image):
        if self.backend == 'sift':
            points, desc = self.extract(image)
            if desc is None or len(desc) < 2:
                return np.empty((0, 2)), np.empty((0, 2))
            bf = cv2.BFMatcher(cv2.NORM_L2)
            forward = bf.knnMatch(self.ref_desc, desc, k=2)
            reverse = bf.match(desc, self.ref_desc)
            pairs = [m for pair in forward if len(pair) == 2
                     for m, n in [pair] if m.distance < 0.75 * n.distance
                     and reverse[m.trainIdx].trainIdx == m.queryIdx]
            return (self.ref_points[[m.queryIdx for m in pairs]],
                    points[[m.trainIdx for m in pairs]])
        with self.torch.inference_mode():
            features = self.extractor.extract(self.tensor(image))
            result = self.matcher({'image0': self.ref_features, 'image1': features})
            pairs = result['matches'][0].cpu().numpy()
            points = features['keypoints'][0].cpu().numpy()
        return self.ref_points[pairs[:, 0]], points[pairs[:, 1]]


def estimate(src, dst, reference_shape, frame_shape, min_inliers, min_ratio):
    """Return a verified quadrilateral; reject weak or implausible homographies."""
    if len(src) < max(8, min_inliers):
        return None
    h, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 4.0,
                                 maxIters=5000, confidence=0.999)
    if h is None or mask is None or not np.isfinite(h).all():
        return None
    keep = mask.ravel().astype(bool)
    count = int(keep.sum())
    ratio = count / len(src)
    if count < min_inliers or ratio < min_ratio:
        return None
    rh, rw = reference_shape[:2]
    fh, fw = frame_shape[:2]
    # Reject matches clustered on a tiny logo/corner of the reference.
    coverage = cv2.contourArea(cv2.convexHull(src[keep].astype(np.float32))) / (rh * rw)
    if coverage < 0.035:
        return None
    corners = np.float32([[0, 0], [rw - 1, 0], [rw - 1, rh - 1], [0, rh - 1]])
    quad = cv2.perspectiveTransform(corners[None], h)[0]
    if not np.isfinite(quad).all() or not cv2.isContourConvex(quad):
        return None
    area = abs(cv2.contourArea(quad))
    edges = np.linalg.norm(quad - np.roll(quad, 1, axis=0), axis=1)
    if area < 100 or area > fw * fh * 1.5 or edges.min() < 5 or edges.max() / edges.min() > 25:
        return None
    # Do not report a mostly off-screen object as a valid observation.
    screen = np.float32([[0, 0], [fw - 1, 0], [fw - 1, fh - 1], [0, fh - 1]])
    visible, _ = cv2.intersectConvexConvex(quad.astype(np.float32), screen)
    if visible / area < 0.35:
        return None
    projected = cv2.perspectiveTransform(src[keep][None].astype(np.float32), h)[0]
    error = float(np.median(np.linalg.norm(projected - dst[keep], axis=1)))
    if error > 4.0:
        return None
    return {'quad': quad, 'inliers': count, 'inlier_ratio': ratio,
            'reference_coverage': float(coverage), 'reprojection_error_px': error}


class Tracker:
    def __init__(self, reference, matcher, min_inliers=12, min_ratio=0.35, global_every=15):
        self.reference = reference
        self.matcher = matcher
        self.min_inliers = min_inliers
        self.min_ratio = min_ratio
        self.global_every = global_every
        self.last_quad = None
        self.index = 0

    def update(self, frame):
        self.index += 1
        fh, fw = frame.shape[:2]
        regions = []
        if self.last_quad is not None and self.index % self.global_every:
            lo, hi = self.last_quad.min(axis=0), self.last_quad.max(axis=0)
            margin = np.maximum((hi - lo) * 0.75, 80)
            x0, y0 = np.maximum(lo - margin, 0).astype(int)
            x1, y1 = np.minimum(hi + margin, [fw, fh]).astype(int)
            if x1 > x0 and y1 > y0:
                regions.append((x0, y0, x1, y1))
        regions.append((0, 0, fw, fh))
        for x0, y0, x1, y1 in regions:
            src, dst = self.matcher.match(frame[y0:y1, x0:x1])
            dst = dst + np.float32([x0, y0])
            result = estimate(src, dst, self.reference.shape, frame.shape,
                              self.min_inliers, self.min_ratio)
            if result is not None:
                self.last_quad = result['quad']
                return result
        self.last_quad = None
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='Camera index, video path, RTSP or HTTP URL')
    parser.add_argument('--target', required=True, help='Reference screenshot')
    parser.add_argument('--backend', choices=['sift', 'lightglue'], default='lightglue')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--select-target', action='store_true', help='Select object in screenshot once')
    selection.add_argument('--target-roi', type=int, nargs=4, metavar=('X', 'Y', 'W', 'H'))
    parser.add_argument('--max-width', type=int, default=960)
    parser.add_argument('--min-inliers', type=int, default=12)
    parser.add_argument('--min-ratio', type=float, default=0.35)
    parser.add_argument('--global-every', type=int, default=15)
    parser.add_argument('--jsonl', help='Write observations; coordinates refer to original source resolution')
    parser.add_argument('--no-display', action='store_true')
    args = parser.parse_args()
    if args.max_width <= 0 or args.global_every <= 0 or args.min_inliers < 4 or not 0 < args.min_ratio <= 1:
        parser.error('Invalid tracking thresholds or dimensions')
    reference = cv2.imread(args.target)
    if reference is None:
        parser.error('Cannot read target screenshot')
    roi = args.target_roi
    if args.select_target:
        if args.no_display:
            parser.error('--select-target requires a display; use --target-roi on a server')
        roi = cv2.selectROI('Select object, ENTER to confirm', reference, False, False)
        cv2.destroyAllWindows()
    if roi is not None:
        x, y, w, h = roi
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > reference.shape[1] or y + h > reference.shape[0]:
            parser.error('Target ROI must be inside screenshot and have positive size')
        reference = reference[y:y+h, x:x+w].copy()
    else:
        print('WARNING: full screenshot used; background features may be tracked. Prefer --select-target.', flush=True)
    scale = min(1.0, args.max_width / reference.shape[1])
    reference = cv2.resize(reference, None, fx=scale, fy=scale)
    matcher = Matcher(reference, args.backend, args.device)
    tracker = Tracker(reference, matcher, args.min_inliers, args.min_ratio, args.global_every)
    source = int(args.source) if args.source.isdecimal() else args.source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError('Cannot open source')
    output = open(Path(args.jsonl), 'w', encoding='utf-8') if args.jsonl else None
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print('Source ended or disconnected.', flush=True)
                break
            acquired = time.time()
            fh, fw = frame.shape[:2]
            width = min(args.max_width, fw)
            height = max(1, round(fh * width / fw))
            work = cv2.resize(frame, (width, height))
            started = time.perf_counter()
            result = tracker.update(work)
            record = {'frame': frame_index, 'received_at_unix': acquired,
                      'status': 'tracking' if result else 'lost',
                      'frame_size': [fw, fh], 'polygon': None, 'bbox_xywh': None}
            if result:
                quad = result.pop('quad')
                original = quad * np.float32([fw / width, fh / height])
                lo = np.maximum(original.min(axis=0), [0, 0])
                hi = np.minimum(original.max(axis=0), [fw, fh])
                record.update(result)
                record.update(polygon=original.tolist(), bbox_xywh=[*lo.tolist(), *(hi-lo).tolist()])
                cv2.polylines(work, [quad.astype(np.int32)], True, (0, 230, 0), 2)
            record['processing_ms'] = (time.perf_counter() - started) * 1000
            if output:
                output.write(json.dumps(record) + '\n')
                output.flush()
            if not args.no_display:
                label = f"{record['status']} | {record['processing_ms']:.0f} ms"
                cv2.putText(work, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 230, 0) if result else (0, 0, 255), 2)
                cv2.imshow('Reference tracker - Q to exit', work)
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break
            frame_index += 1
    finally:
        capture.release()
        if output:
            output.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
