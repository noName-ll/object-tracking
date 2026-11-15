"""Synthetic geometry, camera/background motion, loss and reacquisition checks."""
import unittest

import cv2
import numpy as np

from track import Matcher, Tracker


class TrackingTest(unittest.TestCase):
    def test_motion_loss_and_reacquisition(self):
        rng = np.random.default_rng(42)
        reference = rng.integers(0, 256, (240, 300, 3), dtype=np.uint8)
        reference = cv2.GaussianBlur(reference, (3, 3), 0)
        for i in range(35):
            x, y = rng.integers([10, 10], [260, 210])
            cv2.putText(reference, str(i), (int(x), int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
        tracker = Tracker(reference, Matcher(reference, 'sift', 'cpu'))
        corners = np.float32([[0, 0], [299, 0], [299, 239], [0, 239]])

        def scene(quad):
            # A new independent background on every observation.
            frame = rng.integers(0, 130, (600, 900, 3), dtype=np.uint8)
            transform = cv2.getPerspectiveTransform(corners, quad)
            patch = cv2.warpPerspective(reference, transform, (900, 600))
            mask = cv2.warpPerspective(np.full((240, 300), 255, np.uint8), transform, (900, 600))
            frame[mask > 0] = patch[mask > 0]
            return frame

        for quad in [np.float32([[50, 70], [350, 60], [360, 300], [60, 310]]),
                     np.float32([[450, 250], [760, 220], [780, 470], [460, 490]])]:
            result = tracker.update(scene(quad))
            self.assertIsNotNone(result)
            self.assertLess(np.linalg.norm(result['quad'] - quad, axis=1).max(), 5)
        self.assertIsNone(tracker.update(np.zeros((600, 900, 3), np.uint8)))
        quad = np.float32([[180, 100], [470, 120], [450, 350], [160, 330]])
        result = tracker.update(scene(quad))
        self.assertIsNotNone(result)
        self.assertLess(np.linalg.norm(result['quad'] - quad, axis=1).max(), 5)


if __name__ == '__main__':
    unittest.main()
