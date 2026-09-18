import os
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import qrcode

from vms_sop_analyzer import VMSSOPAnalyzer


class TestVMSSOPAnalyzer(unittest.TestCase):
    def _create_video(self, path, include_barcode=False):
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
        self.assertIsNotNone(writer)

        qr_frame = None
        if include_barcode:
            qr = qrcode.QRCode(version=2, box_size=10, border=4)
            qr.add_data("SOP-RETURN-12345")
            qr.make(fit=True)
            qr_img = qr.make_image(fill_color="black", back_color="white")
            qr_bgr = np.array(qr_img.convert("RGB"))[:, :, ::-1].copy()
            qr_frame = cv2.resize(qr_bgr, (120, 120), interpolation=cv2.INTER_AREA)

        for i in range(20):
            frame = np.full((240, 320, 3), 245, dtype=np.uint8)
            if include_barcode and qr_frame is not None:
                frame[60:180, 100:220] = qr_frame
                cv2.putText(
                    frame,
                    "Shipping Label",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (10, 10, 10),
                    1,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    frame,
                    "SOP SAMPLE",
                    (60, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (20, 20, 20),
                    2,
                )
            writer.write(frame)

        writer.release()

    def test_missing_video_raises(self):
        with self.assertRaises(FileNotFoundError):
            VMSSOPAnalyzer().analyze("missing-video.mp4")

    def test_empty_video_returns_zero_pointers(self):
        with patch.object(VMSSOPAnalyzer, "_sample_video", return_value=([], 0.0)):
            result = VMSSOPAnalyzer(process_type="Return").analyze("empty-video.mp4")

        self.assertEqual(result["total_score"], 0)
        self.assertEqual(result["confidence"], 0.0)
        self.assertEqual(result["frames_sampled"], 0)
        self.assertEqual(result["metrics"]["estimated_sides"], 0)
        self.assertEqual(result["metrics"]["direction_changes"], 0)

    def test_analyzer_returns_status_and_confidence_for_valid_video(self):
        fd, path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        try:
            self._create_video(path)
            result = VMSSOPAnalyzer(process_type="Return").analyze(path)
            self.assertIn("status", result)
            self.assertIn("decision", result)
            self.assertIn("confidence", result)
            self.assertIn(result["status"], {"pass", "review", "fail"})
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_barcode_detection_path_accepts_real_qr_code(self):
        fd, path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        try:
            self._create_video(path, include_barcode=True)
            result = VMSSOPAnalyzer(process_type="Return").analyze(path)
            self.assertIn("detected_barcode", result)
            self.assertIsNotNone(result["detected_barcode"])
            self.assertIn("SOP-RETURN-12345", result["detected_barcode"])
        finally:
            if os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    unittest.main()
