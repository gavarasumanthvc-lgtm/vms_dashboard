"""
VMS Logistics & Return Inspection SOP Analyzer
4-step CV scoring engine aligned to the VMS SOP document.
"""
import os
import cv2
import numpy as np
import logging

try:
    from pyzbar.pyzbar import decode as pyzbar_decode
    HAS_PYZBAR = True
except ImportError:
    HAS_PYZBAR = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class VMSSOPAnalyzer:
    def __init__(self, blur_threshold=30.0, min_hold_sec=2.0, process_fps=3, process_type="Return"):
        self.blur_threshold = blur_threshold
        self.min_hold_sec   = min_hold_sec
        self.process_fps    = process_fps
        self.process_type   = process_type  # "Return" or "Forward"

    def _decision(self, total_score, confidence):
        """Return safe production-style status rather than raw score-only verdict."""
        if total_score >= 85 and confidence >= 0.7:
            status = "pass"
            decision = "Accepted"
        elif total_score >= 50 and confidence >= 0.4:
            status = "review"
            decision = "Review Required"
        elif total_score < 50:
            status = "fail"
            decision = "Rejected"
        else:
            status = "review"
            decision = "Review Required"
        return status, decision, round(float(confidence), 2)

    def _confidence_score(self, quality_failures, n, metrics, barcode):
        """Estimate confidence for the final decision. Uncertain signals default to review."""
        if n <= 0:
            return 0.0

        quality_ratio = 1.0 - min(quality_failures / max(n, 1), 1.0)
        package_ratio = min(metrics.get("pct_package", 0) / 100.0, 1.0)
        label_ratio = min(metrics.get("pct_label", 0) / 100.0, 1.0)
        object_ratio = min(metrics.get("pct_obj", 0) / 100.0, 1.0)
        barcode_factor = 1.0 if barcode else 0.4

        confidence = (
            0.35 * quality_ratio +
            0.25 * package_ratio +
            0.20 * label_ratio +
            0.10 * object_ratio +
            0.10 * barcode_factor
        )
        return max(0.1, min(1.0, confidence))

    # ── Frame sampling ────────────────────────────────────────────────────────
    def _sample_video(self, path):
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Video file not found: {path}")

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps
        stride = max(1, int(video_fps / self.process_fps))

        frames = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % stride == 0:
                h, w = frame.shape[:2]
                small = cv2.resize(frame, (640, int(h * 640 / w)), interpolation=cv2.INTER_AREA)
                frames.append((idx, small))
            idx += 1
        cap.release()
        return frames, duration

    # ── Frame quality ─────────────────────────────────────────────────────────
    def _quality(self, gray):
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        bright = float(np.mean(gray))
        valid = blur >= self.blur_threshold and 10 <= bright <= 255
        return valid, blur, bright

    def _label_regions(self, gray):
        """Return candidate label-like rectangular regions in a frame."""
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 100)
        dilated = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
        cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = gray.shape
        frame_area = h * w
        boxes = []
        for c in cnts:
            area = cv2.contourArea(c)
            if frame_area * 0.003 < area < frame_area * 0.55:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.03 * peri, True)
                if 4 <= len(approx) <= 12:
                    x, y, rw, rh = cv2.boundingRect(c)
                    if 0.2 < rw / max(rh, 1) < 7.0:
                        boxes.append((x, y, rw, rh))
        boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
        return boxes[:5]

    # ── Barcode reading (multiple methods) ───────────────────────────────────
    def _try_barcode(self, frame, gray):
        """
        Try multiple approaches to read barcodes:
        1. pyzbar on original frame
        2. pyzbar on cropped label regions
        3. pyzbar on preprocessed (sharpened, contrast-enhanced) frames
        4. QR / barcode detector fallback
        """
        def decode_candidates(images):
            for image in images:
                if image is None:
                    continue
                if image.ndim == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                if HAS_PYZBAR:
                    try:
                        results = pyzbar_decode(image)
                        if results:
                            return results[0].data.decode("utf-8")
                    except Exception:
                        pass
                try:
                    qr = cv2.QRCodeDetector()
                    data, _, _ = qr.detectAndDecode(image)
                    if data:
                        return data
                except Exception:
                    pass
                try:
                    detector = cv2.barcode.BarcodeDetector()
                    ok, decoded, _, _ = detector.detectAndDecodeMulti(image)
                    if ok and decoded:
                        return decoded[0]
                except Exception:
                    pass
            return None

        candidates = [frame]

        # Try label-focused crops first
        label_boxes = self._label_regions(gray)
        for x, y, w, h in label_boxes:
            pad = 20
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(frame.shape[1], x + w + pad)
            y1 = min(frame.shape[0], y + h + pad)
            crop = frame[y0:y1, x0:x1]
            if crop.size:
                candidates.append(crop)

        # Add multiple preprocessing variants
        for image in list(candidates):
            h, w = image.shape[:2]
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            candidates.append(cv2.equalizeHist(gray_img))
            kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]], dtype=np.float32)
            candidates.append(cv2.filter2D(image, -1, kernel))
            candidates.append(cv2.resize(image, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC))

            for angle in (0, 90, 180, 270):
                if angle == 0:
                    rotated = image
                else:
                    rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE if angle == 90 else cv2.ROTATE_180 if angle == 180 else cv2.ROTATE_90_COUNTERCLOCKWISE)
                candidates.append(rotated)

        # Final attempt on a more robust grayscale thresholding
        thresholded = cv2.adaptiveThreshold(
            cv2.GaussianBlur(gray, (5, 5), 0),
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            10,
        )
        candidates.append(cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR))

        return decode_candidates(candidates)

    # ── CV detectors ──────────────────────────────────────────────────────────
    def _detect_label_region(self, gray):
        """Rectangular label/tag region."""
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 100)
        dilated = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
        cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = gray.shape
        fa = h * w
        for c in cnts:
            area = cv2.contourArea(c)
            if fa * 0.003 < area < fa * 0.55:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.03 * peri, True)
                if 4 <= len(approx) <= 12:
                    x, y, rw, rh = cv2.boundingRect(c)
                    if 0.2 < rw / max(rh, 1) < 7.0:
                        return True
        return False

    def _detect_text_density(self, gray):
        """Count horizontal text blobs."""
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 4))
        dilated = cv2.dilate(adaptive, kernel, iterations=1)
        cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = gray.shape
        return len([c for c in cnts if 200 < cv2.contourArea(c) < h * w * 0.15])

    def _detect_large_object(self, gray):
        """Dominant object (product/package) filling >8% of frame."""
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = gray.shape
        return any(cv2.contourArea(c) > h * w * 0.08 for c in cnts)

    def _detect_package_in_frame(self, gray):
        """
        Detect if a package/polybag is prominently in frame.
        Package should fill at least 15% of frame (larger threshold than product).
        """
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = gray.shape
        frame_area = h * w
        for c in cnts:
            area = cv2.contourArea(c)
            if area > frame_area * 0.15:
                # Check if roughly rectangular (package-like)
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.04 * peri, True)
                if 4 <= len(approx) <= 12:
                    return True, area / frame_area
        return False, 0.0

    def _detect_rotation_direction(self, prev_gray, curr_gray):
        """
        Detect if package is being rotated using optical flow.
        Returns: direction (1=clockwise, -1=counter, 0=static), magnitude
        """
        if prev_gray is None:
            return 0, 0.0
        try:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mean_mag = float(np.mean(mag))
            # Check dominant direction (left vs right rotation)
            h, w = flow.shape[:2]
            left_flow  = float(np.mean(flow[:, :w//2, 0]))
            right_flow = float(np.mean(flow[:, w//2:, 0]))
            direction = 1 if right_flow > left_flow else -1 if left_flow > right_flow else 0
            return direction, mean_mag
        except Exception:
            diff = cv2.absdiff(prev_gray, curr_gray)
            return 0, float(np.mean(diff))

    def _detect_small_tag(self, gray):
        """Small hangtag."""
        edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 40, 120)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = gray.shape
        for c in cnts:
            area = cv2.contourArea(c)
            if h * w * 0.001 < area < h * w * 0.07:
                x, y, rw, rh = cv2.boundingRect(c)
                aspect = rw / max(rh, 1)
                solidity = area / (rw * rh) if rw * rh > 0 else 0
                if 0.2 < aspect < 5.0 and solidity > 0.25:
                    return True
        return False

    def _detect_motion(self, prev_gray, curr_gray):
        if prev_gray is None:
            return 0.0
        return float(np.mean(cv2.absdiff(prev_gray, curr_gray)))

    # ── Main analysis ─────────────────────────────────────────────────────────
    def analyze(self, video_path):
        frames, duration = self._sample_video(video_path)
        n = len(frames)

        blur_scores, bright_scores, motions = [], [], []
        label_flags, tag_flags, obj_flags, text_counts = [], [], [], []
        package_in_frame_flags = []
        rotation_directions = []
        rotation_magnitudes = []
        barcode = None
        prev_gray = None
        quality_failures = 0

        for idx, frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            valid, blur, bright = self._quality(gray)

            blur_scores.append(blur)
            bright_scores.append(bright)
            direction, mag = self._detect_rotation_direction(prev_gray, gray)
            motions.append(mag)
            rotation_directions.append(direction)
            rotation_magnitudes.append(mag)

            if not valid:
                quality_failures += 1
                label_flags.append(False)
                tag_flags.append(False)
                obj_flags.append(False)
                text_counts.append(0)
                package_in_frame_flags.append(False)
                prev_gray = gray
                continue

            label_flags.append(self._detect_label_region(gray))
            tag_flags.append(self._detect_small_tag(gray))
            obj_flags.append(self._detect_large_object(gray))
            text_counts.append(self._detect_text_density(gray))
            pkg_detected, _ = self._detect_package_in_frame(gray)
            package_in_frame_flags.append(pkg_detected)

            # Try barcode on every frame until found
            if barcode is None:
                barcode = self._try_barcode(frame, gray)

            prev_gray = gray

        # ── Aggregates ────────────────────────────────────────────────────────
        pct_label   = sum(label_flags) / n if n else 0
        pct_tag     = sum(tag_flags) / n if n else 0
        pct_obj     = sum(obj_flags) / n if n else 0
        pct_package = sum(package_in_frame_flags) / n if n else 0
        avg_blur    = float(np.mean(blur_scores)) if blur_scores else 0
        avg_bright  = float(np.mean(bright_scores)) if bright_scores else 0
        avg_motion  = float(np.mean(motions[1:])) if len(motions) > 1 else 0
        avg_text    = float(np.mean(text_counts)) if text_counts else 0

        label_blur_vals = [blur_scores[i] for i in range(n) if label_flags[i]]
        avg_label_blur  = float(np.mean(label_blur_vals)) if label_blur_vals else 0

        tag_text_vals  = [text_counts[i] for i in range(n) if tag_flags[i]]
        avg_tag_text   = float(np.mean(tag_text_vals)) if tag_text_vals else 0

        obj_motion_vals = [motions[i] for i in range(n) if obj_flags[i] and i < len(motions)]
        avg_obj_motion  = float(np.mean(obj_motion_vals)) if obj_motion_vals else 0

        # Count rotation direction changes (indicates package was rotated multiple sides)
        direction_changes = 0
        prev_dir = 0
        for d in rotation_directions:
            if d != 0 and d != prev_dir:
                direction_changes += 1
                prev_dir = d

        # Estimate sides shown based on direction changes and motion events
        # Each significant motion event with direction change = new side shown
        high_motion_frames = sum(1 for m in rotation_magnitudes if m > 5)
        estimated_sides_shown = 0 if n == 0 else min(6, max(1, direction_changes + 1))
        # Boost sides estimate based on how many high-motion frames there are
        if high_motion_frames > n * 0.3:
            estimated_sides_shown = min(6, estimated_sides_shown + 2)

        # ── Step 1: Shipping Label & Barcode (30 pts) ─────────────────────────
        s1, s1_notes = 0, []

        if label_blur_vals:
            if avg_label_blur > 200:   s1 += 10
            elif avg_label_blur > 80:  s1 += 6;  s1_notes.append("Label partially blurred — hold still for a clearer shot")
            else:                      s1 += 2;  s1_notes.append("Label too blurry — move closer and hold steady")
        else:
            s1_notes.append("No label region detected — hold polybag label flat toward camera before opening")

        if avg_text > 15:   s1 += 10
        elif avg_text > 5:  s1 += 5;  s1_notes.append("Limited text visible — ensure full label fits in frame")
        else:               s1 += 0;  s1_notes.append("No readable text/barcode content detected on label")

        if barcode:
            s1 += 10
        else:
            s1_notes.append("Barcode not decoded — hold label closer in sharp focus for ≥2s")

        s1 = min(s1, 30)

        # ── Step 2: Packing (Forward) or Unboxing/Seal (Return) (20 pts) ────────
        s2, s2_notes = 0, []

        if self.process_type == "Forward":
            # FORWARD: Check packing process — item packed and sealed on camera
            # Package must be visible (being packed)
            if pct_package > 0.6:
                s2 += 6
            elif pct_package > 0.3:
                s2 += 3
                s2_notes.append("Package not consistently in frame — keep package centered while packing")
            else:
                s2 += 0
                s2_notes.append("Package not visible — show packing process clearly on camera")

            # Duration adequate for packing
            if duration > 30:   s2 += 4
            elif duration > 10: s2 += 2;  s2_notes.append(f"Short video ({duration:.0f}s) — packing may be incomplete")
            else:               s2 += 0;  s2_notes.append(f"Video too short ({duration:.0f}s) for full packing documentation")

            # Motion indicates active packing (folding, placing, sealing)
            if avg_motion > 8:   s2 += 10
            elif avg_motion > 3: s2 += 5;  s2_notes.append("Limited packing activity — show full packing and sealing process")
            else:                s2 += 0;  s2_notes.append("No packing activity detected — record the full packing process on camera")

        else:
            # RETURN: Check seal integrity and unboxing — all 6 sides must be shown
            # Package must be visible
            if pct_package > 0.6:
                s2 += 4
            elif pct_package > 0.3:
                s2 += 2
                s2_notes.append("Package not consistently in frame — keep package centered toward camera")
            else:
                s2 += 0
                s2_notes.append("Package not visible — hold package directly toward camera before opening")

            # Duration
            if duration > 30:   s2 += 2
            elif duration > 10: s2 += 1;  s2_notes.append(f"Short video ({duration:.0f}s) — unboxing may be incomplete")
            else:               s2 += 0;  s2_notes.append(f"Video too short ({duration:.0f}s) for full unboxing documentation")

            # All 6 sides must be shown before opening
            if estimated_sides_shown >= 6:
                s2 += 14
            elif estimated_sides_shown >= 4:
                s2 += 9
                s2_notes.append(f"~{estimated_sides_shown} sides shown — must show all 6 sides before opening")
            elif estimated_sides_shown >= 2:
                s2 += 4
                s2_notes.append(f"Only ~{estimated_sides_shown} sides shown — rotate package to show all 6 sides")
            else:
                s2 += 0
                s2_notes.append("Package not rotated — show all 6 sides (front, back, top, bottom, left, right)")

        s2 = min(s2, 20)

        # ── Step 3: Brand Tag & Price Hangtag (25 pts) ────────────────────────
        s3, s3_notes = 0, []

        if pct_tag > 0.3:   s3 += 12
        elif pct_tag > 0.1: s3 += 6;  s3_notes.append("Tag shown briefly — hold close to camera for ≥3 seconds")
        else:               s3 += 0;  s3_notes.append("Brand/price hangtag not detected — present all tags close to camera")

        if tag_text_vals:
            if avg_tag_text > 5:  s3 += 8
            else:                 s3 += 3;  s3_notes.append("Tag text not clearly readable — hold tag steady and closer")
        else:
            s3_notes.append("No tag text detected — ensure label text faces the camera lens")

        tag_blur_vals = [blur_scores[i] for i in range(n) if tag_flags[i]]
        if tag_blur_vals:
            if float(np.mean(tag_blur_vals)) > 150: s3 += 5
            else:                                   s3 += 2;  s3_notes.append("Tag image blurred — hold steady while presenting")

        s3 = min(s3, 25)

        # ── Step 4: Product Display & Condition (25 pts) ──────────────────────
        s4, s4_notes = 0, []

        if pct_obj > 0.7:   s4 += 8
        elif pct_obj > 0.4: s4 += 4;  s4_notes.append("Product not consistently unfolded in frame — hold item fully open")
        else:               s4 += 0;  s4_notes.append("Product display insufficient — unfold and hold item toward camera")

        if pct_obj > 0.3:
            if avg_obj_motion > 5:   s4 += 10
            elif avg_obj_motion > 2: s4 += 5;  s4_notes.append("Product flipped only once — show both front and back clearly")
            else:                    s4 += 0;  s4_notes.append("Product held static — rotate to show front and back")
        else:
            s4_notes.append("Product not sufficiently visible to assess both sides")

        obj_blur_vals = [blur_scores[i] for i in range(n) if obj_flags[i]]
        if obj_blur_vals:
            if float(np.mean(obj_blur_vals)) > 200: s4 += 7
            else:                                   s4 += 3;  s4_notes.append("Product images blurred — hold item steady under good lighting")

        s4 = min(s4, 25)

        total_score = s1 + s2 + s3 + s4
        confidence = self._confidence_score(quality_failures, n, {
            "pct_package": pct_package * 100,
            "pct_label": pct_label * 100,
            "pct_obj": pct_obj * 100,
        }, barcode)
        status, decision, confidence = self._decision(total_score, confidence)

        return {
            "total_score": total_score,
            "status": status,
            "decision": decision,
            "confidence": confidence,
            "duration_s": round(duration, 1),
            "detected_barcode": barcode,
            "frames_sampled": n,
            "quality_failures": quality_failures,
            "scores": {
                "shipping_label": {"score": s1, "notes": s1_notes},
                "unboxing":       {"score": s2, "notes": s2_notes},
                "tags":           {"score": s3, "notes": s3_notes},
                "product":        {"score": s4, "notes": s4_notes},
            },
            "metrics": {
                "avg_blur":           round(avg_blur, 1),
                "avg_brightness":     round(avg_bright, 1),
                "avg_motion":         round(avg_motion, 2),
                "pct_label":          round(pct_label * 100, 1),
                "pct_tag":            round(pct_tag * 100, 1),
                "pct_obj":            round(pct_obj * 100, 1),
                "pct_package":        round(pct_package * 100, 1),
                "avg_text":           round(avg_text, 1),
                "estimated_sides":    estimated_sides_shown,
                "direction_changes":  direction_changes,
            },
        }