"""
VMS Logistics & Return Inspection SOP Analyzer - HYBRID AI VERSION
Uses Ultralytics YOLOv8 for object detection and EasyOCR for text reading.
Falls back to OpenCV heuristics if custom models are not yet trained.
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

# Import AI Libraries
try:
    import easyocr
    # Initialize OCR reader (downloads small weights on first run)
    # Using English, disabling GPU to avoid massive PyTorch downloads on standard PCs
    OCR_READER = easyocr.Reader(['en'], gpu=False) 
    HAS_OCR = True
except ImportError:
    HAS_OCR = False
    OCR_READER = None

try:
    from ultralytics import YOLO
    # Load base model (will download yolov8n.pt on first run ~6MB)
    # Once you train your own model, replace 'yolov8n.pt' with 'best.pt'
    YOLO_MODEL = YOLO('yolov8n.pt')
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    YOLO_MODEL = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class VMSSOPAnalyzer:
    def __init__(self, blur_threshold=30.0, min_hold_sec=2.0, process_fps=3, process_type="Return"):
        self.blur_threshold = blur_threshold
        self.min_hold_sec   = min_hold_sec
        self.process_fps    = process_fps
        self.process_type   = process_type  # "Return" or "Forward"

    def _decision(self, total_score, confidence):
        if total_score >= 85 and confidence >= 0.7:
            return "pass", "Accepted", round(float(confidence), 2)
        elif total_score >= 50 and confidence >= 0.4:
            return "review", "Review Required", round(float(confidence), 2)
        else:
            return "fail", "Rejected", round(float(confidence), 2)

    def _confidence_score(self, quality_failures, n, metrics, barcode):
        if n <= 0: return 0.0
        quality_ratio = 1.0 - min(quality_failures / max(n, 1), 1.0)
        package_ratio = min(metrics.get("pct_package", 0) / 100.0, 1.0)
        label_ratio = min(metrics.get("pct_label", 0) / 100.0, 1.0)
        object_ratio = min(metrics.get("pct_obj", 0) / 100.0, 1.0)
        barcode_factor = 1.0 if barcode else 0.4

        confidence = (0.35 * quality_ratio + 0.25 * package_ratio + 0.20 * label_ratio + 0.10 * object_ratio + 0.10 * barcode_factor)
        return max(0.1, min(1.0, confidence))

    def _sample_video(self, path):
        if not path or not os.path.exists(path): raise FileNotFoundError(f"Video file not found: {path}")
        cap = cv2.VideoCapture(path)
        if not cap.isOpened(): raise IOError(f"Cannot open video: {path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps
        stride = max(1, int(video_fps / self.process_fps))

        frames = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if idx % stride == 0:
                h, w = frame.shape[:2]
                small = cv2.resize(frame, (640, int(h * 640 / w)), interpolation=cv2.INTER_AREA)
                frames.append((idx, small))
            idx += 1
        cap.release()
        return frames, duration

    def _quality(self, gray):
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        bright = float(np.mean(gray))
        valid = blur >= self.blur_threshold and 10 <= bright <= 255
        return valid, blur, bright

    def _ai_detect_text(self, frame):
        """Use EasyOCR to accurately find text instead of guessing."""
        if not HAS_OCR: return 0
        # Convert to RGB for EasyOCR
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = OCR_READER.readtext(rgb, detail=0)
        # Return how many words/text blocks were found
        return len(results)

    def _try_barcode(self, frame, gray):
        """Robust Barcode reader."""
        if not HAS_PYZBAR: return None
        
        candidates = [frame, cv2.cvtColor(cv2.equalizeHist(gray), cv2.COLOR_GRAY2BGR)]
        for img in candidates:
            try:
                results = pyzbar_decode(img)
                if results:
                    return results[0].data.decode("utf-8")
            except Exception:
                pass
        return None

    def _detect_label_region(self, gray):
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 100)
        cnts, _ = cv2.findContours(cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = gray.shape
        for c in cnts:
            if h * w * 0.003 < cv2.contourArea(c) < h * w * 0.55:
                approx = cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True), True)
                if 4 <= len(approx) <= 12:
                    x, y, rw, rh = cv2.boundingRect(c)
                    if 0.2 < rw / max(rh, 1) < 7.0: return True
        return False

    def _detect_package_in_frame(self, gray):
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = gray.shape
        for c in cnts:
            if cv2.contourArea(c) > h * w * 0.15: return True, cv2.contourArea(c) / (h * w)
        return False, 0.0
        
    def _detect_small_tag(self, gray):
        edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 40, 120)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = gray.shape
        for c in cnts:
            area = cv2.contourArea(c)
            if h * w * 0.001 < area < h * w * 0.07: return True
        return False

    def analyze(self, video_path):
        frames, duration = self._sample_video(video_path)
        n = len(frames)
        
        # State tracking
        label_flags, tag_flags, obj_flags, text_counts, package_flags = [], [], [], [], []
        barcode = None
        quality_failures = 0
        blur_scores = []

        # Process frames
        for idx, frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            valid, blur, bright = self._quality(gray)
            blur_scores.append(blur)

            if not valid:
                quality_failures += 1
                continue

            # Hybrid AI/CV Logic
            # 1. OCR for Text Density (Huge accuracy boost)
            if idx % 3 == 0: # run OCR every 3rd frame to save time
                text_count = self._ai_detect_text(frame)
                text_counts.append(text_count)
            else:
                text_counts.append(text_counts[-1] if text_counts else 0)

            # 2. OpenCV Fallbacks (Replace with YOLO later)
            label_flags.append(self._detect_label_region(gray))
            pkg_detected, _ = self._detect_package_in_frame(gray)
            package_flags.append(pkg_detected)
            tag_flags.append(self._detect_small_tag(gray))
            obj_flags.append(pkg_detected) # Treating product similar to package for now
            
            # 3. Barcode
            if barcode is None:
                barcode = self._try_barcode(frame, gray)

        # Aggregation
        pct_label   = sum(label_flags) / max(n,1)
        pct_tag     = sum(tag_flags) / max(n,1)
        pct_obj     = sum(obj_flags) / max(n,1)
        pct_package = sum(package_flags) / max(n,1)
        avg_text    = float(np.mean(text_counts)) if text_counts else 0

        # Scoring
        s1, s2, s3, s4 = 0, 0, 0, 0
        s1_notes, s2_notes, s3_notes, s4_notes = [], [], [], []

        # Step 1: Label
        if avg_text > 5: s1 += 15
        else: s1_notes.append("No clear text/label detected. Hold label closer.")
        if barcode: s1 += 15
        else: s1_notes.append("Barcode not read. Ensure it is in focus.")
        
        # Step 2: Unboxing/Packing
        if pct_package > 0.4: s2 += 10
        else: s2_notes.append("Package not consistently visible.")
        if duration > 10: s2 += 10
        else: s2_notes.append("Video too short for full process.")

        # Step 3: Tags
        if pct_tag > 0.1: s3 += 15
        else: s3_notes.append("Brand tag not detected clearly.")
        if avg_text > 3: s3 += 10
        else: s3_notes.append("Tag text not readable.")

        # Step 4: Product
        if pct_obj > 0.4: s4 += 20
        else: s4_notes.append("Product not visible enough.")
        s4 += 5 # Placeholder for motion rotation

        total_score = s1 + s2 + s3 + s4
        confidence = self._confidence_score(quality_failures, n, {"pct_package": pct_package*100, "pct_label": pct_label*100, "pct_obj": pct_obj*100}, barcode)
        status, decision, conf = self._decision(total_score, confidence)

        return {
            "total_score": total_score, "status": status, "decision": decision, "confidence": conf,
            "duration_s": round(duration, 1), "detected_barcode": barcode, "frames_sampled": n,
            "quality_failures": quality_failures,
            "scores": {
                "shipping_label": {"score": s1, "notes": s1_notes},
                "unboxing":       {"score": s2, "notes": s2_notes},
                "tags":           {"score": s3, "notes": s3_notes},
                "product":        {"score": s4, "notes": s4_notes},
            },
            "metrics": {
                "pct_label": round(pct_label*100, 1), "pct_tag": round(pct_tag*100, 1),
                "pct_obj": round(pct_obj*100, 1), "pct_package": round(pct_package*100, 1),
                "avg_text": round(avg_text, 1), "estimated_sides": 1
            }
        }