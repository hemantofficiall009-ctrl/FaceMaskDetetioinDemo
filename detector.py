import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

COLORS = {
    "with_mask": (46, 160, 67),
    "without_mask": (215, 58, 73),
    "mask_weared_incorrect": (227, 160, 8),
}


class MaskDetector:
    def __init__(self, model_path="model.pt"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(model_path, map_location=self.device)
        self.classes = ckpt["classes"]
        size = ckpt.get("size", 128)

        m = models.mobilenet_v2(weights=None)
        m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.last_channel, len(self.classes)))
        m.load_state_dict(ckpt["state_dict"])
        self.model = m.to(self.device).eval()

        self.tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        try:
            import mediapipe as mp
            self._mp = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        except Exception:
            self._mp = None
            self._haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def _detect(self, rgb):
        H, W = rgb.shape[:2]
        boxes = []
        if self._mp is not None:
            res = self._mp.process(rgb)
            for d in (res.detections or []):
                r = d.location_data.relative_bounding_box
                boxes.append((int(r.xmin * W), int(r.ymin * H), int(r.width * W), int(r.height * H)))
        else:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            boxes = [tuple(b) for b in self._haar.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))]
        return boxes

    @torch.no_grad()
    def predict(self, rgb, pad=0.15):
        out = rgb.copy()
        H, W = rgb.shape[:2]
        results = []
        for (x, y, w, h) in self._detect(rgb):
            px, py = int(w * pad), int(h * pad)
            x1, y1 = max(0, x - px), max(0, y - py)
            x2, y2 = min(W, x + w + px), min(H, y + h + py)
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue
            face = Image.fromarray(rgb[y1:y2, x1:x2])
            probs = torch.softmax(self.model(self.tf(face).unsqueeze(0).to(self.device)), 1)[0]
            idx = int(probs.argmax())
            label, conf = self.classes[idx], float(probs[idx])
            results.append({"label": label, "confidence": conf, "box": (x1, y1, x2, y2)})
            color = COLORS.get(label, (255, 255, 255))
            t = max(2, W // 300)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, t)
            text = f"{label.replace('_', ' ')} {conf:.0%}"
            scale = max(0.5, W / 1200)
            cv2.putText(out, text, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, scale, color, max(1, t - 1))
        return out, results
