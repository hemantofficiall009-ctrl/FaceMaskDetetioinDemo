"""
Step 3: Web demo.  Run:  python app.py
Upload a photo or use your webcam; faces get boxes + mask labels.
"""
import gradio as gr
import numpy as np

from detector import MaskDetector

detector = MaskDetector("model.pt")


def run(image):
    if image is None:
        return None, "Upload an image first."
    image = image.convert("RGB")
    image.thumbnail((700, 700))   # badi photo ko chhota karo
    annotated, res = detector.predict(np.array(image))
    if not res:
        return annotated, "No faces detected."
    lines = [f"Face {i+1}: {r['label'].replace('_', ' ')} ({r['confidence']:.0%})" for i, r in enumerate(res)]
    return annotated, "\n".join(lines)
demo = gr.Interface(
    fn=run,
    inputs=gr.Image(type="pil", sources=["upload", "webcam"], label="Photo"),
    outputs=[gr.Image(label="Result"), gr.Textbox(label="Detections")],
    title="Face Mask Detection Demo",
    description="Detects faces and classifies: with mask / without mask / mask worn incorrectly. "
                "Educational demo — not for real safety or surveillance decisions.",
)

import os

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 10000)))
