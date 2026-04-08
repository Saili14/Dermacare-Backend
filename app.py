# backend-py/app.py

import io, os
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# 🔥 YOUR MODEL
from model_loader_fr import predict_image

app = FastAPI(title="Skin Models API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Helpers
# ----------------------------
def pil_from(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")

def cv2_from(b: bytes):
    import cv2
    arr = np.frombuffer(b, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

# ----------------------------
# Model A: Severity (✅ FIXED)
# ----------------------------
from models import severity as sev

@app.post("/analyze/severity")
async def analyze_severity(
    condition: str = Form(...),
    file: UploadFile = File(...)
):
    raw = await file.read()
    img_bgr = cv2_from(raw)

    if img_bgr is None:
        raise HTTPException(400, "Invalid image")

    # 🔥 NEW STRUCTURED RESULT
    # 🔥 NEW STRUCTURED RESULT
    result = sev.analyze_skin_image(img_bgr, condition)

    if "error" in result:
        raise HTTPException(400, result["error"])

        # 🔥 ADD THIS LINE
    recommendations = sev.generate_recommendations(
        result["severity"],
        result["condition"].lower(),
        {}
    )

    return {
        "model": "severity",
        "ok": True,
        "condition": result["condition"],
        "severity": result["severity"],
        "remedies": result["remedies"],
        "recommendations": recommendations   # ✅ NEW
    }

# ----------------------------
# Model B: Disease (UNCHANGED)
# ----------------------------
@app.post("/analyze/disease")
async def analyze_disease(file: UploadFile = File(...)):
    raw = await file.read()

    if not raw:
        raise HTTPException(400, "Empty upload")

    temp_path = f"temp_{file.filename}"

    try:
        with open(temp_path, "wb") as buffer:
            buffer.write(raw)

        result = predict_image(temp_path)
        print("🧠 FULL RESULT:", result)

        top_data = result.get("top", {})
        label = top_data.get("label", "unknown")
        confidence = float(top_data.get("confidence", 0.0))
        probabilities = result.get("probabilities", {})

        # ✅ Healthy first
        if probabilities.get("healthy", 0) > 0.5:
            label = "healthy"
            confidence = probabilities.get("healthy")

        else:
            sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

            if len(sorted_probs) >= 2:
                top1, top2 = sorted_probs[0], sorted_probs[1]

                if abs(top1[1] - top2[1]) < 0.1:
                    label = "uncertain"
                    confidence = top1[1]

            elif len(sorted_probs) == 1:
                label = sorted_probs[0][0]
                confidence = sorted_probs[0][1]

            if confidence < 0.4:
                label = "uncertain"

        return {
            "model": "disease",
            "ok": True,
            "top": {
                "label": label,
                "p": confidence
            },
            "probabilities": probabilities
        }

    except Exception as e:
        print("❌ Prediction Error:", str(e))
        raise HTTPException(500, f"Prediction failed: {str(e)}")

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ----------------------------
# Model C: SkinTone (SKIPPED)
# ----------------------------
@app.post("/analyze/skintone")
async def analyze_skintone_skip(file: UploadFile = File(...)):
    return {
        "model": "skintone",
        "ok": False,
        "note": "Skin tone model skipped for now (no checkpoint found).",
        "top": {"label": "N/A", "p": 0.0}
    }

# ----------------------------
# Health & Warmup
# ----------------------------
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/warmup")
def warmup_post():
    try:
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Warmup failed: {e}")

@app.get("/warmup")
def warmup_get():
    return warmup_post()