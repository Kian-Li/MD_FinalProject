# ============================================================
# SAGEMAKER INFERENCE SCRIPT
# Credit Score Classification
# ============================================================

import os
import json
import joblib
import pandas as pd

# ============================================================
# LOAD MODEL
# ============================================================


def model_fn(model_dir):
    model_path = os.path.join(model_dir, "best_pipeline.pkl")
    label_encoder_path = os.path.join(model_dir, "label_encoder.pkl")
    model = joblib.load(model_path)
    label_encoder = joblib.load(label_encoder_path)
    return {"model": model, "label_encoder": label_encoder}


# ============================================================
# INPUT
# ============================================================


def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        data = json.loads(request_body)
        df = pd.DataFrame([data])
        return df
    raise ValueError(f"Unsupported content type: {request_content_type}")


# ============================================================
# PREDICTION
# ============================================================


def predict_fn(input_data, model):
    pipeline = model["model"]
    label_encoder = model["label_encoder"]
    prediction = pipeline.predict(input_data)
    prediction_label = label_encoder.inverse_transform(prediction)
    probability = None
    if hasattr(pipeline, "predict_proba"):
        probability = pipeline.predict_proba(input_data)
    return {
        "prediction": prediction_label[0],
        "probability": (probability[0].tolist() if probability is not None else None),
        "classes": label_encoder.classes_.tolist(),
    }


# ============================================================
# OUTPUT
# ============================================================


def output_fn(prediction, accept):
    return json.dumps(prediction), "application/json"
