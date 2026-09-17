import os
import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

# 38 classes matching the trained dataset
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites_Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# Plant care and disease information database
DISEASE_KNOWLEDGE_BASE = {
    "Apple___Apple_scab": {
        "severity": "Moderate",
        "description": "Fungal disease causing olive-green to black velvety spots on leaves and fruit scabs.",
        "treatment": "Apply fungicides in early spring. Rake and destroy fallen leaves to reduce overwintering spores."
    },
    "Apple___Black_rot": {
        "severity": "High",
        "description": "Fungal infection causing circular brown leaf spots (frog-eye) and fruit rot.",
        "treatment": "Prune dead wood and mummified fruit. Apply copper-based fungicides during the growing season."
    },
    "Apple___Cedar_apple_rust": {
        "severity": "Moderate",
        "description": "Rust-colored or yellow-orange spots on leaves caused by Gymnosporangium fungi.",
        "treatment": "Remove nearby cedar/juniper hosts if possible. Apply preventive fungicides during leaf emergence."
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "severity": "Moderate",
        "description": "White powdery fungal growth on leaves leading to leaf curling and stunted growth.",
        "treatment": "Ensure good air circulation, prune crowded branches, and spray with sulfur or neem oil."
    },
    "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot": {
        "severity": "High",
        "description": "Rectangular tan/gray lesions running parallel between leaf veins.",
        "treatment": "Rotate crops, practice residue management, and apply foliar fungicides when conditions are humid."
    },
    "Corn_(maize)___Common_rust_": {
        "severity": "Moderate",
        "description": "Cinnamon-brown pustules scattered across upper and lower leaf surfaces.",
        "treatment": "Plant resistant hybrids and apply triazole or strobilurin fungicides if infection is early."
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "severity": "High",
        "description": "Large cigar-shaped grayish-green to tan lesions on foliage.",
        "treatment": "Use blight-resistant varieties, manage tillage debris, and apply fungicides at tassel emergence."
    },
    "Grape___Black_rot": {
        "severity": "High",
        "description": "Small circular reddish-brown leaf spots with black fruiting pycnidia, shriveling grapes.",
        "treatment": "Prune grape canopy for sunlight and air flow; spray protective fungicides from bud break to veraison."
    },
    "Grape___Esca_(Black_Measles)": {
        "severity": "High",
        "description": "Tiger-stripe leaf discoloration, wood necrosis, and speckled fruit.",
        "treatment": "Remove severely infected vines; disinfect pruning shears between cuts to avoid vascular spread."
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "severity": "Moderate",
        "description": "Irregular dark brown patches on foliage causing premature defoliation.",
        "treatment": "Apply broad-spectrum organic or copper fungicides; maintain balanced soil nutrition."
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "severity": "Critical",
        "description": "Bacterial disease spread by psyllid insects causing blotchy mottled yellow leaves and bitter, misshapen fruit.",
        "treatment": "Control Asian citrus psyllid vectors, remove infected trees, and provide balanced micronutrient nutrition."
    },
    "Peach___Bacterial_spot": {
        "severity": "High",
        "description": "Angular water-soaked leaf lesions that turn purple-black and drop out ('shot-hole' effect).",
        "treatment": "Apply copper sprays in dormancy and oxytetracycline during early fruit development."
    },
    "Pepper,_bell___Bacterial_spot": {
        "severity": "High",
        "description": "Water-soaked lesions on foliage that turn dark brown with yellow halos.",
        "treatment": "Avoid overhead watering; use copper bactericides and plant certified disease-free seeds."
    },
    "Potato___Early_blight": {
        "severity": "Moderate",
        "description": "Target-like concentric ring spots on older leaves caused by Alternaria solani.",
        "treatment": "Rotate crops, maintain adequate nitrogen, and apply mancozeb or chlorothalonil fungicides."
    },
    "Potato___Late_blight": {
        "severity": "Critical",
        "description": "Rapidly spreading water-soaked dark lesions with white mold on leaf undersides (Phytophthora infestans).",
        "treatment": "Immediately destroy affected foliage; apply systemic fungicides and avoid wet foliage."
    },
    "Squash___Powdery_mildew": {
        "severity": "Moderate",
        "description": "White talcum-powder-like fungal patches on leaf surfaces reducing photosynthesis.",
        "treatment": "Water at the soil level; apply potassium bicarbonate, neem oil, or sulfur-based sprays."
    },
    "Strawberry___Leaf_scorch": {
        "severity": "Moderate",
        "description": "Purplish to dark brown blotches covering leaves, causing foliage to dry and curl up.",
        "treatment": "Prune old infected leaves post-harvest; avoid overhead irrigation and spray preventive fungicide."
    },
    "Tomato___Bacterial_spot": {
        "severity": "High",
        "description": "Small dark brown spots with yellow margins on leaves; fruit develops raised scabby spots.",
        "treatment": "Use drip irrigation; apply copper-based bactericides combined with mancozeb."
    },
    "Tomato___Early_blight": {
        "severity": "Moderate",
        "description": "Dark brown circular spots with distinct concentric rings ('bulls-eye' pattern) on lower leaves.",
        "treatment": "Mulch around base to prevent soil splash; remove infected lower leaves; apply copper fungicide."
    },
    "Tomato___Late_blight": {
        "severity": "Critical",
        "description": "Devastating water-soaked lesions turning brown/black with pale halos and white underside fuzz.",
        "treatment": "Remove and bag infected plants immediately; ensure wide spacing and spray copper protectant."
    },
    "Tomato___Leaf_Mold": {
        "severity": "Moderate",
        "description": "Pale greenish-yellow spots on upper leaf surface with olive-brown velvety mold underneath.",
        "treatment": "Reduce humidity in greenhouses, improve airflow, and avoid wetting tomato foliage."
    },
    "Tomato___Septoria_leaf_spot": {
        "severity": "Moderate",
        "description": "Numerous small circular spots with gray centers and dark brown borders.",
        "treatment": "Crop rotation, prune bottom leaves, clean plant debris, and apply preventive organic fungicides."
    },
    "Tomato___Spider_mites_Two-spotted_spider_mite": {
        "severity": "Moderate",
        "description": "Fine yellow stippling on leaves, accompanied by fine silken webbing on leaf undersides.",
        "treatment": "Spray with insecticidal soap, neem oil, or introduce predatory mites."
    },
    "Tomato___Target_Spot": {
        "severity": "High",
        "description": "Pinpoint brown lesions enlarging into concentric target-like rings with yellow chlorotic halos.",
        "treatment": "Apply fungicides such as azoxystrobin or copper; enhance air circulation and weed control."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "severity": "High",
        "description": "Upward curling and yellowing of leaf margins with severe plant stunting, transmitted by whiteflies.",
        "treatment": "Control whitefly vectors using yellow sticky traps or insect netting; remove infected plants."
    },
    "Tomato___Tomato_mosaic_virus": {
        "severity": "High",
        "description": "Mottled dark and light green mosaic pattern on leaves with distorted leaf growth.",
        "treatment": "Sterilize tools and hands; remove infected plants; no chemical cure exists for viral infection."
    }
}

DEFAULT_HEALTHY_INFO = {
    "severity": "None",
    "description": "Leaf appears healthy, vigorous, and free from significant signs of pathogen damage.",
    "treatment": "Maintain proper watering schedule, balanced soil fertility, and inspect foliage periodically."
}


def clean_name(raw_name: str) -> dict:
    """Format raw label (e.g. 'Cherry_(including_sour)___Powdery_mildew') into clean UI names."""
    parts = raw_name.split("___")
    plant_raw = parts[0]
    condition_raw = parts[1] if len(parts) > 1 else "Unknown"

    plant_clean = plant_raw.replace("_", " ").replace("(including sour)", "").replace("(maize)", "").strip().title()
    is_healthy = "healthy" in condition_raw.lower()

    condition_clean = "Healthy & Vigorous" if is_healthy else condition_raw.replace("_", " ").strip().title()

    info = DISEASE_KNOWLEDGE_BASE.get(raw_name, DEFAULT_HEALTHY_INFO if is_healthy else {
        "severity": "Moderate",
        "description": "Foliar disease symptoms detected on leaf surface.",
        "treatment": "Isolate plant, monitor progression, and consult agricultural extension guidelines."
    })

    return {
        "raw": raw_name,
        "plant": plant_clean,
        "disease": condition_clean,
        "is_healthy": is_healthy,
        "severity": info.get("severity", "Moderate"),
        "description": info.get("description", ""),
        "treatment": info.get("treatment", "")
    }


_INTERPRETER = None
_INPUT_INDEX = None
_OUTPUT_INDEX = None
_ACTIVE_MODEL_PATH = None


def get_interpreter(model_path: str = None):
    """Initialize or return cached TFLite interpreter."""
    global _INTERPRETER, _INPUT_INDEX, _OUTPUT_INDEX, _ACTIVE_MODEL_PATH
    if _INTERPRETER is not None:
        return _INTERPRETER, _INPUT_INDEX, _OUTPUT_INDEX, _ACTIVE_MODEL_PATH

    if model_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Try float32 first, fallback to float16
        candidates = [
            os.path.join(base_dir, "best_plant_model_tf", "best_plant__model_float32.tflite"),
            os.path.join(base_dir, "best_plant_model_tf", "best_plant__model_float16.tflite"),
            os.path.join(base_dir, "best_plant__model_float32.tflite")
        ]
        for p in candidates:
            if os.path.exists(p):
                model_path = p
                break

    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"TFLite model file could not be found at {model_path}")

    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    _INPUT_INDEX = input_details[0]["index"]
    _OUTPUT_INDEX = output_details[0]["index"]
    _ACTIVE_MODEL_PATH = os.path.abspath(model_path)
    _INTERPRETER = interpreter

    return _INTERPRETER, _INPUT_INDEX, _OUTPUT_INDEX, _ACTIVE_MODEL_PATH


def preprocess_image(image_input) -> np.ndarray:
    """
    Applies the exact preprocessing pipeline from the training/evaluation notebook:
    1. Load image and convert to RGB
    2. Convert to Grayscale replicated across 3 channels (matching torchvision Grayscale(num_output_channels=3))
    3. Resize to 224x224
    4. Normalize with ImageNet mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
    5. Return float32 array shaped (1, 224, 224, 3)
    """
    if isinstance(image_input, (str, os.PathLike)):
        image = Image.open(image_input)
    elif hasattr(image_input, "read"):
        image = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        image = image_input
    else:
        raise ValueError("Unsupported image input type")

    # Match notebook: transforms.Grayscale(num_output_channels=3)
    # Convert to grayscale ('L') then to 'RGB' so all 3 channels have the gray values
    image = image.convert("L").convert("RGB")
    image = image.resize((224, 224), Image.Resampling.BILINEAR)

    # Convert to float32 in range [0, 1]
    arr = np.array(image, dtype=np.float32) / 255.0

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std

    # Expand to batch dimension (1, 224, 224, 3)
    arr = np.expand_dims(arr, axis=0).astype(np.float32)
    return arr


def predict_disease(image_input, top_k: int = 5) -> dict:
    """
    Run inference on an image using the loaded TFLite model and return structured prediction results.
    """
    import time
    interpreter, in_idx, out_idx, model_path = get_interpreter()

    # Preprocess input image into normalized tensor
    input_tensor = preprocess_image(image_input)

    # Feed input tensor into the TFLite interpreter node
    interpreter.set_tensor(in_idx, input_tensor)

    # Execute model graph computation with latency measurement
    t0 = time.perf_counter()
    interpreter.invoke()
    t1 = time.perf_counter()
    inference_ms = round((t1 - t0) * 1000, 2)

    # Extract raw output logits directly from output node
    logits = interpreter.get_tensor(out_idx)[0]  # Shape: (38,)

    # Softmax normalization over the 38 class logits
    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / np.sum(exp_logits)

    # Top-K predictions
    top_indices = np.argsort(probabilities)[::-1][:top_k]

    predictions = []
    for idx in top_indices:
        raw_name = CLASS_NAMES[idx]
        info = clean_name(raw_name)
        prob = float(probabilities[idx])
        predictions.append({
            "class_index": int(idx),
            "raw_name": raw_name,
            "plant": info["plant"],
            "disease": info["disease"],
            "is_healthy": info["is_healthy"],
            "severity": info["severity"],
            "description": info["description"],
            "treatment": info["treatment"],
            "confidence": round(prob * 100, 2),
            "probability": prob
        })

    top_prediction = predictions[0]

    return {
        "success": True,
        "predicted_class": top_prediction["raw_name"],
        "plant": top_prediction["plant"],
        "disease": top_prediction["disease"],
        "is_healthy": top_prediction["is_healthy"],
        "severity": top_prediction["severity"],
        "confidence": top_prediction["confidence"],
        "description": top_prediction["description"],
        "treatment": top_prediction["treatment"],
        "top_predictions": predictions,
        "telemetry": {
            "model_file": os.path.basename(model_path),
            "model_path": model_path,
            "model_size_mb": round(os.path.getsize(model_path) / (1024 * 1024), 2),
            "inference_time_ms": inference_ms,
            "input_tensor_shape": list(input_tensor.shape),
            "output_tensor_shape": list(logits.shape)
        }
    }
