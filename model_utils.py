import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image

IMG_SIZE = (224, 224) 

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

CLASS_FULL_NAMES = {
    "akiec": "Actinic keratoses / intraepithelial carcinoma (Bowen's disease)",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesion",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevus (common mole)",
    "vasc": "Vascular lesion",
}

MALIGNANT_CLASSES = {"mel", "bcc", "akiec"}

CONFIDENCE_THRESHOLD = 0.50

def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    img = pil_image.convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype(np.float32)
    arr = preprocess_input(arr) 
    arr = np.expand_dims(arr, axis=0)
    return arr


def decode_prediction(pred_probs: np.ndarray):
    pred_probs = np.asarray(pred_probs).flatten()
    top_idx = int(np.argmax(pred_probs))
    top_class = CLASS_NAMES[top_idx]
    confidence = float(pred_probs[top_idx])
    per_class = {CLASS_NAMES[i]: float(pred_probs[i]) for i in range(len(CLASS_NAMES))}
    return top_class, CLASS_FULL_NAMES[top_class], confidence, per_class
