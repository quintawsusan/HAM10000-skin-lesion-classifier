import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

from model_utils import (
    CLASS_NAMES,
    CLASS_FULL_NAMES,
    MALIGNANT_CLASSES,
    CONFIDENCE_THRESHOLD,
    preprocess_image,
    decode_prediction,
)

MODEL_PATH = "saved_model/ham10000_efficientnetb0"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


st.set_page_config(page_title="Skin Lesion Classifier", page_icon="🔬", layout="centered")

st.title("Skin Lesion Classifier")
st.markdown(
    """
This tool analyzes a dermatoscopic photograph of a pigmented skin lesion and
predicts which of 7 diagnostic categories it most likely belongs to:
Melanoma, Melanocytic nevus (common mole), Basal cell carcinoma, Dermatofibroma, Vascular lesion, Benign keratosis-like lesion and Actinic keratoses.
    
> **WARNING**: This is a student project not a medical device.
> It has not been clinically validated. Do not use it to make, delay or avoid
> any real medical decision. If you have a concerning skin lesion, kindly see a
> qualified dermatologist.
"""
)

st.divider()

uploaded_file = st.file_uploader("Upload a dermatoscopic image (JPG or PNG)", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded image", use_column_width=True)

    try:
        model = load_model()
    except Exception as e:
        st.error(
            f"Could not load the model from `{MODEL_PATH}`. "
            f"Make sure you've run the notebook's A7 section to produce it. "
            f"Details: {e}"
        )
        st.stop()

    input_batch = preprocess_image(image)
    probs = model.predict(input_batch, verbose=0)[0]
    top_class, top_class_full, confidence, per_class_probs = decode_prediction(probs)

    with col2:
        st.subheader("Prediction")

        if confidence < CONFIDENCE_THRESHOLD:
            st.warning(
                f"WARNING:**Low confidence ({confidence:.0%}).** The model is not very sure about this prediction. "
                f"The image may look different from the examples the model learned from, or the "
                f"lesion may be difficult to classify. Please treat this result with caution."
            )
        else:
            st.success(f"**{top_class_full}** ({top_class})")
            st.metric("Confidence", f"{confidence:.0%}")

        concern_note = (
            " This category is one the model associates with higher clinical concern."
            if top_class in MALIGNANT_CLASSES
            else " This category is generally considered benign."
        )
        st.caption(
            f"The model predicts this image shows signs of **{top_class_full.lower()}**, "
            f"with {confidence:.0%} confidence.{concern_note}"
        )

    st.divider()
    st.subheader("Full probability breakdown")
    st.bar_chart(
        {CLASS_FULL_NAMES[c]: per_class_probs[c] for c in CLASS_NAMES}
    )

else:
    st.info("Upload an image above to get a prediction.")

with st.expander("How to run this APP locally"):
    st.markdown(
        """
### What this app does
Loads a pretrained EfficientNetB0-based classifier trained on the HAM10000
dermatoscopic image dataset (7-class skin lesion classification) and runs
inference on an uploaded image.

### Running locally
1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv env
   source env/bin/activate
   pip install -r requirements.txt
   ```
2. Make sure `saved_model/ham10000_efficientnetb0` exists (produced by
   running `notebook.ipynb` through section A7).
3. Run:
   ```bash
   streamlit run streamlit_app.py
   ```
4. Open the local URL Streamlit prints (usually http://localhost:8501).

### Limitations
This is a student project prototype. It has not been clinically validated and must
not be used for real medical decisions.
"""
    )
