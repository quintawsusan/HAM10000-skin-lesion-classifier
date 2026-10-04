# HAM10000 Skin Lesion Classifier

A deep learning image classification project that uses **EfficientNetB0 transfer learning** to classify dermatoscopic skin lesion images into seven diagnostic categories from the HAM10000 dataset.

The project includes a Jupyter notebook for data preparation, training and evaluation, as well as a Streamlit web application for making predictions on individual images.

> **Warning:** This is a student machine-learning project for educational purposes. It is not a medical device, has not been clinically validated and should not be used to diagnose skin conditions or make medical decisions.


## Project Overview

Skin lesion classification is an image classification problem where a model learns to identify patterns in dermatoscopic images and assign them to predefined diagnostic categories.

This project uses the **HAM10000 dataset** and an **EfficientNetB0-based neural network** trained using transfer learning.

The model predicts one of seven classes:

| Code    | Description                                     |
| ------- | ----------------------------------------------- |
| `akiec` | Actinic keratoses  |
| `bcc`   | Basal cell carcinoma                            |
| `bkl`   | Benign keratosis-like lesions                   |
| `df`    | Dermatofibroma                                  |
| `mel`   | Melanoma                                        |
| `nv`    | Melanocytic news (common mole)                              |
| `vasc`  | Vascular lesions                                |

The trained model is then connected to a Streamlit application where a user can upload an image and receive the model's predicted class and confidence.


## Project Workflow

The project follows this general machine-learning pipeline:

```text
HAM10000 Dataset
       ↓
Data loading and sanity checks
       ↓
Preprocessing
       ↓
Lesion-level train/validation/test split
       ↓
Data augmentation
       ↓
EfficientNetB0 transfer learning
       ↓
Model training
       ↓
Model evaluation
       ↓
Saved trained model
       ↓
Streamlit application
       ↓
Image upload → Prediction → Confidence
```


## Dataset

This project uses the **HAM10000 ("Human Against Machine with 10000 training images")** dataset which as organized as:

```text
data/
├── images/
└── HAM10000_metadata.csv
```

The metadata file contains information associated with the images, including the lesion identifier and diagnostic class.

### Why the split is done using `lesion_id`

The dataset can contain multiple images associated with the same physical lesion.

For example:

```text
Lesion A
├── image_1
├── image_2
└── image_3
```

If individual images were randomly divided between training and testing, images from the same lesion could appear in both sets causing data leakage, because the model would be evaluated on an image related to something it had already seen during training.

To reduce this problem, the project performs the dataset split using `lesion_id`, keeping images belonging to the same lesion within the same dataset split.

The intended split is:
* 70% training
* 15% validation
* 15% testing


## Model

The project uses EfficientNetB0 with transfer learning.

EfficientNetB0 provides a pretrained image feature extractor, which is then adapted to the seven skin-lesion classes in this project.

The final model produces a probability distribution across the seven classes.

For example:

```text
mel   → 0.72
nv    → 0.15
bkl   → 0.06
bcc   → 0.04
```
The class with the highest predicted probability is selected as the model's top prediction.


## Training

The model is trained using the training dataset while the validation dataset is used to monitor performance during training.

The project also accounts for the class imbalance present in the dataset by using class weights during training.

Training includes image augmentation to expose the model to variations in the training images.

The augmentation pipeline includes operations such as:

* Random flipping
* Random rotation
* Random zoom


## Evaluation

After training, the model is evaluated using the held-out test dataset.

The notebook produces evaluation results and example predictions to help assess how the model performs on images that were not used for training.

Because this is a medical image classification problem with imbalanced classes, accuracy should not be considered the only useful measure of model performance.

The notebook also examines classification performance using metrics such as:

* Precision
* Recall
* F1-score
* Confusion matrix

Particular attention should be given to the performance of individual classes, including melanoma.


## Project Structure

```text
skin classification/
│
├── data/
│   ├── images/
│   └── HAM10000_metadata.csv
│
├── notebook.ipynb
│
├── model_utils.py
│
├── streamlit_app.py
│
├── requirements.txt
│
├── saved_model/
│
└── README.md
```

### File descriptions

| File / Folder      | Purpose                                                          |
| ------------------ | ---------------------------------------------------------------- |
| `data/`            | Dataset and metadata                                             |
| `notebook.ipynb`   | Data preparation, training and evaluation                        |
| `model_utils.py`   | Shared preprocessing, class definitions and prediction utilities |
| `streamlit_app.py` | Streamlit interface for model predictions                        |
| `requirements.txt` | Python dependencies                                              |
| `saved_model/`     | Location of the trained model                                    |
| `README.md`        | Project documentation                                            |


## Requirements

The project requires Python and the packages listed in:

```text
requirements.txt
```

The main components include:

* TensorFlow
* Streamlit
* NumPy
* Pillow
* Pandas
* Scikit-learn
* Matplotlib

Install the dependencies using:

```bash
pip install -r requirements.txt
```


## Installation

### 1. Clone the repository

```bash
git clone <GITHUB_REPOSITORY_URL>
cd <PROJECT_FOLDER>
```

### 2. Create a virtual environment

```bash
python3 -m venv env
```

### 3. Activate the environment

On Linux/macOS:

```bash
source env/bin/activate
```

On Windows:

```bash
env\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```


## Dataset Setup

Place the HAM10000 dataset inside the `data/` directory:

```text
data/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
└── HAM10000_metadata.csv
```

The dataset itself is not included in this repository.


## Train the Model

Open:

```text
notebook.ipynb
```

Run the notebook cells from top to bottom.

The notebook performs the following steps:

1. Loads the HAM10000 metadata and images.
2. Performs data and image sanity checks.
3. Splits the data using `lesion_id`.
4. Prepares the training, validation and test datasets.
5. Applies training-time image augmentation.
6. Creates the EfficientNetB0-based classifier.
7. Calculates class weights.
8. Trains the model.
9. Evaluates the trained model.
10. Produces example predictions.
11. Saves the trained model.

The intended model output is:

```text
saved_model/ham10000_efficientnetb0.keras
```


## Run the Streamlit Application

After the model has been trained and saved, run:

```bash
streamlit run streamlit_app.py
```

Streamlit will provide a local address in the terminal.

Open that address in your browser.

The application allows the user to:

1. Upload a skin lesion image.
2. Preprocess the image.
3. Run the image through the trained model.
4. Display the predicted lesion class.
5. Display the model's confidence.
6. Display the probability for each class.



## Confidence Warning

The application includes a confidence threshold.

When the model's highest predicted probability is below the configured threshold, the application warns the user that the model is uncertain about the prediction.


**Warning** A high confidence score should not be interpreted as proof that the prediction is medically correct.



## Limitations

This project has several important limitations.

### Dataset limitations

The model was trained using the HAM10000 dataset, so it may not work as well on images that look very different from the images it learnt from.

Things like image quality, camera or equipment used, lighting, skin appearance and differences in the skin lesion may affect the model's prediction.


### Model limitations

The model can make incorrect predictions.

The confidence score represents the model's predicted probability distribution and does not guarantee correctness.

### Clinical limitations

This project has not been clinically validated.

It should not be used to:

* Diagnose a skin condition
* Decide whether a lesion is cancerous
* Replace a dermatologist
* Guide medical treatment
* Make medical decisions

Anyone concerned about a skin lesion should seek advice from a qualified healthcare professional.


## Technologies Used

* **Python**
* **TensorFlow / Keras**
* **EfficientNetB0**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Pillow**
* **Matplotlib**
* **Jupyter Notebook**
* **Streamlit**
