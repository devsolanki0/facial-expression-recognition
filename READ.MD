# 🧠 FaceSense AI

### Real-Time Facial Expression Recognition Using Deep Learning

FaceSense AI is a real-time facial expression recognition system built using **Deep Learning, TensorFlow/Keras, and OpenCV**.

The system uses a custom **Convolutional Neural Network (CNN)** trained on the **FER-2013 dataset** to recognize seven facial expressions from a live webcam feed.

---

## ✨ Features

* 🎥 Real-time webcam facial expression detection
* 🧠 Custom CNN Deep Learning model
* 👤 Automatic face detection
* 😊 7 facial expression classes
* 📊 Prediction confidence score
* ⚡ Real-time FPS counter
* 🖼️ 48×48 grayscale image preprocessing
* 📈 Model evaluation with accuracy, precision, recall and F1-score
* 🔥 Confusion matrix analysis
* 💾 Saved `.keras` trained model

### Supported Expressions

| Expression  | Class |
| ----------- | ----: |
| 😠 Angry    |     0 |
| 🤢 Disgust  |     1 |
| 😨 Fear     |     2 |
| 😀 Happy    |     3 |
| 😐 Neutral  |     4 |
| 😢 Sad      |     5 |
| 😲 Surprise |     6 |

---

# 🏗️ Project Architecture

```text
                    ┌───────────────────┐
                    │    Webcam Feed    │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │  Face Detection   │
                    │      OpenCV       │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │   Face Cropping   │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │  Preprocessing    │
                    │  48 × 48 Gray     │
                    │   Normalization   │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │    CNN Model      │
                    │   TensorFlow      │
                    │      Keras        │
                    └─────────┬─────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ Expression Prediction   │
                 │ + Confidence Score      │
                 └─────────────────────────┘
```

---

# 📂 Project Structure

```text
Facial-Expression/
│
├── dataset/
│   ├── train/
│   │   ├── angry/
│   │   ├── disgust/
│   │   ├── fear/
│   │   ├── happy/
│   │   ├── neutral/
│   │   ├── sad/
│   │   └── surprise/
│   │
│   └── test/
│       ├── angry/
│       ├── disgust/
│       ├── fear/
│       ├── happy/
│       ├── neutral/
│       ├── sad/
│       └── surprise/
│
├── notebooks/
│   └── facial_expression.ipynb
│
├── models/
│   └── best_facial_expression_model.keras
│
├── app/
│   └── realtime_camera.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 📊 Dataset

This project uses the **FER-2013 (Facial Expression Recognition 2013)** dataset.

The dataset contains grayscale facial images categorized into seven expressions.

### Dataset Classes

```text
Angry
Disgust
Fear
Happy
Neutral
Sad
Surprise
```

### Image Properties

```text
Image Size: 48 × 48
Channels: 1
Color: Grayscale
Classes: 7
```

Dataset:

**FER-2013 — Kaggle**

https://www.kaggle.com/datasets/msambare/fer2013

> The dataset is not included in this repository because of its size. Download it separately and place the `train` and `test` directories inside the `dataset/` folder.

---

# 🛠️ Technologies Used

| Technology       | Purpose                        |
| ---------------- | ------------------------------ |
| Python           | Programming language           |
| TensorFlow       | Deep Learning framework        |
| Keras            | CNN model development          |
| OpenCV           | Webcam & face detection        |
| NumPy            | Numerical processing           |
| Matplotlib       | Visualization                  |
| Seaborn          | Confusion matrix visualization |
| Scikit-learn     | Model evaluation               |
| Jupyter Notebook | Model development              |

---

# 🧠 CNN Architecture

The project uses a custom Convolutional Neural Network.

```text
Input
48 × 48 × 1
       │
       ▼
Conv2D - 32 filters
       │
Batch Normalization
       │
Conv2D - 32 filters
       │
MaxPooling
       │
Dropout
       │
       ▼
Conv2D - 64 filters
       │
Batch Normalization
       │
Conv2D - 64 filters
       │
MaxPooling
       │
Dropout
       │
       ▼
Conv2D - 128 filters
       │
Batch Normalization
       │
Conv2D - 128 filters
       │
MaxPooling
       │
Dropout
       │
       ▼
Flatten
       │
Dense - 256
       │
Batch Normalization
       │
Dropout
       │
       ▼
Dense - 7
       │
Softmax
       │
       ▼
Expression
```

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Facial-Expression.git
```

Move into the project:

```bash
cd Facial-Expression
```

---

## 2. Create a virtual environment

### Windows

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

# 📦 Install Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

If you haven't created `requirements.txt`, use:

```bash
pip install tensorflow opencv-python==4.11.0.86 numpy pandas matplotlib seaborn scikit-learn pillow
```

### Recommended OpenCV version

This project uses:

```text
opencv-python==4.11.0.86
```

OpenCV 4 is used because the project uses:

```python
cv2.CascadeClassifier()
```

---

# 📁 Dataset Setup

After downloading FER-2013, arrange your folders like this:

```text
Facial-Expression/
│
├── dataset/
│   ├── train/
│   │   ├── angry/
│   │   ├── disgust/
│   │   ├── fear/
│   │   ├── happy/
│   │   ├── neutral/
│   │   ├── sad/
│   │   └── surprise/
│   │
│   └── test/
│       ├── angry/
│       ├── disgust/
│       ├── fear/
│       ├── happy/
│       ├── neutral/
│       ├── sad/
│       └── surprise/
```

---

# 🧪 Training the Model

Open the Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
notebooks/facial_expression.ipynb
```

The notebook performs:

```text
Dataset Loading
      ↓
Data Exploration
      ↓
Visualization
      ↓
Preprocessing
      ↓
Data Augmentation
      ↓
CNN Construction
      ↓
Model Training
      ↓
Validation
      ↓
Evaluation
      ↓
Model Saving
```

The trained model is saved as:

```text
models/best_facial_expression_model.keras
```

---

# 🎥 Run Real-Time Facial Expression Recognition

Make sure the trained model exists:

```text
models/best_facial_expression_model.keras
```

From the project root directory run:

```bash
python app\realtime_camera.py
```

For example:

```text
C:\Users\devds\Documents\Facial-Expression>
```

Then:

```bash
python app\realtime_camera.py
```

Your webcam should open automatically.

---

# 🎮 Controls

| Key | Action               |
| --- | -------------------- |
| `Q` | Quit the application |

---

# 🖥️ Real-Time Output

The application detects a face and displays:

```text
Happy: 91.4%
```

along with:

* Face bounding box
* Predicted expression
* Confidence score
* FPS

Example:

```text
┌───────────────────────────────┐
│ FPS: 27.5                     │
│                               │
│       ┌───────────────┐       │
│       │               │       │
│       │     FACE      │       │
│       │               │       │
│       └───────────────┘       │
│                               │
│       Happy: 91.4%            │
└───────────────────────────────┘
```

---

# 📈 Model Evaluation

The project evaluates the model using:

### Accuracy

Measures the overall percentage of correctly classified expressions.

### Precision

Measures how often a predicted expression is correct.

### Recall

Measures how effectively the model detects each expression.

### F1-Score

Combines precision and recall.

### Confusion Matrix

Shows which facial expressions are correctly classified and which expressions are confused with one another.

---

# 🔬 Preprocessing Pipeline

Every detected face goes through:

```text
Webcam Frame
      ↓
Face Detection
      ↓
Face Crop
      ↓
Grayscale Conversion
      ↓
Resize → 48 × 48
      ↓
Pixel Normalization
      ↓
CNN
      ↓
Softmax Prediction
```

Pixel values are normalized from:

```text
0–255
```

to:

```text
0–1
```

---

# 🚀 Future Improvements

Possible improvements for future versions:

* [ ] Multiple face tracking
* [ ] Better face detection using modern detectors
* [ ] Real-time emotion history
* [ ] Expression frequency dashboard
* [ ] Streamlit web application
* [ ] GPU acceleration
* [ ] Mobile deployment
* [ ] Transfer learning with EfficientNet/MobileNet
* [ ] Improved performance under different lighting conditions
* [ ] Emotion timeline visualization

---

# ⚠️ Limitations

Facial expression recognition is affected by:

* Lighting conditions
* Camera quality
* Face orientation
* Occlusion
* Facial appearance
* Dataset limitations

The predicted expression should therefore be considered a **model prediction**, not a definitive assessment of a person's internal emotional state.

---

# 🎯 Learning Outcomes

Through this project, the following concepts were implemented:

* Convolutional Neural Networks
* Image preprocessing
* Data augmentation
* Batch normalization
* Dropout
* Softmax classification
* Model checkpointing
* Early stopping
* Learning-rate scheduling
* Computer vision
* Face detection
* Real-time inference
* Model evaluation
* Confusion matrix analysis
* Webcam-based AI applications

---

# 👨‍💻 Author

**Dev Solanki**

Computer Science & Engineering | AI/ML & Data Analytics

GitHub:
https://github.com/devsolanki0

---

# ⭐ If you found this project useful

Give the repository a ⭐ and feel free to explore, modify, and improve the project.

---

## 📜 License

This project is intended for educational and portfolio purposes.
