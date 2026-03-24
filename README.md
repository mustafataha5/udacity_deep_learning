# 🌸 Intro to Machine Learning with TensorFlow — Image Classifier Project

This repository contains my solution for the **Udacity Intro to Machine Learning with TensorFlow Nanodegree** image classifier project.  
The project builds a flower image classifier using **TensorFlow** and **TensorFlow Hub**, then converts it into a **command-line application** for prediction.

---

## 📊 Dataset

This project uses the **Oxford Flowers 102** dataset.

- **102 flower categories**
- Approximately **20 images per class**
- Dataset is **not included** in this repository because of its size

### Notes
- It is recommended to train the model using a **GPU-enabled environment** when possible
- You can use a **Udacity workspace**, **Kaggle**, or a **local machine**
- Prediction can also run on **CPU**

---

## 🚀 Project Overview

This project implements a complete **end-to-end image classification pipeline**:

- Load data using **TensorFlow Datasets**
- Preprocess images (resize, normalize, batch)
- Use **transfer learning** with **MobileNet** from **TensorFlow Hub**
- Build a custom classifier for **102 flower classes**
- Train the model with validation monitoring
- Visualize model performance with training/validation accuracy and loss
- Evaluate final performance on the test set
- Save the trained model for reuse
- Predict flower classes from the command line

---

## 🧠 Model Details

- **Base Model:** MobileNet (pre-trained)
- **Input Shape:** `224 x 224 x 3`
- **Number of Classes:** `102`
- **Saved Model File:** `clf_model.h5`

---

## ⚙️ Environment Setup

This project was developed and tested with the following setup:

- **Python:** 3.10
- **TensorFlow:** 2.16.x

> ⚠️ **Important:** TensorFlow 2.16.x may not install correctly on some newer Python versions.  
> For best compatibility, use **Python 3.10** or **Python 3.11**.

### 1) Clone the repository

```bash
git clone https://github.com/mustafataha5/udacity_deep_learning.git
cd udacity_deep_learning
```

### 2) Create and activate a virtual environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

### 3) Upgrade pip

```bash
pip install --upgrade pip
```

### 4) Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Suggested Requirements

A compatible `requirements.txt` can look like this:

```txt
tensorflow==2.16.1
tensorflow-datasets==4.9.4
tensorflow-hub==0.16.1
tf-keras==2.16.0
scipy==1.12.0
numpy
pillow
matplotlib
```

---

## ▶️ Usage

## Basic prediction

```bash
python predict.py test_images/image_1.jpg clf_model.h5
```

## Top K predictions

```bash
python predict.py test_images/image_1.jpg clf_model.h5 --top_k 5
```

## Predict with flower names

```bash
python predict.py test_images/image_1.jpg clf_model.h5 --top_k 5 --category_names label_map.json
```

---

## 📊 Example Output

```text
FLOWER SPECIES           | PROBABILITY
----------------------------------------
pink primrose            | 92.45%
hard-leaved orchid       | 3.12%
...
```

---

## 📁 Project Structure

```text
.
├── clf_model.h5
├── label_map.json
├── Project_Image_Classifier_Project.ipynb
├── Project_Image_Classifier_Project.html
├── predict.py
├── test_images/
├── assets/
├── requirements.txt
└── README.md
```

---

## 🧰 Technologies Used

- Python
- TensorFlow / Keras
- TensorFlow Hub
- TensorFlow Datasets
- NumPy
- Pillow (PIL)
- Matplotlib

---

## 📌 Notes

- The model expects input images of size **224 × 224**
- Use `label_map.json` to map prediction indices to flower names
- Class indices may vary depending on the label mapping used
- Training is faster with GPU, but inference works on CPU as well

---

## ✅ What This Project Demonstrates

This project demonstrates practical skills in:

- Deep learning workflow design
- Image preprocessing
- Transfer learning
- Model evaluation
- Saving and loading TensorFlow models
- Building a command-line ML application
- Writing reproducible project documentation

---

## 👨‍💻 Author

**Mustafa Taha**  
Full-Stack Developer | AI Engineer | Data Analyst

- GitHub: https://github.com/mustafataha5
- LinkedIn: https://www.linkedin.com/in/mustafa-taha-ai/
