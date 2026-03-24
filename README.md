# 🌸 Intro to Machine Learning - TensorFlow Project

Project code for Udacity's Intro to Machine Learning with TensorFlow
Nanodegree program. In this project, you will first develop code for an
image classifier built with TensorFlow, then convert it into a
command-line application.

------------------------------------------------------------------------

## 📊 Data

The dataset used in this project is the **Oxford Flowers 102 dataset**.

-   102 flower categories
-   \~20 images per class
-   Large dataset (not included in this repository)

**Note:** - The dataset is not uploaded to GitHub due to size
limitations. - It is recommended to use Udacity GPU workspaces or a
local machine with GPU for training.

------------------------------------------------------------------------

## 🚀 My Implementation

This project implements a complete **end-to-end image classification
pipeline**:

### Key Features

-   Load data using **TensorFlow Datasets**
-   Image preprocessing (resize, normalize, batch)
-   Transfer learning using **MobileNet (TensorFlow Hub)**
-   Custom classifier for 102 classes
-   Model training with validation monitoring
-   Performance visualization (accuracy and loss)
-   Final evaluation on test data

------------------------------------------------------------------------

## 🧠 Model

-   Base Model: MobileNet (pre-trained)
-   Input Size: 224 × 224 × 3
-   Output: 102 flower classes
-   Saved as: `clf_model.h5`

------------------------------------------------------------------------

## ⚙️ Command Line Application

After training, the model is used in a CLI tool to make predictions.

### Features

-   Predict flower class from an image
-   Return top K predictions
-   Map class indices to flower names using JSON
-   Clean formatted output

------------------------------------------------------------------------

## ▶️ Usage

### Basic prediction

``` bash
python predict.py test_images/image_1.jpg clf_model.h5
```

### Top K predictions

``` bash
python predict.py test_images/image_1.jpg clf_model.h5 --top_k 5
```

### With flower names

``` bash
python predict.py test_images/image_1.jpg clf_model.h5 --top_k 5 --category_names label_map.json
```

------------------------------------------------------------------------

## 📊 Example Output

``` text
FLOWER SPECIES           | PROBABILITY
----------------------------------------
pink primrose            | 92.45%
hard-leaved orchid       | 3.12%
...
```

------------------------------------------------------------------------

## 📁 Project Structure

``` text
.
├── clf_model.h5
├── label_map.json
├── Project_Image_Classifier_Project.ipynb
├── Project_Image_Classifier_Project.html
├── predict.py
├── test_images/
├── assets/
└── README.md
```

------------------------------------------------------------------------

## 🧰 Technologies Used

-   Python
-   TensorFlow / Keras
-   TensorFlow Hub
-   NumPy
-   Pillow (PIL)

------------------------------------------------------------------------

## 📌 Notes

-   Model expects input images of size **224x224**
-   Use `label_map.json` for correct flower names
-   Class indices may differ depending on label mapping

------------------------------------------------------------------------

## 👨‍💻 Author

**Mustafa Taha**\
Full Stack Developer \| AI Engineer \| Data Analyst
