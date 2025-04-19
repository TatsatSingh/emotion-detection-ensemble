# 🧠 Emotion Detection from Text using Ensemble Learning

This project is a machine learning pipeline that classifies human emotions (e.g., joy, sadness, fear) from raw text using an ensemble of models. The goal was to build a **fully local, API-free solution** using classical ML techniques enhanced through **soft-voting ensemble learning**.

---

## 📌 Problem Statement

Detect the emotional tone of a given sentence.

---

## 🔧 Tech Stack

- Python 3.x
- `scikit-learn`
- `xgboost`
- `neattext`
- `pandas`, `joblib`

---

## 🧠 Solution Architecture

1. **Text Cleaning**  
   Punctuation and stopword removal using `neattext`

2. **Feature Extraction**  
   TF-IDF Vectorizer with n-grams `(1,2)` and 15,000 features

3. **Ensemble Model**  
   - Logistic Regression
   - Calibrated LinearSVC
   - XGBoost Classifier  
   → Combined using `VotingClassifier` with **soft voting**

4. **CLI Interface** for emotion predictions

5. **Model Saving** with `joblib`

---

## 📊 Performance

- **Accuracy:** 90.45%
- **Macro F1 Score:** 0.87
- Robust performance across all classes, especially improved `love` and `surprise` handling with class balancing and ensemble voting.

---

## 🗂 Dataset

The dataset is provided in `data/` folder as:
- `train.txt`
- `val.txt`
- `test.txt`

Each line is formatted as:
<sentence>;<emotion>


---

## 🚀 How to Run


# Install dependencies
pip install -r requirements.txt

# Run the CLI model
python emotion_detection.py
