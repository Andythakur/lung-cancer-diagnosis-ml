# 🧠 Lung Cancer Survival Predictor

![Lung Cancer Predictor](https://img.shields.io/badge/ML-Scikit--Learn-blue.svg) ![Made with Python](https://img.shields.io/badge/Made%20with-Python%203.12-yellow.svg) ![License](https://img.shields.io/badge/license-MIT-green.svg)

A powerful machine learning system that predicts lung cancer **patient survival outcomes** using real-world diagnosis data. Built using Python, scikit-learn, and advanced preprocessing techniques.

---

## 🚀 Project Highlights

- 🔍 890,000 patient records analyzed
- 📊 Handles 16+ health indicators including cancer stage, smoking status, family history, BMI, etc.
- 🎯 Model: Random Forest Classifier with feature importance insights
- 📈 Achieved ~77.75% accuracy on test set
- 📌 Fully preprocessed with label encoding and feature scaling
- 📉 Feature importance visualization using Seaborn

---

## 📁 Dataset Overview

Each patient entry includes:

- Demographics: Age, Gender, Country
- Medical data: BMI, Cholesterol, Hypertension, Asthma, Cirrhosis
- Diagnosis: Stage, Family History, Smoking Status
- Treatment: Type, Start/End Dates
- Label: **Survived (Yes/No)**

📦 File used: `dataset_med.csv`

---

## 🔧 Tech Stack

- Python 🐍
- Pandas & NumPy
- Matplotlib & Seaborn 📊
- Scikit-learn 🤖

---

## ⚙️ How It Works

1. Load and clean data
2. Encode and scale features
3. Train Random Forest model
4. Predict survival status
5. Evaluate using accuracy, confusion matrix, and classification report
6. Visualize feature impact

---

## 📸 Sample Output

```bash
Accuracy: 0.7775
Confusion Matrix:
 [[138315    324]
 [ 39274     87]]
```

---

## 📌 Future Improvements

- Hyperparameter tuning with GridSearchCV
- Support for real-time predictions with UI
- Add deep learning model comparison (e.g., with TensorFlow/Keras)
- Build a Flask/Streamlit interface

---

## 🙌 Contributing

Pull requests and ideas welcome! Open an issue or drop a star ⭐ if you liked this.

---

## 📄 License

This project is open source under the [MIT License](LICENSE).

---
