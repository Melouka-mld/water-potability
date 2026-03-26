# 💧 Water Potability Prediction

## Abstract
In recent years, the quality of water has degraded tremendously due to industrial discharge and human misuse. Unsafe drinking water can cause gastrointestinal illnesses, nervous system disorders, reproductive issues, and chronic diseases. This project predicts water potability using machine learning models and augmentation techniques.

---

## Motivation
Water pollution is rising due to population growth, agriculture, and untreated sewage. Rural areas with limited medical resources are especially vulnerable. Predicting water quality helps protect communities and supports agriculture and fish farming.

---

## Problem Statement
Testing water quality is critical in rural areas where agriculture and drinking water safety are vital. This project aims to predict potability using machine learning models and augmentation methods.

---

## Dataset and Features
The dataset contains water quality parameters:
- **pH**: 6.52–6.83 in dataset
- **Hardness**: Caused by Ca and Mg
- **Solids (TDS)**: Mineral content of water (500–1000 mg/L suitable)
- **Chloramines**: Safe up to 4 mg/L
- **Sulfate**: Common in natural substances
- **Conductivity**: Should not exceed 400 μS/cm
- **Organic Carbon**: Total organic carbon in water
- **Trihalomethanes**: Safe drinking water level ≤ 80 ppm
- **Turbidity**: Measures light scattering
- **Potability**: Target variable (0 = not potable, 1 = potable)

---

## Methods
We tested multiple algorithms:
- KNN
- Decision Tree
- Random Forest
- Naive Bayes
- Logistic Regression
- Support Vector Machine
- Stochastic Gradient Descent
- XGBoost
- Neural Network (Keras)

We also applied **data augmentation** using:
- SMOTE
- Borderline-SMOTE
- SMOTEENN
- SMOTETomek

---

## Experiments and Results
The dataset was split into train/test (70/30).  
Each algorithm was tuned with hyperparameters.  
Metrics used:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

**Observations:**
- Random Forest achieved high training accuracy but showed overfitting.
- SVM performed consistently well without overfitting.
- Neural Network achieved competitive results but requires more tuning.
- Augmentation improved recall for minority class.

---

## Conclusion
- **pH** is the most influential attribute affecting water quality.
- **SVM** provided the best balance between accuracy and generalization.
- Random Forest showed strong performance but risked overfitting.

---

## Future Work
- Collect more training samples to stabilize models.
- Tune SGD and Neural Network for better performance.
- Extend prediction to irrigation water quality in arid regions.

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/Water-Potability-Prediction.git
cd Water-Potability-Prediction
