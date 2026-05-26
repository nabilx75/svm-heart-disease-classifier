# Heart Disease Prediction using SVM

A machine learning project that predicts the presence of heart disease in patients using a Support Vector Machine (SVM) classifier.

## Dataset

The project uses the **Cleveland Heart Disease dataset** (`heart_cleveland_upload.csv`) which contains 297 patient records with 13 clinical features:

| Feature | Description |
|---|---|
| age | Age of the patient |
| sex | Sex (1 = male, 0 = female) |
| cp | Chest pain type |
| trestbps | Resting blood pressure |
| chol | Serum cholesterol |
| fbs | Fasting blood sugar |
| restecg | Resting ECG results |
| thalach | Maximum heart rate achieved |
| exang | Exercise induced angina |
| oldpeak | ST depression |
| slope | Slope of peak exercise ST segment |
| ca | Number of major vessels |
| thal | Thalassemia |
| **condition** | **Target: 0 = No disease, 1 = Disease** |

## Pipeline

1. **Data loading & exploration** — shape, statistics, class distribution
2. **Preprocessing** — feature/label split, train-test split (80/20), StandardScaler normalization
3. **Model training** — SVM classifier with RBF kernel (`C=0.3`)
4. **Evaluation** — accuracy score, confusion matrix, classification report

## Results

| Set | Accuracy |
|---|---|
| Training | 88.2% |
| Test | 86.7% |

**Classification Report (Test Set):**
- Precision: 0.87 | Recall: 0.87 | F1-score: 0.87

## Tech Stack

- Python
- scikit-learn
- pandas
- numpy

## Usage

```python
# Example prediction
input_data = (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
# The model will output: 0 (No disease) or 1 (Disease present)
```
