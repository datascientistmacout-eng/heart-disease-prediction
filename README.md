# Heart Disease Prediction

**ML Zoomcamp Project** – Binary classification to predict presence of heart disease.

## Problem Description

Heart disease is one of the leading causes of death worldwide. Early detection can significantly improve outcomes.  

This project builds a machine learning model that predicts whether a patient is likely to have heart disease based on 13 clinical features (age, sex, chest pain type, resting blood pressure, cholesterol, etc.).

**How the model can be used**:  
A simple web service that accepts patient measurements and returns a probability / risk flag. This can serve as a quick screening tool for clinicians or as part of a larger health dashboard.

**Dataset**: UCI Cleveland Heart Disease dataset (processed version).  
- 297 patients after cleaning  
- Target: `target` (0 = no disease, 1 = disease)

## Project Structure

```
heart-disease-prediction/
├── data/
│   ├── heart.csv              # original
│   └── heart_clean.csv        # cleaned + binarized target
├── models/                    # saved model artifacts
├── notebook.ipynb             # EDA + model selection + tuning
├── train.py                   # training script
├── predict.py                 # Flask web service
├── requirements.txt
├── Dockerfile
└── README.md
```

## How to Run

### 1. Setup environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# or .venv\Scripts\activate on Windows

pip install -r requirements.txt
```

### 2. Train the model

```bash
python train.py
```

This will:
- Load the cleaned data
- Train Logistic Regression + Random Forest (with hyperparameter tuning)
- Select the best model
- Save it to `models/model.bin` (pickle) + feature names

### 3. Run the web service locally

```bash
python predict.py
```

Service runs at `http://localhost:9696`

**Example request**:

```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63,
    "sex": 1,
    "cp": 3,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 0,
    "ca": 0,
    "thal": 1
  }'
```

### 4. Run with Docker

```bash
# Build
docker build -t heart-disease-prediction .

# Run
docker run -it --rm -p 9696:9696 heart-disease-prediction
```

Then use the same `curl` command as above.

## Model Performance (approximate)

| Model              | Accuracy | ROC-AUC |
|--------------------|----------|---------|
| Logistic Regression| ~0.83    | ~0.90   |
| Random Forest (tuned) | ~0.85 | ~0.91   |

(Exact numbers depend on the random seed and train/test split.)

## Features used

- `age`, `sex`, `cp` (chest pain type), `trestbps`, `chol`, `fbs`,  
  `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`

## Notes for reviewers

- Full EDA + feature importance is in `notebook.ipynb`
- Multiple models trained + hyperparameter tuning performed
- Training logic extracted to `train.py`
- Model served with Flask
- Fully containerized with Dockerfile
- Clear instructions to reproduce everything

## Cloud deployment (bonus)

You can easily deploy this to Render, Railway, or Fly.io by connecting the GitHub repo and using the Dockerfile.  
Example (Render):  
1. Push this repo to GitHub  
2. Create a new Web Service on Render  
3. Select Docker  
4. Set port 9696

---

Created for **ML Zoomcamp**.
