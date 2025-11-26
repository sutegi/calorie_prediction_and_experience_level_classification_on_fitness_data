# Gym ML App — Smart Data Entry
### Predict Calories Burned & Experience Level Using Machine Learning

<p align="center">
  <img src="https://img.shields.io/badge/Framework-Streamlit-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/ML-Scikit--Learn-F7931E?logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

<p align="center">
  <img src="https://img.shields.io/github/last-commit/sutegi/calorie_prediction_and_experience_level_classification_on_fitness_data?color=blue" />
  <img src="https://img.shields.io/github/repo-size/sutegi/calorie_prediction_and_experience_level_classification_on_fitness_data" />
</p>

---

## Overview

**Gym ML App** is an interactive Streamlit application powered by two machine-learning models:

- **Regression Model** — predicts **calories burned**
- **Classification Model** — predicts **experience level** (Beginner / Intermediate / Advanced)

Features include:
- Smart auto-filling of missing inputs  
- BMI and Fat Percentage auto-calculation  
- Full sklearn preprocessing pipelines  
- Probability visualization for classification
- Clean UI with responsive layout  

---

##  Features

### Regression — *Calories Burned*
Uses personal data:
- Age, Gender, Weight, Height  
- Max BPM, Average BPM, Resting BPM  
- Workout frequency, Water intake  
- Session duration  
- Auto BMI & Fat %  

### Classification — *Experience Level*
Predicts:
- Beginner  
- Intermediate  
- Advanced  

Also shows probability distribution (if supported by the model).

### Smart Input System
- Detects missing values  
- Applies intelligent defaults  
- Ensures complete feature set for model inference  
- Handles categorical encodings automatically  

---

## Tech Stack

| Component | Technology |
|----------|------------|
| UI | Streamlit |
| ML Models | Scikit-learn |
| Data | Pandas / NumPy |
| Storage | Joblib |
| Python | 3.10+ |

---

##  Project Structure

```
├── streamlit_app.py         # Main application script
├── best_regressor.pkl       # Trained regression model bundle
├── best_classifier.pkl      # Trained classification model bundle
├── requirements.txt         # Dependencies
└── README.md                # Documentation
```

## Installation & Usage

###  Clone the repository
```bash
git clone https://github.com/sutegi/calorie_prediction_and_experience_level_classification_on_fitness_data.git
```

###  Install dependencies
```bash
pip install -r requirements.txt
```

###  Run the app
```bash
streamlit run streamlit_app.py
```

App will launch at:
```
http://localhost:8501
```

## Model Architecture

Each `.pkl` model bundle contains:

```python
{
  "model": ...,
  "preprocessor": ...,
  "features": ...,
  "class_mapping": ...  # classifier only
}
```

The app automatically:
- Reconstructs ColumnTransformer  
- Encodes categories  
- Aligns features  
- Applies scaling + OHE
