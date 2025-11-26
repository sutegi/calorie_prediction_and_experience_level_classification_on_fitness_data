import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(
    page_title="Gym ML App — Smart Input",
    layout="wide",
    page_icon="🏋️‍♂️"
)

# CSS
st.markdown("""
<style>
.pred-card {
    background-color: #ffffff;
    padding: 25px;
    border-radius: 16px;
    font-size: 26px;
    font-weight: bold;
    text-align: center;
    box-shadow: 0 6px 20px rgba(0,0,0,0.15);
}
</style>
""", unsafe_allow_html=True)

# Load Models
reg_bundle = joblib.load("best_regressor.pkl")
clf_bundle = joblib.load("best_classifier.pkl")

reg_model = reg_bundle["model"]
clf_model = clf_bundle["model"]

reg_prep = reg_bundle["preprocessor"]
clf_prep = clf_bundle["preprocessor"]

reg_features = reg_bundle["features"]
clf_features = clf_bundle["features"]

class_map = clf_bundle["class_mapping"]
inv_class = {v: k for k, v in class_map.items()}

# Extract Columns
def extract_columns(preprocessor):
    num_cols = list(preprocessor.transformers_[0][2])
    cat_cols = list(preprocessor.transformers_[1][2])
    ohe = preprocessor.transformers_[1][1]
    cat_values = ohe.categories_
    return num_cols, cat_cols, cat_values

reg_num_cols, reg_cat_cols, reg_cat_vals = extract_columns(reg_prep)
clf_num_cols, clf_cat_cols, clf_cat_vals = extract_columns(clf_prep)

# Default fallback values
def get_default_numeric(col):
    default_map = {
        "Age": 30,
        "Weight (kg)": 70,
        "Height (m)": 1.7,
        "Max_BPM": 150,
        "Avg_BPM": 120,
        "Resting_BPM": 70,
        "Session_Duration (hours)": 1.0,
        "Water_Intake (liters)": 2.0,
        "Workout_Frequency (days/week)": 3,
        "BMI": 22.0,
        "Fat_Percentage": 20.0,
    }
    return default_map.get(col, 0.0)

# Title
st.title("Gym ML App — Smart Data Entry")
st.write("Predicting calories you will burn and your experience level by information about you.")

tab_reg, tab_clf = st.tabs(["Regression (Calories Burned)", "Classification (Experience Level)"])


# Smart Regression Tab
with tab_reg:
    st.header("Predict Calories Burned (Smart Input)")

    col1, col2 = st.columns(2)

    # User input
    age = col1.number_input("Age", min_value=0, max_value=120, value=0, key="reg_age")
    gender = col2.selectbox("Gender", ["Male", "Female", "Unknown"], key="reg_gender")

    weight = col1.number_input("Weight (kg)", min_value=0.0, value=0.0, key="reg_weight")
    height = col2.number_input("Height (m)", min_value=0.0, value=0.0, key="reg_height")

    max_bpm = col1.number_input("Max BPM (optional)", min_value=0.0, value=0.0, key="reg_max_bpm")
    avg_bpm = col2.number_input("Average BPM (optional)", min_value=0.0, value=0.0, key="reg_avg_bpm")

    resting_bpm = col1.number_input("Resting BPM (optional)", min_value=0.0, value=0.0, key="reg_rest_bpm")
    session_dur = col2.number_input("Session Duration (hours)", min_value=0.0, value=0.0, key="reg_session")

    water_intake = col1.number_input("Water Intake (liters)", min_value=0.0, value=0.0, key="reg_water")
    freq = col2.number_input("Workout Frequency (days/week)", min_value=0, max_value=7, value=0, key="reg_freq")

    # Smart data dict
    data = {
        "Age": age or get_default_numeric("Age"),
        "Gender": gender,
        "Weight (kg)": weight or get_default_numeric("Weight (kg)"),
        "Height (m)": height or get_default_numeric("Height (m)"),
        "Max_BPM": max_bpm or get_default_numeric("Max_BPM"),
        "Avg_BPM": avg_bpm or get_default_numeric("Avg_BPM"),
        "Resting_BPM": resting_bpm or get_default_numeric("Resting_BPM"),
        "Session_Duration (hours)": session_dur or get_default_numeric("Session_Duration (hours)"),
        "Water_Intake (liters)": water_intake or get_default_numeric("Water_Intake (liters)"),
        "Workout_Frequency (days/week)": freq or get_default_numeric("Workout_Frequency (days/week)"),
    }

    # Auto BMI
    if height > 0:
        data["BMI"] = weight / (height ** 2)
    else:
        data["BMI"] = get_default_numeric("BMI")

    # Auto Fat_Percentage
    if gender == "Male":
        data["Fat_Percentage"] = 18
    elif gender == "Female":
        data["Fat_Percentage"] = 25
    else:
        data["Fat_Percentage"] = get_default_numeric("Fat_Percentage")

    # Auto Workout_Type
    data["Workout_Type"] = "Unknown"

    # Ensure all features exist
    for feat in reg_features:
        if feat not in data:
            if feat in reg_num_cols:
                data[feat] = get_default_numeric(feat)
            elif feat in reg_cat_cols:
                idx = reg_cat_cols.index(feat)
                data[feat] = list(reg_cat_vals[idx])[0]
            else:
                data[feat] = 0.0

    X_df = pd.DataFrame([data])
    X_processed = reg_prep.transform(X_df)

    if st.button("Predict Calories"):
        pred = reg_model.predict(X_processed)[0]

        st.markdown(f"""
        <div class="pred-card">
            Predicted Calories Burned:<br>
            <span style='color:#ff3333'>{pred:.2f}</span>
        </div>
        """, unsafe_allow_html=True)



# Smart Classification Tab
with tab_clf:
    st.header("Predict Experience Level")

    col1, col2 = st.columns(2)

    # User input
    age = col1.number_input("Age", min_value=0, value=0, key="clf_age")
    gender = col2.selectbox("Gender", ["Male", "Female", "Unknown"], key="clf_gender")

    weight = col1.number_input("Weight (kg)", min_value=0.0, value=0.0, key="clf_weight")
    height = col2.number_input("Height (m)", min_value=0.0, value=0.0, key="clf_height")

    max_bpm = col1.number_input("Max BPM (optional)", min_value=0.0, value=0.0, key="clf_max_bpm")
    avg_bpm = col2.number_input("Average BPM (optional)", min_value=0.0, value=0.0, key="clf_avg_bpm")

    resting_bpm = col1.number_input("Resting BPM (optional)", min_value=0.0, value=0.0, key="clf_rest_bpm")
    session_dur = col2.number_input("Session Duration (hours)", min_value=0.0, value=0.0, key="clf_session")

    water_intake = col1.number_input("Water Intake (liters)", min_value=0.0, value=0.0, key="clf_water")
    freq = col2.number_input("Workout Frequency", min_value=0, max_value=7, value=0, key="clf_freq")

    data = {
        "Age": age or 30,
        "Gender": gender,
        "Weight (kg)": weight or 70,
        "Height (m)": height or 1.7,
        "Max_BPM": max_bpm or 150,
        "Avg_BPM": avg_bpm or 120,
        "Resting_BPM": resting_bpm or 70,
        "Session_Duration (hours)": session_dur or 1.0,
        "Water_Intake (liters)": water_intake or 2.0,
        "Workout_Frequency (days/week)": freq or 3,
    }

    data["BMI"] = (weight / height ** 2) if height > 0 else 22.0
    data["Fat_Percentage"] = 18 if gender == "Male" else (25 if gender == "Female" else 20)
    data["Workout_Type"] = "Unknown"

    # Ensure full feature set
    for feat in clf_features:
        if feat not in data:
            if feat in clf_num_cols:
                data[feat] = get_default_numeric(feat)
            elif feat in clf_cat_cols:
                idx = clf_cat_cols.index(feat)
                data[feat] = list(clf_cat_vals[idx])[0]
            else:
                data[feat] = 0.0

    X_df = pd.DataFrame([data])
    X_processed = clf_prep.transform(X_df)

    if st.button("Predict Level"):
        pred = clf_model.predict(X_processed)[0]
        st.success(f"Experience Level: **{inv_class[pred]}**")

        if hasattr(clf_model, "predict_proba"):
            proba = clf_model.predict_proba(X_processed)[0]
            df_proba = pd.DataFrame({
                "Level": [inv_class[i] for i in range(len(proba))],
                "Probability": proba
            })
            st.dataframe(df_proba.style.background_gradient(cmap="Blues").format({"Probability": "{:.4f}"}))