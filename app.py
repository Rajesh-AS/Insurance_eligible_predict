import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="Insurance Predictor",
    page_icon="🛡️",
    layout="wide",
)

st.title("Insurance Purchase Predictor")
st.caption("A simple Streamlit app built from your age vs bought_insurance dataset.")


@st.cache_data
def load_data(csv_path: str = "insurance_data.csv") -> pd.DataFrame:
    """Load and validate the dataset."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Could not find '{csv_path}'. Put the CSV in the same folder as app.py."
        )

    df = pd.read_csv(csv_path)

    required_cols = {"age", "bought_insurance"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required column(s): {', '.join(sorted(missing))}"
        )

    df = df.loc[:, ["age", "bought_insurance"]].copy()
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["bought_insurance"] = pd.to_numeric(df["bought_insurance"], errors="coerce")

    df = df.dropna().reset_index(drop=True)
    df["bought_insurance"] = df["bought_insurance"].astype(int)

    if df.empty:
        raise ValueError("Dataset is empty after cleaning. Check your CSV file.")

    if df["bought_insurance"].nunique() < 2:
        raise ValueError(
            "The target column needs at least two classes (0 and 1) to train a classifier."
        )

    return df


@st.cache_resource
def train_model(df: pd.DataFrame) -> Tuple[LogisticRegression, StandardScaler, float]:
    """Train a logistic regression model and return model, scaler, and accuracy."""
    X = df[["age"]].values
    y = df["bought_insurance"].values

    class_counts = pd.Series(y).value_counts()
    can_stratify = class_counts.min() >= 2

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y if can_stratify else None,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred) if len(y_test) > 0 else float("nan")

    return model, scaler, accuracy


def predict_probability(model: LogisticRegression, scaler: StandardScaler, age: float) -> float:
    """Return probability of buying insurance for a single age value."""
    age_array = np.array([[age]], dtype=float)
    age_scaled = scaler.transform(age_array)
    return float(model.predict_proba(age_scaled)[0, 1])


try:
    data = load_data()
    model, scaler, test_accuracy = train_model(data)
except Exception as exc:
    st.error(str(exc))
    st.stop()

min_age = int(np.floor(data["age"].min()))
max_age = int(np.ceil(data["age"].max()))

left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("Make a Prediction")
    age_input = st.slider(
        "Select age",
        min_value=max(1, min_age - 5),
        max_value=max_age + 5,
        value=int(np.clip(data["age"].median(), min_age, max_age)),
        step=1,
    )

    predict_button = st.button("Predict", type="primary")

    if predict_button:
        prob = predict_probability(model, scaler, age_input)
        label = 1 if prob >= 0.5 else 0

        if label == 1:
            st.success("Predicted: Bought Insurance")
        else:
            st.warning("Predicted: Did Not Buy Insurance")

        st.metric("Probability of buying insurance", f"{prob * 100:.2f}%")

with right_col:
    st.subheader("Model Summary")
    st.metric("Test accuracy", f"{test_accuracy * 100:.2f}%")
    st.write("Dataset preview:")
    st.dataframe(data.head(10), use_container_width=True)

st.divider()

st.subheader("Training Data and Model Curve")

age_range = np.linspace(data["age"].min(), data["age"].max(), 300).reshape(-1, 1)
age_range_scaled = scaler.transform(age_range)
probabilities = model.predict_proba(age_range_scaled)[:, 1]

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(
    data["age"],
    data["bought_insurance"],
    label="Actual data",
    alpha=0.8,
)
ax.plot(age_range, probabilities, label="Logistic regression probability", linewidth=2)
ax.set_xlabel("Age")
ax.set_ylabel("Bought Insurance / Probability")
ax.set_title("Insurance Purchase Prediction")
ax.grid(True, alpha=0.3)
ax.legend()
st.pyplot(fig, use_container_width=True)

with st.expander("See all data"):
    st.dataframe(data, use_container_width=True)

st.info(
    "Deploy this app by placing app.py, requirements.txt, and insurance_data.csv in the same GitHub repository, then connect the repo to Streamlit Community Cloud."
)