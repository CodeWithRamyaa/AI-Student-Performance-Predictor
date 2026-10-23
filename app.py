```python
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Page settings
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓"
)

st.title("🎓 AI Student Performance Predictor")
st.write("Predict student marks using study hours and attendance.")

# Sample dataset
data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Attendance": [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
    "Marks": [35, 40, 45, 50, 55, 60, 68, 75, 85, 92]
}

df = pd.DataFrame(data)

# Train model
X = df[["Study_Hours", "Attendance"]]
y = df["Marks"]

model = LinearRegression()
model.fit(X, y)

# User inputs
study_hours = st.slider(
    "Daily study hours",
    min_value=0.0,
    max_value=24.0,
    value=6.0,
    step=0.5
)

attendance = st.slider(
    "Attendance percentage",
    min_value=0,
    max_value=100,
    value=75
)

if st.button("Predict Marks"):
    input_data = pd.DataFrame(
        [[study_hours, attendance]],
        columns=["Study_Hours", "Attendance"]
    )

    prediction = float(model.predict(input_data)[0])
    marks = max(0.0, min(100.0, prediction))

    st.metric("Predicted Marks", f"{marks:.2f}/100")

    if marks >= 80:
        st.success("Performance: Excellent")
    elif marks >= 60:
        st.info("Performance: Good")
    elif marks >= 40:
        st.warning("Performance: Average")
    else:
        st.error("Performance: Needs Improvement")

# Display graph
st.subheader("Study Hours vs Marks")

fig, ax = plt.subplots()
ax.scatter(df["Study_Hours"], df["Marks"])
ax.set_xlabel("Daily Study Hours")
ax.set_ylabel("Marks")
ax.set_title("Sample Student Data")
ax.grid(True)

st.pyplot(fig)
```
