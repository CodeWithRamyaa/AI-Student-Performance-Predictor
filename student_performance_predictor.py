import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample student data
data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Attendance": [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
    "Marks": [35, 40, 45, 50, 55, 60, 68, 75, 85, 92]
}

df = pd.DataFrame(data)

# Prepare the training data
X = df[["Study_Hours", "Attendance"]]
y = df["Marks"]

# Train the machine learning model
model = LinearRegression()
model.fit(X, y)

# Get user input
print("STUDENT PERFORMANCE PREDICTOR")
print("-----------------------------")

study_hours = float(input("Enter daily study hours: "))
attendance = float(input("Enter attendance percentage: "))

# Validate input
if not 0 <= study_hours <= 24:
    print("Please enter study hours between 0 and 24.")
elif not 0 <= attendance <= 100:
    print("Please enter attendance between 0 and 100.")
else:
    # Predict marks
    prediction = model.predict(
        pd.DataFrame(
            [[study_hours, attendance]],
            columns=["Study_Hours", "Attendance"]
        )
    )[0]

    # Keep predicted marks within 0–100
    marks = max(0, min(100, prediction))

    print(f"\nPredicted Marks: {marks:.2f} out of 100")

    # Display performance category
    if marks >= 80:
        print("Performance: Excellent")
    elif marks >= 60:
        print("Performance: Good")
    elif marks >= 40:
        print("Performance: Average")
    else:
        print("Performance: Needs Improvement")

    # Plot study hours versus marks
    plt.scatter(df["Study_Hours"], df["Marks"])
    plt.xlabel("Daily Study Hours")
    plt.ylabel("Marks")
    plt.title("Study Hours vs Student Marks")
    plt.grid(True)
    plt.show()

