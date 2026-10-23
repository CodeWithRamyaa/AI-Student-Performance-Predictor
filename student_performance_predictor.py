
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

# Select input features and target
X = df[["Study_Hours", "Attendance"]]
y = df["Marks"]

# Train the machine learning model
model = LinearRegression()
model.fit(X, y)

# Get input from the user
print("STUDENT PERFORMANCE PREDICTOR")
print("-----------------------------")

study_hours = float(input("Enter daily study hours: "))
attendance = float(input("Enter attendance percentage: "))

# Predict marks
prediction = model.predict([[study_hours, attendance]])

print(f"\nPredicted Marks: {prediction[0]:.2f} out of 100")

# Plot student data
plt.scatter(df["Study_Hours"], df["Marks"])
plt.xlabel("Daily Study Hours")
plt.ylabel("Marks")
plt.title("Study Hours vs Student Marks")
plt.grid(True)
plt.show()

