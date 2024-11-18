# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the dataset
# For simplicity, we'll create a small dataset manually
data = {
    'SquareFeet': [850, 900, 1200, 1500, 1800, 2000, 2200, 2500, 2700, 3000],
    'Price': [150000, 160000, 200000, 250000, 300000, 330000, 360000, 400000, 420000, 500000]
}
df = pd.DataFrame(data)

# Step 2: Visualize the data
plt.scatter(df['SquareFeet'], df['Price'], color='blue')
plt.xlabel('Square Feet')
plt.ylabel('Price')
plt.title('House Price vs Square Feet')
plt.show()

# Step 3: Preprocess the data
# Split the data into features (X) and target (y)
X = df[['SquareFeet']]  # Feature
y = df['Price']         # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Step 7: Visualize the regression line
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red', linewidth=2)
plt.xlabel('Square Feet')
plt.ylabel('Price')
plt.title('Linear Regression Fit')
plt.show()