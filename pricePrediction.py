import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model    import LinearRegression
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score
from sympy.benchmarks.bench_meijerint import alpha
from torch.optim.radam import radam

housing_data = pd.read_csv("USA_Housing.csv")

features = housing_data[['Avg. Area Income', 'Avg. Area House Age',
                         'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms',
                         'Area Population']]

output_prediction = housing_data['Price']

X_train, X_test, y_train, y_test = train_test_split(features, output_prediction, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

pd.DataFrame(model.coef_, features.columns, columns=['Coefficient'])

house_price_prediction = model.predict(X_test)

mae = mean_absolute_error(y_test, house_price_prediction)

mse = mean_squared_error(y_test, house_price_prediction)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, house_price_prediction)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=house_price_prediction, alpha=0.7)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()

sns.histplot(y_test - house_price_prediction, bins=30, kde=True)
plt.title("Residual Distribution")
# plt.show()

# print("Model Performance Summary")
# print(f"MAE: {mae:.2f}")
# print(f"RMSE: {rmse:.2f}")
# print(f"R² Score: {r2:.2f}")

