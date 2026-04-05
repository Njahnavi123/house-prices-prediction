import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("house_prices.csv")
print(data.info())
data = data[['Carpet Area', 'Bathroom', 'location', 'Price (in rupees)']]

data['Carpet Area'] = data['Carpet Area'].astype(str)
data['Carpet Area'] = data['Carpet Area'].str.extract(r'(\d+)', expand=False)
data['Carpet Area'] = pd.to_numeric(data['Carpet Area'], errors='coerce')

data['Bathroom'] = data['Bathroom'].astype(str)
data['Bathroom'] = data['Bathroom'].str.extract(r'(\d+)', expand=False)
data['Bathroom'] = pd.to_numeric(data['Bathroom'], errors='coerce')

data['location'] = data['location'].fillna('Unknown')
data['location'] = data['location'].astype('category').cat.codes

data = data.dropna(subset=['Carpet Area', 'Bathroom', 'Price (in rupees)'])
X = data[['Carpet Area', 'Bathroom', 'location']]
y = data['Price (in rupees)']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, predictions)
print("MAE:", mae)
plt.scatter(y_test, predictions, alpha=0.3)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted")
plt.show()
