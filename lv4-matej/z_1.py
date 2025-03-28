
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data = pd.read_csv('data_C02_emission.csv')

numericke_varijable = [
    'Engine Size (L)',
    'Cylinders',
    'Fuel Consumption City (L/100km)',
    'Fuel Consumption Hwy (L/100km)',
    'Fuel Consumption Comb (L/100km)',
    'Fuel Consumption Comb (mpg)'
]

X = data[numericke_varijable]
y = data['CO2 Emissions (g/km)']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)


plt.figure(figsize=(10, 6))
plt.scatter(X_train['Engine Size (L)'], y_train, c='blue', label='Train', alpha=0.5)
plt.scatter(X_test['Engine Size (L)'], y_test, c='red', label='Test', alpha=0.5)
plt.xlabel('Engine Size (L)')
plt.ylabel('CO2 Emissions (g/km)')
plt.legend()
plt.title('Ovisnost emisije CO2 o veličini motora')
plt.show()


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(X_train['Engine Size (L)'], bins=20, color='blue')
plt.title('Prije skaliranja')

plt.subplot(1, 2, 2)
plt.hist(X_train_scaled[:, 0], bins=20, color='red')
plt.title('Nakon skaliranja')
plt.show()


model = LinearRegression()
model.fit(X_train_scaled, y_train)
print("Parametri modela:")
print(f"Intercept (theta0): {model.intercept_}")
for i, coef in enumerate(model.coef_):
    print(f"theta{i+1} ({numericke_varijable[i]}): {coef:.2f}")


y_pred = model.predict(X_test_scaled)

plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
plt.xlabel('Stvarne vrijednosti')
plt.ylabel('Predviđene vrijednosti')
plt.title('Stvarne vs. predviđene emisije CO2')
plt.show()


mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nEvaluacijske metrike:")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.2f}")



varijable_za_test = [
    ['Engine Size (L)'],
    ['Engine Size (L)', 'Cylinders'],
    numericke_varijable  # sve varijable
]



print("\nUtjecaj broja varijabli na R²:")
for vars in varijable_za_test:
    X_sub = X[vars]
    X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(
        X_sub, y, test_size=0.2, random_state=1
    )
    scaler_sub = StandardScaler()
    X_train_sub_scaled = scaler_sub.fit_transform(X_train_sub)
    X_test_sub_scaled = scaler_sub.transform(X_test_sub)
    
    model_sub = LinearRegression()
    model_sub.fit(X_train_sub_scaled, y_train_sub)
    y_pred_sub = model_sub.predict(X_test_sub_scaled)
    r2_sub = r2_score(y_test_sub, y_pred_sub)
    print(f"Broj varijabli: {len(vars)}, R²: {r2_sub:.2f}")
