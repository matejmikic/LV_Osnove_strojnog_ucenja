
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
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
kategoricka_varijabla = ['Fuel Type']

X = data[numericke_varijable + kategoricka_varijabla]
y = data['CO2 Emissions (g/km)']

preprocessor = ColumnTransformer(
    transformers=[
        ('numericke', 'passthrough', numericke_varijable),
        ('kategoricke', OneHotEncoder(), kategoricka_varijabla)
    ])

X_processed = preprocessor.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, 
    test_size=0.2, 
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
max_error = np.max(np.abs(y_test - y_pred))

test_data = data.iloc[y_test.index].copy()
test_data['Predikcija'] = y_pred
test_data['Pogreska'] = np.abs(y_test - y_pred)
max_error_row = test_data.loc[test_data['Pogreska'].idxmax()]

print("METRIKE EVALUACIJE:")
print(f"- MSE: {mse:.2f}")
print(f"- MAE: {mae:.2f}")
print(f"- R²: {r2:.2f}")
print(f"- Maksimalna pogreška: {max_error:.2f} g/km\n")

print("VOZILO S NAJVEĆOM POGREŠKOM:")
print(f"- Proizvođač: {max_error_row['Make']}")
print(f"- Model: {max_error_row['Model']}")
print(f"- Tip goriva: {max_error_row['Fuel Type']}")
print(f"- Stvarna emisija: {max_error_row['CO2 Emissions (g/km)']} g/km")
print(f"- Predviđena emisija: {max_error_row['Predikcija']:.2f} g/km")
print(f"- Pogreška: {max_error_row['Pogreska']:.2f} g/km")