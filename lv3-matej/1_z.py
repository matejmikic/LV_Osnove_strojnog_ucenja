import pandas as pd

df = pd.read_csv('data_C02_emission.csv')


print("a)")
print("Broj mjerenja:", df.shape[0])
print("Tipovi veličina:\n", df.dtypes)
df.drop_duplicates(inplace=True) 
categorical_columns = ['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type']
df[categorical_columns] = df[categorical_columns].astype('category')



print("\nb)")

city_usage = df.sort_values("Fuel Consumption City (L/100km)")
print("Biggest spenders:")
print(city_usage[["Make", "Model", "Fuel Consumption City (L/100km)"]].tail(3))
print("Smallest spenders:")
print(city_usage[["Make", "Model", "Fuel Consumption City (L/100km)"]].head(3))


print("\nc)")

medium_motors = df[(df["Engine Size (L)"] >= 2.5) & df["Engine Size (L)"] <= 3.5]
print(f"{len(medium_motors)} Cars with medium sized motors")
print(f"Their average CO2 consumption: {medium_motors["CO2 Emissions (g/km)"].mean()} g/km")

print("\nd)")

audi = df[df["Make"] == "Audi"]
print(f"Number of Audis: {len(audi)}")
print(f"Average CO2 emissions of 4-cylinder audis: {audi[audi["Cylinders"] == 4]["CO2 Emissions (g/km)"].mean()}")


print("\ne)")

cylinders = df[(df["Cylinders"] > 2)]
print(f"Number cars with more than 2 cylinders: {len(cylinders)}")
print(f'Average emission: ',df.groupby('Cylinders')['CO2 Emissions (g/km)'].mean().to_string())

print("\nf)")

diesel = df[df["Fuel Type"] == "D"]
gasoline = df[(df["Fuel Type"] == "X")]
print("Average city consumption for:")
print(f"diesel cars: {diesel["Fuel Consumption City (L/100km)"].mean()} L/100km")
print(f"gasoline cars: {gasoline["Fuel Consumption City (L/100km)"].mean()} L/100km")
print("Median city consumption for:")
print(f"diesel cars: {diesel["Fuel Consumption City (L/100km)"].median()} L/100km")
print(f"gasoline cars: {gasoline["Fuel Consumption City (L/100km)"].median()} L/100km")