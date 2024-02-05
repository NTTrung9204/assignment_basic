import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def deg_to_rad(deg):
    return deg/180 * np.pi

def distance(lat1, lat2, long1, long2, R = 6371):
    a = np.sin(deg_to_rad(lat2-lat1)/2)**2 + np.cos(deg_to_rad(lat1)) * np.cos(deg_to_rad(lat2)) * np.sin(deg_to_rad(long2-long1)/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

df = pd.read_csv('deliverytime.txt')
df['Type_of_vehicle'], _ = pd.factorize(df['Type_of_vehicle'])
df['distance'] = distance(df['Restaurant_latitude'], df['Delivery_location_latitude'], df['Restaurant_longitude'], df['Delivery_location_longitude'])

upper_limit = df["distance"].mean() + 1 * df["distance"].std()

df = df[df["distance"] < upper_limit]

X = df.loc[:, ['Delivery_person_Age', 'Delivery_person_Ratings', 'Type_of_vehicle', 'distance']].values
Y = df.loc[:, ['Time_taken(min)']].values.ravel()

X_train_temp, X_test, Y_train_temp, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

X_train, X_val, Y_train, Y_val = train_test_split(X_train_temp, Y_train_temp, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train_temp, Y_train_temp)

Y_pred = model.predict(X_test)

print(f"Error: {mean_squared_error(Y_pred, Y_test)}")

print(model.predict([[20, 4.9, 2, 20.4]]))