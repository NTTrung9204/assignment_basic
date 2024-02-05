import pandas as pd 
import matplotlib.pyplot as plt
df = pd.read_csv('deliverytime.txt')

# print(df.head(10))
# print(df['Delivery_person_Ratings'])

rate = df['Delivery_person_Ratings'].values
age  = df['Delivery_person_Age'].values
order= df['Type_of_order'].values
vehi = df['Type_of_vehicle'].values
time = df['Time_taken(min)'].values 

print(df['Type_of_vehicle'].value_counts())
# plt.scatter(vehi, time, alpha=0.01)
# plt.show()