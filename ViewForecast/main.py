import pandas as pd 
import matplotlib.pyplot as plt
import statsmodels.api as sm

df = pd.read_csv('View_test.csv')
df = df.dropna()
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
sm.graphics.tsa.plot_pacf(df["Views"], lags=100)
pd.plotting.autocorrelation_plot(df["Views"])

print(len(df))

p, d, q = 5, 1, 2
model = sm.tsa.statespace.SARIMAX(df['Views'],
                                  order=(p, d, q),
                                  seasonal_order=(p, d, q, 12))
model = model.fit()

prediction = model.predict(len(df), len(df)+40)
print(prediction)

df["Views"].plot(legend=True, label="Training Data", 
                   figsize=(15, 10))
prediction.plot(legend=True, label="Predictions")

plt.show()