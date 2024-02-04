import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("train.csv")
df = df.drop(['Type_of_Loan'], axis=1)
df = df.drop(['ID'], axis=1)
df = df.drop(['Customer_ID'], axis=1)
df = df.drop(['Age'], axis=1)
df = df.drop(['Name'], axis=1)

df["Occupation"], _ = pd.factorize(df["Occupation"])
df["Credit_Mix"], _ = pd.factorize(df["Credit_Mix"])
df["Payment_of_Min_Amount"], _ = pd.factorize(df["Payment_of_Min_Amount"])
df["Payment_Behaviour"], _ = pd.factorize(df["Payment_Behaviour"])
df["Credit_Score"], _ = pd.factorize(df["Credit_Score"])

X = df.iloc[:, :-1]
Y = df.iloc[:, -1]


model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)

feature_importances = pd.Series(model.feature_importances_, index = X.columns)
feature_importances.nlargest(10).plot(kind='barh')
plt.show()


