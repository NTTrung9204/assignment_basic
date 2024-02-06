import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("data.csv")

df = df[['CONTENT', 'CLASS']]

df["CLASS"] = df["CLASS"].map({0: "Not Spam", 1: "Spam Comment"})

print(df["CLASS"].value_counts())

cv = CountVectorizer()

X = cv.fit_transform(np.array(df["CONTENT"]))
Y = np.array(df["CLASS"])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=43)

# model = BernoulliNB()
# model = RandomForestClassifier()
model = LogisticRegression()
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

print("Accuracy: %.2f %%" %(accuracy_score(Y_pred, Y_test)))

sample = "Need a loan? Get approved instantly with no credit check!" 
data = cv.transform([sample]).toarray()
print(model.predict(data))