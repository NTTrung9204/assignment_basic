import numpy as np 
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("data.csv", error_bad_lines=False)
data = data.dropna()

print(data.isnull().sum())

def word(password):
    character=[]
    for i in password:
        character.append(i)
    return character
  
x = np.array(data["password"])
y = np.array(data["strength"])

tdif = TfidfVectorizer(tokenizer=word)

# print(tdif.fit_transform(['2', '3']))

x = tdif.fit_transform(x)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.05, random_state=42)

model = RandomForestClassifier()
model.fit(xtrain, ytrain)

ypred = model.predict(xtest)

print(accuracy_score(ytest, ypred))