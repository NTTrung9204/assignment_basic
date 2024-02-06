import numpy as np 
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def vectorize_string(input_string):
    # Khởi tạo một vector đếm cho các chữ cái (bao gồm cả chữ hoa và chữ thường), số và ký tự đặc biệt
    vector_length = 128 + 10 + 33 + 1 # 128 chữ cái ASCII, 10 số, 33 ký tự đặc biệt, ký tự siêu đặc biệt
    vector = np.zeros(vector_length)

    # Đếm tần suất xuất hiện của từng ký tự trong chuỗi
    for char in input_string:
        if char.isalpha():
            if char.isupper():
                index = ord(char) - ord('A')
            else:
                index = ord(char) - ord('a') + 26  # Chữ hoa từ 0 đến 25, chữ thường từ 26 đến 51
        elif char.isdigit():
            try:
                index = 128 + int(char)
            except:
                index = -1
        else:
            special_chars = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
            index = 128 + 10 + special_chars.find(char)
            if index >= 128 + 10: index = -1
        if index >= 128 + 10 + 33: index = -1
        vector[index] += 1

    return vector

data = pd.read_csv("data.csv", error_bad_lines=False)
data = data.dropna()

# print(data['strength'].value_counts())

data['_password'] = [vectorize_string(item) for item in data['password']]
matrix = np.vstack(data['_password'].values)

x = matrix
y = np.array(data.loc[:, ['strength']].values.ravel())

print(x.shape)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(xtrain, ytrain)

ypred = model.predict(xtest)

print(accuracy_score(ytest, ypred))