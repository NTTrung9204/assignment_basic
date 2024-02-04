from sklearn.model_selection import train_test_split
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt 

df = pd.read_csv("train.csv")
df["Credit_Mix"], _ = pd.factorize(df["Credit_Mix"])
df["Credit_Score"], _ = pd.factorize(df["Credit_Score"])

X = df.loc[:, ['Credit_Mix', 'Outstanding_Debt', 'Interest_Rate', 'Credit_History_Age', 'Delay_from_due_date', 'Num_Credit_Inquiries', 'Num_Credit_Card', 'Month', 'Changed_Credit_Limit', 'Num_of_Delayed_Payment']].values
Y = df.loc[:, ['Credit_Score']].values.ravel()

X_train_temp, X_test, Y_train_temp, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_train_temp, Y_train_temp, test_size=0.2, random_state=42)

# model = LogisticRegression(max_iter=20000)

max_dep = [item + 1 for item in range(0, 30, 3)]
train_acc = []
valid_acc = []
test_acc  = []
for depth in max_dep:
    print(depth)
    # model = DecisionTreeClassifier(max_depth=depth)
    model = RandomForestClassifier(max_depth=depth)
    model.fit(X_train, Y_train)

    Y_train_pred = model.predict(X_train)
    train_acc.append(100*mean_squared_error(Y_train, Y_train_pred))

    Y_val_pred = model.predict(X_val)
    valid_acc.append(100*mean_squared_error(Y_val, Y_val_pred))

    Y_pred = model.predict(X_test)
    test_acc.append(100*mean_squared_error(Y_test, Y_pred))


plt.plot(train_acc, label='training')
plt.plot(valid_acc, label='validation')
plt.plot(test_acc, label='testing')
plt.show()
# Y_pred = model.predict(X_test)


# print ("Accuracy of 1NN: %.2f %%" %(100*accuracy_score(Y_test, Y_pred)))
