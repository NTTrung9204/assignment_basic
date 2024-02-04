#import libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

df=pd.read_csv('housing.csv')
# df.hist(bins=50, figsize=(20,15))


df["income_cat"] = pd.cut(df["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])

df["income_cat"].hist()

plt.show()