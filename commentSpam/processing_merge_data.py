import pandas as pd 

list_file = ['Youtube01-Psy.csv', 'Youtube02-KatyPerry.csv', 'Youtube03-LMFAO.csv', 'Youtube04-Eminem.csv', 'Youtube05-Shakira.csv']

df = [pd.read_csv(file) for file in list_file]

merge_df = pd.concat(df)

merge_df.to_csv('data.csv', index=False)