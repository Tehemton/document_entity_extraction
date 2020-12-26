import pandas as pd 

df = pd.read_json('.\\TRAIN.json', orient='records')
df.columns = ['train_str','annotation']
print(df.head)
op=[]

for i, rows in df.iterrows():
    op.append((rows.train_str, {'entities':rows.annotation}))
print(op)