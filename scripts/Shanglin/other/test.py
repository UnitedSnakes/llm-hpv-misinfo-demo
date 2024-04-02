import pandas as pd

data = {'a': [1, 2, 3], 'b': [4, 5, 6]}
df = pd.DataFrame(data)

value = df.loc[1, 'a']
print(df)
print(value)
print(df, 111)