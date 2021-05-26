import pandas as pd

from ipdb import set_trace


# build data
data = {
    'name'  : ['Adam', 'Ben', 'Cat', 'David'],
    'age'   : [19, 20, 19, 19],
    'gender': ['male', 'male', 'female', 'male'],
    'grades': [100, 90, 95, 80]
}

set_trace()
# build dataframe
df = pd.DataFrame(data, index=['A','B','C','D'])


# save csv file
df.to_csv('saved_file_header-T_index-T.csv', header=True,  index=True)
df.to_csv('saved_file_header-F_index-F.csv', header=False, index=False)


# 查看整個 dataframe
print(df)
# 查看前五列資料
print(df.head(5))
# 查看後三列資料
print(df.tail(3))


# 讀取某一欄資訊
print (df['age'])
# 讀取多欄位資訊 (需要多一個中括號)
print (df[['age', 'name']])

# 讀取某一列資訊
print (df[0:1])
# 讀取多列資訊
print (df[1:3])


# 讀取 某一列 某一欄 資訊
print (df.loc['B', 'age'])

# 讀取 許多列 多欄位 資訊 (需要多一個中括號)
print (df.loc[['B', 'C'], ['age', 'name']])

# 讀取 某一列 所有欄 資訊
print (df.loc['A', :])

# 讀取 某一行 所有列 資訊
print (df.loc[:, 'name'])


# 讀取 某一列 某一欄 資訊
print (df.iloc[0, 1])

# 讀取 特定多列 多欄 資訊 (需要多一個中括號)
print (df.iloc[[0, 3], [1, 3]])

# 讀取 連續多列 多欄 資訊
print (df.iloc[0:3, 1:3])

# 讀取 某一列 所有欄 資訊
print (df.iloc[0, :])

# 讀取 某一行 所有列 資訊
print (df.iloc[:, 2])


mask = df['gender']=='male'
print (df[mask])
