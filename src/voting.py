import pandas as pd
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 100)

df0 = pd.read_csv('../data/predictions/prediction_test_20230610.csv')
df1 = pd.read_csv('../data/predictions/prediction_40_1.csv')
df2 = pd.read_csv('../data/predictions/prediction_60_.6.csv')
df3 = pd.read_csv('../data/predictions/prediction_4x2.csv')
df4 = pd.read_csv('../data/predictions/prediction_rgb_4o.csv')
df5 = pd.read_csv('../data/predictions/prediction_1xx.csv')
df6 = pd.read_csv('../data/predictions/prediction_16_1.csv')

f0, f1, f2, f3, f4, f5, f6 = .8196, .7921, .7133, .8284, .7023, .7370, .8398  # results
df0['f'] = f0
df1['f'] = f1
df2['f'] = f2
df3['f'] = f3
df4['f'] = f4
df5['f'] = f5
df6['f'] = f6
# df = pd.concat([df0, df1, df2, df3, df4, df5, df6])
df = pd.concat([df0, df1, df3, df6])
print(df)
dfg = df.groupby(['file_name', 'predicted_class_id'], as_index=False).agg(v=('f', 'sum'))
print(dfg)
dfg['rank'] = dfg.groupby(['file_name'], as_index=False)['v'].rank(method='first', ascending=False)
dfg = dfg[dfg['rank'] == 1]
print(dfg)
dfg[['file_name', 'predicted_class_id']].to_csv('../data/predictions/prediction_voting.csv', index=False)
