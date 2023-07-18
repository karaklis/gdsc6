import pandas as pd
from scipy.stats.mstats import gmean
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 100)

df0 = pd.read_csv('../predictions/prediction_test_20230610.csv')
# df1 = pd.read_csv('../predictions/prediction_40_1.csv')
# df2 = pd.read_csv('../predictions/prediction_60_.6.csv')
# df3 = pd.read_csv('../predictions/prediction_4x2.csv')
print(df0)
print(pd.get_dummies(df0['predicted_class_id']).astype(float))

dfd0 = df0.merge(pd.get_dummies(df0['predicted_class_id']).astype(float), left_index=True, right_index=True)
dfd0.columns = dfd0.columns.astype(str)
dfd0[[str(x) for x in range(66)]] /= 67
dfd0[[str(x) for x in range(66)]] += 1/67
dfd1 = pd.read_csv('../predictions/pred_details_40_1.csv')
dfd2 = pd.read_csv('../predictions/pred_details_60_.6.csv')
dfd3 = pd.read_csv('../predictions/pred_details_4x2.csv')
dfd4 = pd.read_csv('../predictions/pred_details_rgb_4o.csv')
f0, f1, f2, f3, f4 = .8196, .7921, .7133, .8284, .7023  # results

dfd1.drop(columns=['idx'], inplace=True)
dfd2.drop(columns=['idx'], inplace=True)
dfd3.drop(columns=['idx'], inplace=True)
dfd4.drop(columns=['idx'], inplace=True)
dfd0['src'], dfd1['src'], dfd2['src'], dfd3['src'], dfd4['src'] = 0, 1, 2, 3, 4
dfd0['weights'], dfd1['weights'], dfd2['weights'], dfd3['weights'], dfd4['weights'] = f0, f1, f2, f3, f4
print(dfd0)
# dfd1.loc[:, dfd1.columns!='file_name'] *= (f1 / (f1+f2+f3))
# dfd2.loc[:, dfd2.columns!='file_name'] *= (f2 / (f1+f2+f3))
# dfd3.loc[:, dfd3.columns!='file_name'] *= (f3 / (f1+f2+f3))
dfd = pd.concat([dfd0, dfd1, dfd2, dfd3, dfd4])
print(dfd)

# dfdg = weighted geometrical mean


dfd = dfd.groupby(['file_name'], as_index=False).agg('prod')
print(dfd)
# dfdt = dfd.loc[:, dfd1.columns!='file_name'].T
print(dfd.columns)
# dfd['sum'] = dfd.loc[:, [str(x) for x in range(66)]].sum(axis=1)
dfd['sum'] = dfd[[str(x) for x in range(66)]].sum(axis=1)
print(dfd)
for i in range(66):
    dfd.loc[:, str(i)] /= dfd['sum']

print(dfd)
dfd['predicted_class_id'] = dfd[[str(x) for x in range(66)]].idxmax(axis='columns')
print(dfd)
dfd[['file_name', 'predicted_class_id']].to_csv('../predictions/prediction_detail.csv', index=False)

