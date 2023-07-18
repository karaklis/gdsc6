import pandas as pd
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 100)

df0 = pd.read_csv('../predictions/prediction_test_20230610.csv').rename(columns={'predicted_class_id': 'pc0'})
# dfd3 = pd.read_csv('../predictions/pred_details_4x2.csv').rename(columns={'predicted_class_id': 'pc3'})
dfd3 = pd.read_csv('../predictions/pred_details_16_1.csv').rename(columns={'predicted_class_id': 'pc3'})
df = pd.merge(dfd3, df0, on='file_name')
df['max'] = df[[str(x) for x in range(66)]].max(axis=1)
df['predicted_class_id'] = -1
df.loc[df.pc0==df.pc3, ['predicted_class_id']] = df.pc0
med_v = df.loc[df.pc0!=df.pc3, ['max']].median().values.flatten()[0]
print(med_v)
df.loc[df['max']>med_v, ['predicted_class_id']] = df.pc3
df.loc[df.predicted_class_id==-1, ['predicted_class_id']] = df.pc0
df[['file_name', 'predicted_class_id']].to_csv('../predictions/prediction_magic.csv', index=False)

print(df)
