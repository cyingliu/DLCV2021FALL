import pandas as pd
gnd_path = 'p1_data/val_gt.csv'
result_path = 'output_p1.csv'
df_gnd = pd.read_csv(gnd_path)
df_gnd = df_gnd.sort_values(['image_id'], ascending=True)
df_result = pd.read_csv(result_path)
df_result = df_result.sort_values(['image_id'], ascending=True)
assert len(df_gnd)== len(df_result)
correct = 0
total = 0
for i in range(len(df_gnd)):
  id_gnd, label = df_gnd.iloc[i]['image_id'], df_gnd.iloc[i]['label']
  id_pred, pred = df_result.iloc[i]['image_id'], df_result.iloc[i]['label']
  assert id_gnd == id_pred
  if label == pred:
    correct += 1
  total += 1
print("Val acc: {}".format(correct / total))