import pandas as pd

df = pd.read_csv("fairness_dataset.csv")

df['pred_label'] = (df[['FICO Score', 'Savings Rate (%)', 'Credit History (months)']].sum(axis=1)) / 3 > 197.69
df['pred_label'] = df['pred_label'].astype(int)  # Convert boolean to int for 1 or 0

from scipy.stats import ttest_ind

# 假設 df 是包含資料的 DataFrame，且 'Region' 是區域，'FICO Score' 是特徵
region_a_fico = df[df['Region'] == 'A']['Credit History (months)']
region_b_fico = df[df['Region'] == 'B']['Credit History (months)']

# 執行 t 檢驗
t_stat, p_value = ttest_ind(region_a_fico, region_b_fico, equal_var=False)  # equal_var=False 表示方差不一定相等

print("t-statistic:", t_stat)
print("p-value:", p_value)



