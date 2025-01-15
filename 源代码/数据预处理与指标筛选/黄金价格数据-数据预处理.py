import pandas as pd
from sklearn.preprocessing import StandardScaler
df=pd.read_csv('XAU_15m_data_2004_to_2024-20-09.csv')
# 合并日期和时间字段
df['Timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')

# 删除原始 Date 和 Time 列
df.drop(columns=['Date', 'Time'], inplace=True)

# 确保数值列为浮点数
for col in ['Open', 'High', 'Low', 'Close']:
    df[col] = df[col].astype(float)

# 求 Volume 的最大最小值
volume_min = df['Volume'].min()
volume_max = df['Volume'].max()


# 定义区间和编码
bins = [0, 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 41505]
labels = list(range(1, len(bins)))
df['Volume_Binned'] = pd.cut(df['Volume'], bins=bins, labels=labels, right=False)

df.drop(columns=['Volume'], inplace=True)

df = df.dropna()

# # 查看处理后的数据
# print(df.head())



# 计算最高价与最低价之差
df['High_Low_Diff'] = df['High'] - df['Low']

# 计算收盘价与开盘价之差
df['Close_Open_Diff'] = df['Close'] - df['Open']

# 计算收盘价涨跌状况 (0：大幅下跌，1：小幅下跌，2：小幅上涨，3：大幅上涨)
def categorize_close_change(row):
    if row > 0.5:
        return '3'
    elif 0 < row <= 0.5:
        return '2'
    elif row <= -0.5:
        return '0'
    else:
        return '1'

# 计算涨跌幅
df['Close_Change'] = df['Close'].pct_change() * 100
df['Close_Change_Category'] = df['Close_Change'].apply(categorize_close_change)

df = df.drop('Close_Change', axis=1)
df = df.dropna()

df.to_csv("clean_XAU_15m_data_2004_to_2024-20-09.csv", index=False)

