import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import dataframe_image as dfi
# 读取数据
data = pd.read_csv("clean_XAU_15m_data_2004_to_2024-20-09.csv")


# 对 High_Low_Diff 和 Close_Open_Diff 进行离散化处理，
bin1 = [0, 1, 2, 3, 4, 5, 10, 20, 120]
bin2 = [-35, -5, -3, 0, 3, 5, 10, 20, 55]
data['High_Low_Diff_Discretized'] = pd.cut(data['High_Low_Diff'], bins=bin1, 
    labels=[1, 2, 3, 4, 5, 6, 7, 8])
data['Close_Open_Diff_Discretized'] = pd.cut(data['Close_Open_Diff'], bins=bin2, 
    labels=[1, 2, 3, 4, 5, 6, 7, 8])

# 特征和目标变量
X = data[['Volume_Binned', 'High_Low_Diff_Discretized', 'Close_Open_Diff_Discretized']]
y = data['Close_Change_Category']

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义评估函数
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    return accuracy, precision, recall

# 初始化结果列表
results = []

# 1. 决策树
clf = DecisionTreeClassifier(random_state=42)
accuracy, precision, recall = evaluate_model(clf, X_train, X_test, y_train, y_test)
results.append(['Decision Tree', accuracy, precision, recall])

# 2. KNN（需要标准化）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
accuracy, precision, recall = evaluate_model(knn, X_train_scaled, X_test_scaled, 
    y_train_scaled, y_test_scaled)
results.append(['KNN', accuracy, precision, recall])

# 3. 朴素贝叶斯
nb = GaussianNB()
accuracy, precision, recall = evaluate_model(nb, X_train_scaled, X_test_scaled, 
    y_train_scaled, y_test_scaled)
results.append(['Naive Bayes', accuracy, precision, recall])

# 4. 随机森林
rf = RandomForestClassifier(n_estimators=25, random_state=42)
accuracy, precision, recall = evaluate_model(rf, X_train_scaled, X_test_scaled, 
    y_train_scaled, y_test_scaled)
results.append(['Random Forest', accuracy, precision, recall])

# 将结果转换为 DataFrame 并输出
results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall'])
dfi.export(results_df, "黄金价格数据分类评估结果.png")  # 导出为图片
print(results_df)