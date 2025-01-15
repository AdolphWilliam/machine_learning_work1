import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
import dataframe_image as dfi
# 读取数据
df = pd.read_csv('clean_diversified_ecommerce_dataset.csv')

# 划分特征变量和目标变量     
X = df[[
    'Category',
    'Price',
    "Discount",
    "Tax Rate",
    "Stock Level",
    "Customer Age Group",
    "Customer Gender",
    "Shipping Cost",
    "Shipping Method",
    "Seasonality",
    "Popularity Index"
]]
y = df["Return Rate"]

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 初始化结果列表
results = []

# 定义评估函数
def evaluate_model(model_name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    results.append({
        "Model": model_name,
        "Accuracy": f"{accuracy:.2f}",
        "Precision": f"{precision:.2f}",
        "Recall": f"{recall:.2f}"
    })

# 1. 决策树模型
dt = DecisionTreeClassifier(random_state=42)
evaluate_model("Decision Tree", dt, X_train, X_test, y_train, y_test)

# 2. K-最近邻模型
knn = KNeighborsClassifier(n_neighbors=3)
evaluate_model("K-Nearest Neighbors", knn, X_train, X_test, y_train, y_test)

# 3. 朴素贝叶斯模型
gnb = GaussianNB()
evaluate_model("Naive Bayes", gnb, X_train, X_test, y_train, y_test)

# 4. 随机森林模型
rf = RandomForestClassifier(n_estimators=25, random_state=42)
evaluate_model("Random Forest", rf, X_train, X_test, y_train, y_test)

# 输出结果表格
results_df = pd.DataFrame(results)
dfi.export(results_df, "电子商务数据分类评估结果.png")  # 导出为图片
print(results_df)