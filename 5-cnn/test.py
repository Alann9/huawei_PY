
import numpy as np
from sklearn.linear_model import LinearRegression

# 输入数据
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 5, 9, 8, 10])

# 创建线性回归模型
model = LinearRegression()

# 拟合模型
model.fit(X, y)

# 预测
X_test = np.array([[6], [7]])
y_pred = model.predict(X_test)

# 输出预测结果
print(y_pred)
