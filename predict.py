import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

# 加载已保存的模型
loaded_model = load_model('best_model_epoch_44.h5') #lstm_model lstm_model-bestdlt99

# 加载之前拟合好的MinMaxScaler实例（替换为实际用于训练的数据）7 8 19 20 25 2 11
scaler = MinMaxScaler()
# 在这里加载训练数据，用于拟合MinMaxScaler实例
training_data = pd.read_csv('data.csv', header=None)  # 替换为实际训练数据文件名
scaler.fit(training_data)

# 预测步数，即要预测未来多少个时间步
prediction_steps = 5

# 设置时间序列长度（根据训练模型时的设置）
sequence_length = 10

# 获取最后一组用于预测的输入数据
last_input = training_data.head(sequence_length).values
print(last_input)
# 预测未来数据
future_predictions = []
for _ in range(prediction_steps):
    normalized_last_input = scaler.transform(last_input)  # 归一化输入数据
    X_last_input = np.array([normalized_last_input])  # 将输入数据转换为3D数组
    next_prediction = loaded_model.predict(X_last_input)
    future_predictions.append(next_prediction)

    # 更新输入数据，将新预测值加入
    last_input = np.concatenate((last_input[1:, :], next_prediction), axis=0)

# 反向归一化并转换为整数
future_predictions_original = scaler.inverse_transform(np.array(future_predictions).reshape(prediction_steps, -1))
future_predictions_int = np.round(future_predictions_original).astype(int)

# 输出预测结果
print("未来时间最可能出现的数据预测:")
print(future_predictions_int)
