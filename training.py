import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint



data = pd.read_csv('data.csv', header=None)
epochs = 1000
batch_size = 32

# 进行归一化
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# 创建训练数据和目标数据
sequence_length = 10
X = []
y = []
# for i in range(len(data_scaled) - sequence_length):
#     X.append(data_scaled[i:i+sequence_length])
#     y.append(data_scaled[i+sequence_length])

# 反向遍历数据，以使用后面的数据预测前面的数据
for i in range(len(data_scaled) - sequence_length - 1, -1, -1):
    X.append(data_scaled[i:i + sequence_length])
    y.append(data_scaled[i + sequence_length])
X = np.array(X)
y = np.array(y)

# 将数据分为训练集、验证集和测试集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

# 创建LSTM模型
model = Sequential()
model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(sequence_length, data.shape[1])))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dense(data.shape[1]))
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')

# 添加早停策略、学习率调度器和模型保存回调
early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, verbose=1)
model_checkpoint = ModelCheckpoint('best_model_epoch_{epoch:02d}.h5', monitor='val_loss', save_best_only=True)
model_checkpoint_last = ModelCheckpoint('last_model_{epoch:02d}.h5', save_best_only=True)  # 不仅保存最好的模型

# 训练模型
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2,
                    validation_data=(X_val, y_val), callbacks=[early_stopping, reduce_lr, model_checkpoint, model_checkpoint_last])

