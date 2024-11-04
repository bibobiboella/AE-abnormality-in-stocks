import os
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler
from keras import layers
from keras.models import Sequential, load_model

# 设置Keras后端
os.environ["KERAS_BACKEND"] = "tensorflow"

# 加载数据
st.title("S&P 500 Stock Data Analysis")
sp500 = pickle.load(open("./sp500.pickle", "rb"))

# 显示数据示例
st.subheader("S&P 500 Data Example")
selected_stock = st.selectbox("Select a stock to view data", list(sp500.keys()))
st.write(sp500[selected_stock])

# 数据处理函数
WIN_SIZE = 7
def create_windows(data, win_size):
    X = []
    for i in range(len(data) - win_size):
        X.append(data[i : i + win_size])
    return np.array(X), np.array(X)[:, :, 0]

# 数据预处理
training_windows = []
training_labels = []
testing_windows = {}
testing_labels = {}
scalars = {}

for symbol, stock in sp500.items():
    stock.set_index(stock.columns[0], inplace=True)
    stock.index = pd.to_datetime(stock.index)
    stock["log_return"] = np.log(stock["adjusted_close"] / stock["adjusted_close"].shift(1)).fillna(0)
    stock = stock[["log_return", "volume"]].astype(np.float32)

    train = stock[stock.index < "2021-01-01"]
    test = stock[stock.index >= "2021-01-01"]

    scaler = StandardScaler()
    if not train.empty:
        train_scaled = scaler.fit_transform(train)
        test_scaled = scaler.transform(test)
        train_windows, train_labels = create_windows(train_scaled, WIN_SIZE)
        test_windows, test_labels = create_windows(test_scaled, WIN_SIZE)
        training_windows.append(train_windows)
        training_labels.append(train_labels)
        testing_windows[symbol] = test_windows
        testing_labels[symbol] = test_labels
    else:
        if len(test) >= WIN_SIZE:
            test_scaled = scaler.fit_transform(test)
            test_windows, test_labels = create_windows(test_scaled, WIN_SIZE)
            testing_windows[symbol] = test_windows
            testing_labels[symbol] = test_labels
            scaler.fit(test)

    scalars[symbol] = scaler

training_windows = np.concatenate(training_windows)
training_labels = np.concatenate(training_labels)

# 加载模型和历史记录
st.subheader("Model Training History")
with open("history.pickle", "rb") as f:
    history = pickle.load(f)
ae = load_model("autoencoder.keras")

# 显示训练和验证损失
fig, ax = plt.subplots()
ax.plot(history["loss"], label="Training Loss")
ax.plot(history["val_loss"], label="Validation Loss")
ax.legend()
st.pyplot(fig)

# 计算一次训练集的预测和MAE损失
train_predictions = ae.predict(training_windows)
train_mae_loss = np.mean(np.abs(train_predictions - training_labels), axis=1)
THRESHOLD = np.percentile(train_mae_loss, 90)

# 绘制训练MAE损失分布
st.subheader("Training MAE Loss Distribution")
fig = px.histogram(train_mae_loss[train_mae_loss < THRESHOLD], nbins=100, title="Training MAE Loss Distribution")
st.plotly_chart(fig)

# 选择股票并检测异常
st.subheader("Anomaly Detection")
selected_stock_for_anomaly = st.selectbox("Select a stock to detect anomalies", list(testing_windows.keys()))
test_predictions = ae.predict(testing_windows[selected_stock_for_anomaly])
test_mae_loss = np.mean(np.abs(test_predictions - testing_labels[selected_stock_for_anomaly]), axis=1)
plot_data = sp500[selected_stock_for_anomaly].loc[sp500[selected_stock_for_anomaly].index >= "2021-01-01"]
anomalies = plot_data[:-WIN_SIZE][test_mae_loss > THRESHOLD]

# 绘制价格和异常
fig_price = px.line(plot_data, x=plot_data.index, y="adjusted_close", title=f"Stock Price and Anomalies for {selected_stock_for_anomaly}")
fig_price.add_scatter(x=anomalies.index, y=anomalies["adjusted_close"], mode="markers", marker=dict(color="red", size=3), name="Anomaly")
st.plotly_chart(fig_price)

# 绘制返回率和异常
fig_return = px.line(plot_data, x=plot_data.index, y="log_return", title=f"Stock Returns and Anomalies for {selected_stock_for_anomaly}")
fig_return.add_scatter(x=anomalies.index, y=anomalies["log_return"], mode="markers", marker=dict(color="red", size=3), name="Anomaly")
st.plotly_chart(fig_return)

# 显示异常列表
st.subheader("List of Anomalies")
anomalies_data = anomalies.loc[:, ["adjusted_close", "log_return"]]
anomalies_data["return%"] = (np.exp(anomalies_data["log_return"]) - 1) * 100
anomalies_data["period_mae"] = test_mae_loss[test_mae_loss > THRESHOLD]
anomalies_data["threshold"] = THRESHOLD
st.write(anomalies_data)
