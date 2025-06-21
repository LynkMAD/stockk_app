import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import torch
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler

from core.Autoformer import Model as Autoformer
from core.autoformer_dataloader import AutoformerDataLoader
from core.LSTM_modified_model import Model as LSTMModel
from tst import Transformer
import json
from argparse import Namespace

DATA_DIR = "data"
MODEL_DIR = "models"

def get_available_stocks():
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    # Trả về dict: {'AAPL': 'aapl', 'GOOG': 'goog', ...}
    stocks = {f.split(".")[0].upper(): f.split(".")[0] for f in files}
    return stocks

def dict2namespace(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict2namespace(v)
    return Namespace(**d)

def load_stock_data(stock_name):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{stock_name}.csv"))
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    return df

def plot_predictions(df, forecast_prices, selected_range):
    last_date = df['Date'].iloc[-1]
    forecast_dates = [last_date + timedelta(days=i+1) for i in range(len(forecast_prices))]
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Close': forecast_prices})

    if selected_range != "all":
        days_back = int(selected_range)
        df = df.tail(days_back + 1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Giá thực tế', line=dict(color='cyan')))
    forecast_dates = [df['Date'].iloc[-1]] + list(forecast_df['Date'])
    forecast_prices = [df['Close'].iloc[-1]] + list(forecast_df['Close'])
    fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_prices, mode='lines+markers', name='Dự báo', line=dict(color='orange', dash='dash')))
    fig.update_layout(title='Dự báo giá cổ phiếu', xaxis_title='Ngày', yaxis_title='Giá', template='plotly_dark')
    return fig

def predict_lstm(df, model_path, predict_days):
    features = ['Close', 'Volume', 'Scaled_sentiment']
    data = df[features].values
    cols_to_norm = [0, 1]
    seq_len = 50
    val_ratio = 0.1
    train_end_idx = int(len(data) * (1 - val_ratio))
    df_train = df.iloc[:train_end_idx].copy()
    last_seq = data[train_end_idx - seq_len:train_end_idx].copy()
    base_vals = last_seq[0].copy()
    for col in cols_to_norm:
        if base_vals[col] == 0: base_vals[col] = 1
        last_seq[:, col] = (last_seq[:, col] / base_vals[col]) - 1
    input_seq = np.array([last_seq])
    model = LSTMModel()
    model.load_model(model_path, configs=None)
    prediction_seqs = model.predict_sequences_multiple_modified(input_seq, seq_len, predict_days)
    last_preds = prediction_seqs[-1]
    future = [(p + 1) * base_vals[0] for p in last_preds]
    return future, df_train

def predict_transformer_autoregressive(df, model_path, predict_days):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if 'Scaled_sentiment' in df.columns:
        data = df[['Volume', 'Open', 'Close', 'Scaled_sentiment']].values
        d_input = 4
        mode = 'Sentiment'
    else:
        data = df[['Volume', 'Open', 'Close']].values
        d_input = 3
        mode = 'Nonsentiment'

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    input_length = 50
    sequence = scaled_data[-input_length:].copy()

    d_output = 1
    d_model = 32
    q = 8
    v = 8
    h = 8
    N = 8
    attention_size = 512
    dropout = 0.1
    pe = 'regular'
    chunk_mode = None

    model = Transformer(d_input, d_model, d_output, q, v, h, N,
                        attention_size=attention_size, dropout=dropout,
                        chunk_mode=chunk_mode, pe=pe).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    predicted_scaled_list = []

    for _ in range(predict_days):
        input_seq = torch.tensor(sequence[-input_length:], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_seq)
            next_step = output.cpu().numpy().flatten()[0]
            predicted_scaled_list.append(next_step)

        next_row = sequence[-1].copy()
        next_row[2] = next_step
        sequence = np.vstack([sequence, next_row])

    if mode == 'Sentiment':
        expanded = np.zeros((predict_days, 4))
    else:
        expanded = np.zeros((predict_days, 3))

    expanded[:, 2] = predicted_scaled_list
    predicted = scaler.inverse_transform(expanded)[:, 2]

    return predicted.tolist(), df

def predict_autoformer(df, model_path, config_path, predict_days):
    temp_csv_path = "temp_input_autoformer.csv"
    df.to_csv(temp_csv_path, index=False)
    with open(config_path, "r") as f:
        config = json.load(f)
    configs = dict2namespace(config)
    data = AutoformerDataLoader(temp_csv_path, configs)
    test_data = data.get_test()
    model = Autoformer(configs).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(test_data[0].shape[0]):
            x_enc = torch.tensor(test_data[0][i:i+1], dtype=torch.float32)
            x_mark_enc = torch.tensor(test_data[1][i:i+1], dtype=torch.float32)
            x_dec = torch.tensor(test_data[2][i:i+1], dtype=torch.float32)
            x_mark_dec = torch.tensor(test_data[3][i:i+1], dtype=torch.float32)
            out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            preds.append(out.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    y_pred = preds[-1, :predict_days, 0]
    scaler = data.scaler
    min_val, scale_val = scaler.min_[0], scaler.scale_[0]
    y_pred_rescaled = (y_pred - min_val) / scale_val
    return y_pred_rescaled.tolist(), df

# ---------- Streamlit UI ----------
st.set_page_config(layout="wide")
st.title("📈 Dự Báo Giá Cổ Phiếu")

stocks_dict = get_available_stocks()
selected_display = st.selectbox("Chọn mã cổ phiếu:", list(stocks_dict.keys()))
stock = stocks_dict[selected_display]  # Tên đúng (không viết hoa) để đọc file
model_type = st.selectbox("Chọn mô hình:", ["LSTM", "Transformer", "Autoformer"])
train_type = st.radio("Loại mô hình:", ["single", "multi"])
days = st.number_input("Số ngày dự báo:", value=3, min_value=1, step=1)
display_range = st.selectbox("Khoảng thời gian hiển thị:", ["all", "365", "90", "30"])

if st.button("🚀 Dự báo"):
    df = load_stock_data(stock)
    if model_type == "LSTM":
        model_path = f"stocks_model/LSTM_sentiment_50.keras" if train_type == "multi" else os.path.join(MODEL_DIR, f"LSTM/LSTM_sentiment_{stock}.h5")
        forecast_vals, df_train = predict_lstm(df, model_path, days)
    elif model_type == "Transformer":
        model_path = f"stocks_model/Sentiment_50_8layers.pt" if train_type == "multi" else os.path.join(MODEL_DIR, f"Transformer/Sentiment_{stock}_8layers.pt")
        forecast_vals, df_train = predict_transformer_autoregressive(df, model_path, days)
    elif model_type == "Autoformer":
        if train_type == "multi":
            st.warning("🚫 Chưa hỗ trợ Autoformer nhiều cổ phiếu.")
            st.stop()
        model_path = os.path.join(MODEL_DIR, f"Auto/{stock}_autoformer.pth")
        config_path = "sentiment_config.json"
        forecast_vals, df_train = predict_autoformer(df, model_path, config_path, days)

    fig = plot_predictions(df_train, forecast_vals, display_range)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"📅 Kết quả dự báo {days} ngày tiếp theo:")
    last_date = df['Date'].iloc[-1]
    forecast_days = [(last_date + timedelta(days=i+1)).strftime("%Y-%m-%d") for i in range(len(forecast_vals))]
    results_df = pd.DataFrame({"Ngày": forecast_days, "Giá dự báo": forecast_vals})
    st.dataframe(results_df, use_container_width=True)
