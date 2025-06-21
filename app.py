import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from datetime import timedelta
from plotly.offline import plot
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np
from core.Autoformer import Model as Autoformer
from core.autoformer_dataloader import AutoformerDataLoader
from argparse import Namespace
import json
from argparse import Namespace

from tst import Transformer # Dùng model tự định nghĩa

from core.LSTM_modified_model import Model  # Dùng model tự định nghĩa

app = Flask(__name__)

DATA_DIR = "data"
MODEL_DIR = "models"

def load_stock_data(stock_name):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{stock_name}.csv"))
    print(os.path.join(DATA_DIR, f"{stock_name}.csv"))
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    return df

def predict_lstm(df, model_path, predict_days):
    features = ['Close', 'Volume', 'Scaled_sentiment']
    data = df[features].values
    cols_to_norm = [0, 1]
    seq_len = 50
    val_ratio = 0.1

    # Tính số dòng thực sự đã dùng để train (bỏ validation split)
    train_end_idx = int(len(data) * (1 - val_ratio))

    # Tính số dòng cuối cùng của chuỗi cuối cùng trong train
    if train_end_idx < seq_len:
        raise ValueError("Not enough data for sequence length.")

    # Cắt lại df_train sao cho điểm cuối trùng với điểm cuối tập train
    df_train = df.iloc[:train_end_idx].copy()
    last_seq = data[train_end_idx - seq_len:train_end_idx].copy()

    # Chuẩn hóa theo dòng đầu tiên của chuỗi
    base_vals = last_seq[0].copy()
    for col in cols_to_norm:
        if base_vals[col] == 0:
            base_vals[col] = 1
        last_seq[:, col] = (last_seq[:, col] / base_vals[col]) - 1

    # Reshape về (1, 50, 3)
    input_seq = np.array([last_seq])

    # Load model
    model = Model()
    model.load_model(model_path, configs=None)

    # Dự báo
    prediction_seqs = model.predict_sequences_multiple_modified(
        input_seq, window_size=seq_len, prediction_len=predict_days
    )
    last_preds = prediction_seqs[-1]

    # Giải mã
    future = [(p + 1) * base_vals[0] for p in last_preds]
    return future, df_train


def predict_transformer_autoregressive(df, model_path, predict_days):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Chọn dữ liệu và thiết lập input
    if 'Scaled_sentiment' in df.columns:
        data = df[['Volume', 'Open', 'Close', 'Scaled_sentiment']].values
        d_input = 4
        mode = 'Sentiment'
    else:
        data = df[['Volume', 'Open', 'Close']].values
        d_input = 3
        mode = 'Nonsentiment'

    # 2. Chuẩn hóa
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    input_length = 50
    output_length = 1  # mỗi lần dự đoán 1 bước
    sequence = scaled_data[-input_length:].copy()  # (50, d_input)

    # 3. Load mô hình
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

    # 4. Dự đoán autoregressive
    for _ in range(predict_days):
        input_seq = torch.tensor(sequence[-input_length:], dtype=torch.float32).unsqueeze(0).to(device)  # (1, 50, d_input)

        with torch.no_grad():
            output = model(input_seq)  # shape: (1, 1)
            next_step = output.cpu().numpy().flatten()[0]
            predicted_scaled_list.append(next_step)

        # 5. Tạo hàng tiếp theo (chỉ dự đoán cột 'Close')
        next_row = sequence[-1].copy()
        next_row[2] = next_step  # chỉ cập nhật cột 'Close', giữ nguyên các cột khác
        sequence = np.vstack([sequence, next_row])  # thêm dòng mới
        # Không cần pop đầu vì ta luôn lấy [-input_length:] rồi

    # 6. Giải scale
    if mode == 'Sentiment':
        expanded = np.zeros((predict_days, 4))
    else:
        expanded = np.zeros((predict_days, 3))

    expanded[:, 2] = predicted_scaled_list
    predicted = scaler.inverse_transform(expanded)[:, 2]

    df_train = df.copy()
    return predicted.tolist(), df_train

def dict2namespace(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict2namespace(v)
    return Namespace(**d)

def predict_autoformer(df, model_path, config_path, predict_days):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tạo file CSV tạm để tương thích AutoformerDataLoader
    temp_csv_path = "temp_input_autoformer.csv"
    df.to_csv(temp_csv_path, index=False)

    # Load cấu hình
    with open(config_path, "r") as f:
        config = json.load(f)
    configs = dict2namespace(config)

    # Tải và xử lý dữ liệu
    data = AutoformerDataLoader(temp_csv_path, configs)
    test_data = data.get_test()

    # Load model
    model = Autoformer(configs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Dự đoán toàn bộ tập test
    preds = []
    with torch.no_grad():
        for i in range(test_data[0].shape[0]):
            x_enc      = torch.tensor(test_data[0][i:i+1], dtype=torch.float32).to(device)
            x_mark_enc = torch.tensor(test_data[1][i:i+1], dtype=torch.float32).to(device)
            x_dec      = torch.tensor(test_data[2][i:i+1], dtype=torch.float32).to(device)
            x_mark_dec = torch.tensor(test_data[3][i:i+1], dtype=torch.float32).to(device)

            out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)  # (1, pred_len, 1)
            preds.append(out.cpu().numpy())

    preds = np.concatenate(preds, axis=0)  # (N, pred_len, 1)

    # Lấy ra n bước dự báo cuối cùng
    y_pred = preds[-1, :predict_days, 0]

    # Giải scale (dữ liệu đã được scale bằng MinMaxScaler tại AutoformerDataLoader)
    scaler = data.scaler
    min_val, scale_val = scaler.min_[0], scaler.scale_[0]
    y_pred_rescaled = (y_pred - min_val) / scale_val

    return y_pred_rescaled.tolist(), df.copy()


def plot_predictions(df, forecast_prices, selected_range="all"):
    last_date = df['Date'].iloc[-1]
    forecast_dates = [last_date + timedelta(days=i+1) for i in range(len(forecast_prices))]
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Close': forecast_prices})

    # Nếu không phải "all", chỉ lấy số ngày gần nhất
    if selected_range != "all":
        days_back = int(selected_range)
        df = df.tail(days_back + 1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Close'],
        mode='lines',
        name='Giá thực tế',
        line=dict(color='cyan')
    ))

    forecast_dates = [df['Date'].iloc[-1]] + list(forecast_df['Date'])
    forecast_prices = [df['Close'].iloc[-1]] + list(forecast_df['Close'])

    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_prices,
        mode='lines+markers',
        name='Dự báo',
        line=dict(color='orange', dash='dash')
    ))

    fig.update_layout(
        title='📈 Dự báo giá cổ phiếu',
        xaxis_title='Ngày',
        yaxis_title='Giá',
        hovermode='x unified',
        xaxis=dict(
            rangeslider=dict(visible=True),
            type="date"
        ),
        template='plotly_dark',
        font=dict(color='white'),
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e',
    )

    plot(fig, filename="static/plot.html", auto_open=False)

@app.route("/", methods=["GET", "POST"])
def index():
    forecast = None
    stock = model_type = display_range = ""
    days = 3  # Dự báo 3 ngày
    train_type = request.form.get("train_type", "single")
    # Lấy danh sách mã cổ phiếu từ thư mục data/
    def get_available_stocks():
        files = os.listdir(DATA_DIR)
        return sorted([f.split(".")[0].upper() for f in files if f.endswith(".csv")])

    available_stocks = get_available_stocks()

    if request.method == "POST":
        stock = request.form["stock"]
        model_type = request.form["model"]
        display_range = request.form.get("range", "all")

        df = load_stock_data(stock)

        if model_type == "LSTM":
            if train_type == "single":
                model_path = os.path.join(MODEL_DIR, f"LSTM/LSTM_sentiment_{stock}.h5")
            else:
                model_path = os.path.join(f"stocks_model/LSTM_sentiment_50.keras")
            print(model_path)
            forecast_vals, df_train = predict_lstm(df, model_path, days)

        elif model_type == "Transformer":
            if train_type == "single":
                model_path = os.path.join(MODEL_DIR, f"Transformer/Sentiment_{stock}_8layers.pt")
            else:
                model_path = os.path.join(f"stocks_model/Sentiment_50_8layers.pt")
            print(model_path)
            forecast_vals, df_train = predict_transformer_autoregressive(df, model_path, days)
        
        if model_type == "Autoformer":
            if train_type == "single":
                model_path = os.path.join(MODEL_DIR, f"Auto/{stock}_autoformer.pth")
                config_path = "sentiment_config.json"
                forecast_vals, df_train = predict_autoformer(df, model_path, config_path, days)
                plot_predictions(df_train, forecast_vals, display_range)
                print(model_path)
            else:
                forecast = [("Thông báo", "🚫 Hiện tại chưa có mô hình Autoformer nhiều cổ phiếu.")]
            
            
        plot_predictions(df_train, forecast_vals, display_range)

        last_date = df['Date'].iloc[-1]
        forecast_days = [(last_date + timedelta(days=i+1)).strftime("%Y-%m-%d") for i in range(len(forecast_vals))]
        forecast = list(zip(forecast_days, forecast_vals))

    return render_template(
    "index.html",
    stock=stock,
    model_type=model_type,
    train_type=train_type,
    days=days,
    forecast=forecast,
    selected_range=display_range,
    available_stocks=available_stocks
)

if __name__ == "__main__":
    app.run(debug=True)
