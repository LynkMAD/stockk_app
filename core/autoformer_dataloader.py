import pandas as pd
import numpy as np

def time_features(dates):
    if not isinstance(dates, pd.Series):
        dates = pd.Series(dates)
    dates = pd.to_datetime(dates)
    month_idx   = dates.dt.month    .astype(np.int64)  # 1–12
    day_idx     = dates.dt.day      .astype(np.int64)  # 1–31
    weekday_idx = dates.dt.dayofweek.astype(np.int64)  # 0–6
    hour_idx    = dates.dt.hour     .astype(np.int64)  # 0–23
    # Đáp ứng đúng thứ tự [month, day, weekday, hour]
    return np.stack([month_idx, day_idx, weekday_idx, hour_idx], axis=1)

from sklearn.preprocessing import MinMaxScaler

class AutoformerDataLoader:
    def __init__(self, csv_path, config):
        self.config = config
        # Lấy param cho chắc chắn, có thể ở model hoặc data (tùy file config của bạn)
        self.seq_len = getattr(config, "seq_len", getattr(config, "seq_len", 50))
        self.label_len = getattr(config, "label_len", getattr(config, "label_len", 25))
        self.pred_len = getattr(config, "pred_len", getattr(config, "pred_len", 3))
        self.columns = getattr(config, "columns", getattr(config, "columns", ["Close", "Volume", "Scaled_sentiment"]))
        self.normalise = getattr(config, "normalise", getattr(config, "normalise", True))
        self.train_test_split = getattr(config, "train_test_split", getattr(config, "train_test_split", 0.85))
        
        df = pd.read_csv(csv_path)
        self.dates = df['Date']
        self.data = df[self.columns].values
        # Chuẩn hóa
        self.scaler = MinMaxScaler()
        if self.normalise:
            self.data = self.scaler.fit_transform(self.data)
        # Time features
        self.time_feat = time_features(self.dates)
        # Tạo sliding window
        self.make_sliding_windows()
        # Chia train/test
        n_samples = self.X_enc.shape[0]
        split_idx = int(self.train_test_split * n_samples)
        self.train_idx = slice(0, split_idx)
        self.test_idx = slice(split_idx, n_samples)

    def make_sliding_windows(self):
        N = self.data.shape[0]
        n_feat = self.data.shape[1]
        X_enc, X_mark_enc, X_dec, X_mark_dec, Y = [], [], [], [], []
        for i in range(N - self.seq_len - self.pred_len):
            # encoder input
            x_enc = self.data[i:i+self.seq_len]
            x_mark_enc = self.time_feat[i:i+self.seq_len]

            # decoder input
            x_dec = self.data[i+self.seq_len-self.label_len:i+self.seq_len]
            zeros_pred = np.zeros((self.pred_len, n_feat))
            x_dec = np.concatenate([x_dec, zeros_pred], axis=0)
            x_mark_dec = self.time_feat[i+self.seq_len-self.label_len:i+self.seq_len+self.pred_len]

            # target
            y = self.data[i+self.seq_len : i+self.seq_len+self.pred_len, 0:1]

            X_enc.append(x_enc)
            X_mark_enc.append(x_mark_enc)
            X_dec.append(x_dec)
            X_mark_dec.append(x_mark_dec)
            Y.append(y)
        self.X_enc = np.array(X_enc)
        self.X_mark_enc = np.array(X_mark_enc)
        self.X_dec = np.array(X_dec)
        self.X_mark_dec = np.array(X_mark_dec)
        self.Y = np.array(Y)
        
    def get_train(self):
        return (self.X_enc[self.train_idx], self.X_mark_enc[self.train_idx],
                self.X_dec[self.train_idx], self.X_mark_dec[self.train_idx],
                self.Y[self.train_idx])

    def get_test(self):
        return (self.X_enc[self.test_idx], self.X_mark_enc[self.test_idx],
                self.X_dec[self.test_idx], self.X_mark_dec[self.test_idx],
                self.Y[self.test_idx])
