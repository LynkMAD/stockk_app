import os
import torch
import torch.nn as nn
from core.Autoformer import Model as Autoformer
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from argparse import Namespace

def dict2namespace(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict2namespace(v)
    return Namespace(**d)
class Model():
    """Autoformer model builder/trainer/predictor (tương tự LSTM class gốc)"""

    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = None

    def build_model(self, configs):
        # Lấy params từ configs
        # Tạo Autoformer
        configs_ns = dict2namespace(configs)
        self.model = Autoformer(configs_ns).to(self.device)
        print('[Model] Autoformer built!')

    def load_model(self, filepath):
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.eval()
        print(f'[Model] Loaded model from {filepath}')

    def save_model(self, save_dir, name='autoformer_model.pth'):
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, name)
        torch.save(self.model.state_dict(), save_path)
        self.model_path = save_path
        print(f'[Model] Saved at {save_path}')

    def train(self, x_train, y_train, epochs, batch_size, save_dir):
        # x_train, y_train: numpy arrays, shape (N, seq_len, features)
        self.model.train()
        dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            losses = []
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                output = self.model(xb, xb, None, None)
                loss = criterion(output, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            print(f"Epoch {epoch+1}/{epochs}, Loss: {np.mean(losses):.6f}")

        self.save_model(save_dir)
        print('[Model] Training completed')

    def predict(self, x):
        # x: numpy array (N, seq_len, features)
        self.model.eval()
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            out = self.model(x, x, None, None)
        return out.cpu().numpy()

    # Optional: bạn có thể thêm các hàm predict_point_by_point, predict_sequences_multiple như class gốc
    def predict_point_by_point(self, data):
        """
        Dự báo từng điểm một (dùng input thực tế, mỗi lần dự báo 1 bước, giống fit_transform từng sample)
        """
        print('[Model] Predicting Point-by-Point...')
        self.model.eval()
        data_tensor = torch.tensor(data, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predicted = self.model(data_tensor, data_tensor, None, None)
        # predicted shape: (N, pred_len, n_features) -> chỉ lấy bước đầu
        predicted = predicted[:, 0, 0].cpu().numpy()  # Lấy cột 0, bước 0, tuỳ task có thể lấy cả vector
        return predicted

    def predict_sequences_multiple(self, data, window_size, prediction_len):
        """
        Dự báo nhiều chuỗi, mỗi chuỗi predict prediction_len bước, xong dịch window tiếp (classic multi-seq).
        """
        print('[Model] Predicting Sequences Multiple...')
        self.model.eval()
        prediction_seqs = []
        num_seq = int(len(data) / prediction_len)
        for i in range(num_seq):
            curr_frame = data[i * prediction_len]  # shape (window_size, n_features)
            curr_input = np.expand_dims(curr_frame, axis=0)  # (1, seq_len, n_features)
            with torch.no_grad():
                curr_pred = self.model(
                    torch.tensor(curr_input, dtype=torch.float32).to(self.device),
                    torch.tensor(curr_input, dtype=torch.float32).to(self.device),
                    None, None
                ).cpu().numpy()
            # Lấy hết prediction_len bước dự báo
            prediction_seqs.append(curr_pred[0, :, 0].tolist())
        return prediction_seqs

    def predict_sequences_multiple_modified(self, data, window_size, prediction_len):
        """
        Tương tự ở trên, nhưng lặp qua từng frame (0, prediction_len, ...) thay vì len/len.
        """
        print('[Model] Predicting Sequences Multiple (Modified)...')
        self.model.eval()
        prediction_seqs = []
        for i in range(0, len(data), prediction_len):
            curr_frame = data[i]
            curr_input = np.expand_dims(curr_frame, axis=0)
            with torch.no_grad():
                curr_pred = self.model(
                    torch.tensor(curr_input, dtype=torch.float32).to(self.device),
                    torch.tensor(curr_input, dtype=torch.float32).to(self.device),
                    None, None
                ).cpu().numpy()
            prediction_seqs.append(curr_pred[0, :, 0].tolist())
        return prediction_seqs

    def predict_sequence_full(self, data, window_size):
        """
        Dự báo full chuỗi, window trượt từng bước, mỗi lần predict 1 bước (kiểu walk forward).
        """
        print('[Model] Predicting Sequence Full...')
        self.model.eval()
        curr_frame = data[0]
        predicted = []
        for i in range(len(data)):
            curr_input = np.expand_dims(curr_frame, axis=0)  # (1, seq_len, n_features)
            with torch.no_grad():
                curr_pred = self.model(
                    torch.tensor(curr_input, dtype=torch.float32).to(self.device),
                    torch.tensor(curr_input, dtype=torch.float32).to(self.device),
                    None, None
                ).cpu().numpy()
            # Lấy bước dự báo đầu tiên
            predicted.append(curr_pred[0, 0, 0])
            # Cập nhật curr_frame với dự báo mới (window trượt 1 bước)
            curr_frame = np.concatenate([curr_frame[1:], [[curr_pred[0, 0, 0]] * curr_frame.shape[1]]], axis=0)
        return predicted

