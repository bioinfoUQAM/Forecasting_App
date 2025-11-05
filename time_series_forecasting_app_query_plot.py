import sys
#sys.modules['torchvision'] = None
import gradio as gr
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from darts.models import NBEATSModel
from darts import TimeSeries
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import TensorDataset, DataLoader
import math
import io
import time
import logging
from PIL import Image
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Import the real models from the local files
from Bi_iGRU import CustomModel, Configs
# TVNet is not used in the final UI, but we keep the import for completeness
from TVNet import Model as TVNetModel, Configs as TVNetConfigs
from TimeMixerPP import Model as TimeMixerPPModel, Configs as TimeMixerPPConfigs

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def split_train_test_sliding(data, input_len, forecast_horizon, train_ratio=0.8):
    N = len(data)
    train_end = int(N * train_ratio)
    train_data = data[:train_end]
    X_train, Y_train = create_training_samples(train_data, input_len, forecast_horizon)
    test_data = data[train_end - input_len:]
    X_test, Y_test = create_training_samples(test_data, input_len, forecast_horizon)
    return X_train, Y_train, X_test, Y_test

def create_training_samples(data, seq_len, pred_len):
    X, Y = [], []
    if len(data) < seq_len + pred_len:
        return np.empty((0, seq_len, data.shape[1])), np.empty((0, pred_len, data.shape[1]))
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        Y.append(data[i+seq_len:i+seq_len+pred_len])
    return np.stack(X), np.stack(Y)


def run_bi_igru(df, forecast_horizon, input_len, lr, epochs, batch_size, normalize, train_ratio):
    """Generator for Bi-iGRU model, yields progress and final results."""
    try:
        data = df.values.astype(np.float32)
        if data.ndim == 1: data = data[:, None]
        X_train, Y_train, X_test, Y_test = split_train_test_sliding(data, input_len, forecast_horizon, train_ratio)
        if len(X_test) == 0: raise ValueError("Not enough data for test samples.")
        
        X_test_original = X_test.copy()
        N, L, D = X_train.shape

        scaler = StandardScaler() if normalize else None
        if scaler:
            scaler.fit(np.concatenate([X_train.reshape(-1, D), Y_train.reshape(-1, D)]))
            X_train = scaler.transform(X_train.reshape(-1, D)).reshape(X_train.shape)
            Y_train = scaler.transform(Y_train.reshape(-1, D)).reshape(Y_train.shape)
            X_test = scaler.transform(X_test.reshape(-1, D)).reshape(X_test.shape)
        
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
        train_loader = DataLoader(TensorDataset(X_train_t, Y_train_t), batch_size=batch_size, shuffle=True)
        
        model = CustomModel(Configs(seq_len=input_len, pred_len=forecast_horizon, enc_in=D)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            for X_batch, Y_batch in train_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                optimizer.zero_grad()
                pred = model(X_batch)
                loss = loss_fn(pred, Y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            yield f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}", None, None, None
        
        model.eval()
        # --- FIX: Simplified prediction loop to resolve PyTorch UserWarning ---
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        test_loader = DataLoader(X_test_t, batch_size=batch_size)
        with torch.no_grad():
            test_preds_norm = torch.cat([model(X_batch.to(device)).cpu() for X_batch in test_loader]).numpy()

        all_test_preds = scaler.inverse_transform(test_preds_norm.reshape(-1, D)).reshape(Y_test.shape) if scaler else test_preds_norm
        yield "complete", all_test_preds, Y_test, X_test_original
    except Exception as e:
        logger.error(f"Error in run_bi_igru: {e}")
        yield f"Error: {e}", None, None, None


def run_timemixerpp(df, forecast_horizon, input_len, lr, epochs, batch_size, normalize, train_ratio):
    """Generator for TimeMixerPP model, yields progress and final results."""
    try:
        data = df.values.astype(np.float32)
        if data.ndim == 1: data = data[:, None]
        X_train, Y_train, X_test, Y_test = split_train_test_sliding(data, input_len, forecast_horizon, train_ratio)
        if len(X_test) == 0: raise ValueError("Not enough data for test samples.")
        
        X_test_original = X_test.copy()
        N, L, D = X_train.shape

        scaler = StandardScaler() if normalize else None
        if scaler:
            scaler.fit(np.concatenate([X_train.reshape(-1, D), Y_train.reshape(-1, D)]))
            X_train = scaler.transform(X_train.reshape(-1, D)).reshape(X_train.shape)
            Y_train = scaler.transform(Y_train.reshape(-1, D)).reshape(Y_train.shape)
            X_test = scaler.transform(X_test.reshape(-1, D)).reshape(X_test.shape)
        
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
        train_loader = DataLoader(TensorDataset(X_train_t, Y_train_t), batch_size=batch_size, shuffle=True)
        
        model = TimeMixerPPModel(TimeMixerPPConfigs(seq_len=input_len, pred_len=forecast_horizon, enc_in=D)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            for X_batch, Y_batch in train_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                optimizer.zero_grad()
                pred = model(X_batch, None, None, None)
                loss = loss_fn(pred, Y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            yield f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}", None, None, None
        
        model.eval()
        # --- FIX: Simplified prediction loop to resolve PyTorch UserWarning ---
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        test_loader = DataLoader(X_test_t, batch_size=batch_size)
        with torch.no_grad():
            test_preds_norm = torch.cat([model(X_batch.to(device), None, None, None).cpu() for X_batch in test_loader]).numpy()

        all_test_preds = scaler.inverse_transform(test_preds_norm.reshape(-1, D)).reshape(Y_test.shape) if scaler else test_preds_norm
        yield "complete", all_test_preds, Y_test, X_test_original
    except Exception as e:
        logger.error(f"Error in run timemixerpp: {e}")
        yield f"Error: {e}", None, None, None


def run_arima(df, forecast_horizon, input_len, normalize, train_ratio):
    """Generator for ARIMA model, yields progress and final results."""
    try:
        data = df.values.astype(np.float32)
        if data.ndim == 1: data = data[:, None]
        _, _, X_test, Y_test = split_train_test_sliding(data, input_len, forecast_horizon, train_ratio)
        if len(X_test) == 0: raise ValueError("Not enough data for test samples.")
        
        all_preds_list = []
        for i in range(len(X_test)):
            sample_preds = []
            for j in range(data.shape[1]):
                history = X_test[i, :, j]
                scaler = StandardScaler() if normalize else None
                history_norm = scaler.fit_transform(history.reshape(-1, 1)).flatten() if scaler else history
                model = ARIMA(history_norm, order=(5,1,0)).fit()
                forecast_norm = model.forecast(steps=forecast_horizon)
                forecast = scaler.inverse_transform(forecast_norm.reshape(-1, 1)).flatten() if scaler else forecast_norm
                sample_preds.append(forecast)
            all_preds_list.append(np.stack(sample_preds, axis=-1))
            yield f"Processing test sample {i + 1}/{len(X_test)}", None, None, None
        
        yield "complete", np.stack(all_preds_list), Y_test, X_test
    except Exception as e:
        logger.error(f"Error in run_arima: {e}")
        yield f"Error: {e}", None, None, None

def run_nbeats(df, forecast_horizon, input_len, normalize, train_ratio):
    """Function for N-BEATS model, returns all results at once."""
    try:
        data = df.values.astype(np.float32)
        if data.ndim == 1: data = data[:, None]
        _, _, X_test, Y_test = split_train_test_sliding(data, input_len, forecast_horizon, train_ratio)
        if len(X_test) == 0: raise ValueError("Not enough data for test samples.")

        all_preds_list = []
        for j in range(data.shape[1]):
            series_data = df.iloc[:, j].values.astype(np.float32)
            scaler = StandardScaler() if normalize else None
            series_norm = scaler.fit_transform(series_data.reshape(-1, 1)).flatten() if scaler else series_data
            ts = TimeSeries.from_values(series_norm)
            train_end_idx = int(len(ts) * train_ratio)
            train_ts = ts[:train_end_idx]
            
            model = NBEATSModel(input_chunk_length=input_len, output_chunk_length=forecast_horizon, n_epochs=20, random_state=42, force_reset=True, pl_trainer_kwargs={"enable_progress_bar": False})
            model.fit(train_ts, verbose=False)
            
            preds_ts = model.historical_forecasts(ts, start=train_end_idx, forecast_horizon=forecast_horizon, stride=1, retrain=False, verbose=False, last_points_only=False, show_warnings=False)
            preds_for_var = np.array([p.values() for p in preds_ts]).squeeze()
            if scaler:
                preds_for_var = scaler.inverse_transform(preds_for_var.reshape(-1, 1)).reshape(preds_for_var.shape)
            
            num_samples_to_match = len(Y_test)
            all_preds_list.append(preds_for_var[:num_samples_to_match])
            
        return np.transpose(np.stack(all_preds_list, axis=0), (1, 2, 0)), Y_test, X_test
    except Exception as e:
        logger.error(f"Error in run_nbeats: {e}")
        raise

def plot_forecast(pred, gt, history=None, max_plot_vars=7, var_names=None):
    """Plots the forecast against the ground truth for a selected sample."""
    try:
        num_vars = pred.shape[1] if pred.ndim == 2 else 1
        num_vars_to_plot = min(num_vars, max_plot_vars)
        fig, axes = plt.subplots(num_vars_to_plot, 1, figsize=(10, 3 * num_vars_to_plot), sharex=True, squeeze=False)
        axes = axes.flatten()

        for i in range(num_vars_to_plot):
            ax = axes[i]
            if history is not None:
                history_data = history[:, i] if history.ndim > 1 else history
                ax.plot(range(len(history_data)), history_data, label="History", color='gray')
                start_idx = len(history_data)
            else:
                start_idx = 0
            gt_data = gt[:, i] if gt.ndim > 1 else gt
            pred_data = pred[:, i] if pred.ndim > 1 else pred
            ax.plot(range(start_idx, start_idx + len(gt_data)), gt_data, label="Ground Truth", color='blue', linestyle='--')
            ax.plot(range(start_idx, start_idx + len(pred_data)), pred_data, label="Forecast", color='orange')
            mae = mean_absolute_error(gt_data, pred_data)
            rmse = math.sqrt(mean_squared_error(gt_data, pred_data))
            
            # --- FIX: Changed `if var_names:` to `if var_names is not None:` to resolve pandas ambiguity error ---
            var_name = var_names[i] if var_names is not None else f"Variable {i+1}"
            ax.set_title(f"{var_name} (Sample-Specific) | MAE: {mae:.3f}, RMSE: {rmse:.3f}")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)
    except Exception as e:
        logger.error(f"Error plotting forecast: {str(e)}")
        raise

def plot_time_series(df, selected_vars=None, max_plot_vars=7):
    # This function remains unchanged.
    try:
        if selected_vars is None or len(selected_vars) == 0:
            selected_vars = df.columns.tolist()
        selected_vars = selected_vars[:max_plot_vars]
        selected_data = df[selected_vars]
        fig, ax = plt.subplots(figsize=(10, 5))
        selected_data.plot(ax=ax)
        ax.set_title(f"Input Time Series (Showing up to {max_plot_vars} vars)")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)
    except Exception as e:
        logger.error(f"Error plotting time series: {str(e)}")
        raise

def handle_file_input(file):
    # This function remains unchanged.
    try:
        if not file: return "❌ No file uploaded.", gr.update(choices=[], value=[])
        df = pd.read_csv(file.name)
        if df.columns[0].lower() in ['date', 'time', 'datetime']:
            df = df.iloc[:, 1:]
        df = df.dropna()
        if not df.select_dtypes(include=[np.number]).columns.tolist():
            return "❌ No numeric columns found.", gr.update(choices=[], value=[])
        img = plot_time_series(df)
        choices = df.columns.tolist()
        return img, gr.update(choices=choices, value=choices[:7])
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        return f"❌ Error: {str(e)}", gr.update(choices=[], value=[])

def update_plot(file, selected_vars):
    # This function remains unchanged.
    try:
        if not file: return "❌ No file uploaded."
        df = pd.read_csv(file.name)
        if df.columns[0].lower() in ['date', 'time', 'datetime']:
            df = df.iloc[:, 1:]
        df = df.dropna()
        return plot_time_series(df, selected_vars)
    except Exception as e:
        logger.error(f"Error updating plot: {str(e)}")
        return f"❌ Error: {str(e)}"

def decision_indicator(mae_array, rmse_array, ranges):
    # This function remains unchanged.
    decisions = []
    for mae, rmse, r in zip(mae_array, rmse_array, ranges):
        norm_mae = mae / (r + 1e-9)
        norm_rmse = rmse / (r + 1e-9)
        value = max(norm_mae, norm_rmse)
        if value < 0.05: decisions.append("Good ✅")
        elif value < 0.15: decisions.append("Fair ⚠️")
        else: decisions.append("Weak ❌")
    return decisions


# def forecast_app(file, model_type, forecast_horizon, input_len, lr, epochs, batch_size, normalize, variable_selector, train_ratio, sample_index):
#     try:
#         if not file:
#             return None, None, None, None, "❌ Error: Please upload a CSV file first.", ""

#         df = pd.read_csv(file.name)
#         if df.columns[0].lower() in ['date', 'time', 'datetime']: df = df.iloc[:, 1:]
#         df = df.dropna()
#         if variable_selector: df = df[variable_selector]

#         time_series_plot = plot_time_series(df)
#         yield None, None, None, time_series_plot, "Preparing to run model...", "### 📜 Training Log\n"
        
#         forecast_horizon = int(forecast_horizon)
#         input_len = int(input_len)
#         epochs = int(epochs)
#         batch_size = int(batch_size)
#         sample_index = int(sample_index)
#         lr = float(lr)
#         train_ratio = float(train_ratio)
        
#         all_preds, all_gt, all_history = None, None, None
#         training_log = "### 📜 Training Log\n"
        
#         if model_type == "Bi-iGRU":
#             model_runner = run_bi_igru(df, forecast_horizon, input_len, lr, epochs, batch_size, normalize, train_ratio)
#             for status, p, g, h in model_runner:
#                 if status != "complete":
#                     if "Error" in status: raise ValueError(status)
#                     training_log += status + "\n"
#                     yield None, None, None, time_series_plot, "Training in progress...", training_log
#                 else:
#                     all_preds, all_gt, all_history = p, g, h
#                     break

#         elif model_type == "TimeMixer++":
#             model_runner = run_timemixerpp(df, forecast_horizon, input_len, lr, epochs, batch_size, normalize, train_ratio)
#             for status, p, g, h in model_runner:
#                 if status != "complete":
#                     if "Error" in status: raise ValueError(status)
#                     training_log += status + "\n"
#                     yield None, None, None, time_series_plot, "Training in progress...", training_log
#                 else:
#                     all_preds, all_gt, all_history = p, g, h
#                     break

#         elif model_type == "ARIMA":
#             model_runner = run_arima(df, forecast_horizon, input_len, normalize, train_ratio)
#             for status, p, g, h in model_runner:
#                 if status != "complete":
#                     if "Error" in status: raise ValueError(status)
#                     training_log += status + "\n"
#                     yield None, None, None, time_series_plot, "Processing in progress...", training_log
#                 else:
#                     all_preds, all_gt, all_history = p, g, h
#                     break
#         elif model_type == "N-BEATS":
#             training_log += "Running N-BEATS... (This may take a moment)\n"
#             yield None, None, None, time_series_plot, "Running N-BEATS...", training_log
#             all_preds, all_gt, all_history = run_nbeats(df, forecast_horizon, input_len, normalize, train_ratio)

#         training_log += "✅ Processing complete.\n"

#         if all_preds is None:
#             return None, None, None, time_series_plot, "❌ Error: Model failed to produce predictions.", training_log
        
#         overall_mae_val = mean_absolute_error(all_gt.flatten(), all_preds.flatten())
#         overall_rmse_val = np.sqrt(mean_squared_error(all_gt.flatten(), all_preds.flatten()))

#         num_test_samples = len(all_preds)
#         if not (0 <= sample_index < num_test_samples):
#             sample_index = num_test_samples - 1
        
#         pred_to_plot = all_preds[sample_index]
#         gt_to_plot = all_gt[sample_index]
#         history_to_plot = all_history[sample_index]

#         forecast_img = plot_forecast(pred_to_plot, gt_to_plot, history_to_plot, var_names=df.columns)

#         mae_array, rmse_array, var_ranges = [], [], []
#         for i in range(all_preds.shape[-1]):
#             gt_var_all, pred_var_all = all_gt[..., i], all_preds[..., i]
#             # print(gt_var_all.shape, pred_var_all.shape)  # Debugging line
#             mae_array.append(mean_absolute_error(gt_var_all, pred_var_all))
#             rmse_array.append(math.sqrt(mean_squared_error(gt_var_all, pred_var_all)))
#             var_ranges.append(df.iloc[:, i].max() - df.iloc[:, i].min())
        
#         decisions = decision_indicator(mae_array, rmse_array, var_ranges)
#         decision_msg = "### 📊 Final Decision Summary (across all test samples):\n"
#         for i, (mae, rmse, dec) in enumerate(zip(mae_array, rmse_array, decisions)):
#             decision_msg += f"- **{df.columns[i]}**: MAE={mae:.3f}, RMSE={rmse:.3f} ➡️ {dec}\n"
        
#         yield forecast_img, f"{overall_mae_val:.4f}", f"{overall_rmse_val:.4f}", time_series_plot, decision_msg, training_log

#     except Exception as e:
#         logger.error(f"Forecast app error: {str(e)}")
#         error_message = f"❌ An error occurred: {str(e)}"
#         final_log = (training_log if 'training_log' in locals() else "") + f"\n**{error_message}**"
#         return None, None, None, None, error_message, final_log

def forecast_app(file, model_type, forecast_horizon, input_len, lr, epochs, batch_size, normalize, variable_selector, train_ratio, sample_index):
    try:
        if not file:
            return None, None, None, None, "❌ Error: Please upload a CSV file first.", pd.DataFrame([], columns=["Epoch", "Loss"]), 0.0, ""

        df = pd.read_csv(file.name)
        if df.columns[0].lower() in ['date', 'time', 'datetime']:
            df = df.iloc[:, 1:]
        df = df.dropna()
        if variable_selector:
            df = df[variable_selector]

        time_series_plot = plot_time_series(df)
        log_rows = []
        yield None, None, None, time_series_plot, "Preparing to run model...", pd.DataFrame([], columns=["Epoch", "Loss"]), 0.0, "📊 Decision summary will appear during training..."

        # convert params
        forecast_horizon = int(forecast_horizon)
        input_len = int(input_len)
        epochs = int(epochs)
        batch_size = int(batch_size)
        sample_index = int(sample_index)
        lr = float(lr)
        train_ratio = float(train_ratio)

        all_preds, all_gt, all_history = None, None, None

        def build_decision_message(preds, gt):
            """Compute a provisional decision summary."""
            if preds is None or gt is None:
                return "📊 Collecting results..."
            mae_array, rmse_array, var_ranges = [], [], []
            for i in range(preds.shape[-1]):
                gt_var_all, pred_var_all = gt[..., i], preds[..., i]
                mae_array.append(mean_absolute_error(gt_var_all, pred_var_all))
                rmse_array.append(math.sqrt(mean_squared_error(gt_var_all, pred_var_all)))
                var_ranges.append(df.iloc[:, i].max() - df.iloc[:, i].min())
            decisions = decision_indicator(mae_array, rmse_array, var_ranges)
            msg = "### 📊 Decision Summary (Provisional):\n"
            for i, (mae, rmse, dec) in enumerate(zip(mae_array, rmse_array, decisions)):
                msg += f"- **{df.columns[i]}**: MAE={mae:.3f}, RMSE={rmse:.3f} ➡️ {dec}\n"
            return msg

        # --- Run selected model ---
        if model_type == "Bi-iGRU":
            model_runner = run_bi_igru(df, forecast_horizon, input_len, lr, epochs, batch_size, normalize, train_ratio)
            for idx, (status, p, g, h) in enumerate(model_runner):
                if status != "complete":
                    if "Error" in status:
                        raise ValueError(status)
                    log_rows.append([idx + 1, float(status.split("Loss:")[-1]) if "Loss" in status else None])
                    decision_msg = build_decision_message(p, g)
                    yield None, None, None, time_series_plot, "Training in progress...", pd.DataFrame(log_rows, columns=["Epoch", "Loss"]), (idx + 1) / epochs, decision_msg
                else:
                    all_preds, all_gt, all_history = p, g, h
                    break

        elif model_type == "TimeMixer++":
            model_runner = run_timemixerpp(df, forecast_horizon, input_len, lr, epochs, batch_size, normalize, train_ratio)
            for idx, (status, p, g, h) in enumerate(model_runner):
                if status != "complete":
                    if "Error" in status:
                        raise ValueError(status)
                    log_rows.append([idx + 1, float(status.split("Loss:")[-1]) if "Loss" in status else None])
                    decision_msg = build_decision_message(p, g)
                    yield None, None, None, time_series_plot, "Training in progress...", pd.DataFrame(log_rows, columns=["Epoch", "Loss"]), (idx + 1) / epochs, decision_msg
                else:
                    all_preds, all_gt, all_history = p, g, h
                    break

        elif model_type == "ARIMA":
            model_runner = run_arima(df, forecast_horizon, input_len, normalize, train_ratio)
            for idx, (status, p, g, h) in enumerate(model_runner):
                if status != "complete":
                    if "Error" in status:
                        raise ValueError(status)
                    decision_msg = build_decision_message(p, g)
                    yield None, None, None, time_series_plot, f"{status}", pd.DataFrame(log_rows, columns=["Epoch", "Loss"]), 0.0, decision_msg
                else:
                    all_preds, all_gt, all_history = p, g, h
                    break

        elif model_type == "N-BEATS":
            yield None, None, None, time_series_plot, "Running N-BEATS...", pd.DataFrame(log_rows, columns=["Epoch", "Loss"]), 0.0, "📊 Decision summary updating..."
            all_preds, all_gt, all_history = run_nbeats(df, forecast_horizon, input_len, normalize, train_ratio)

        # --- Evaluation ---
        if all_preds is None:
            return None, None, None, time_series_plot, "❌ Error: Model failed to produce predictions.", pd.DataFrame(log_rows, columns=["Epoch", "Loss"]), 1.0, ""

        overall_mae_val = mean_absolute_error(all_gt.flatten(), all_preds.flatten())
        overall_rmse_val = np.sqrt(mean_squared_error(all_gt.flatten(), all_preds.flatten()))

        # select sample
        num_test_samples = len(all_preds)
        if not (0 <= sample_index < num_test_samples):
            sample_index = num_test_samples - 1
        pred_to_plot = all_preds[sample_index]
        gt_to_plot = all_gt[sample_index]
        history_to_plot = all_history[sample_index]

        forecast_img = plot_forecast(pred_to_plot, gt_to_plot, history_to_plot, var_names=df.columns)

        # --- Final decision summary ---
        decision_msg = build_decision_message(all_preds, all_gt).replace("Provisional", "Final")

        yield forecast_img, f"{overall_mae_val:.4f}", f"{overall_rmse_val:.4f}", \
              time_series_plot, "✅ Training complete.", pd.DataFrame(log_rows, columns=["Epoch", "Loss"]), 1.0, decision_msg

    except Exception as e:
        logger.error(f"Forecast app error: {str(e)}")
        error_message = f"❌ An error occurred: {str(e)}"
        final_log = pd.DataFrame(log_rows if 'log_rows' in locals() else [], columns=["Epoch", "Loss"])
        return None, None, None, None, error_message, final_log, 0.0, ""

    

# with gr.Blocks(theme=gr.themes.Glass()) as app:
#     gr.Markdown("## 📊 Time Series Forecasting App")
#     gr.Markdown("Upload a CSV, select a model, and hit the forecast button. Each column represents a different time series. The first column is usually the timestamp, which is not utilized in the forecasting process. The app calculates MAE/RMSE across all test samples and lets you visualize any specific test sample.")

#     with gr.Row():
#         with gr.Column(scale=1):
#             file_input = gr.File(label="Upload your CSV file")
#             model_selector = gr.Radio(["Bi-iGRU", "TimeMixer++", "ARIMA", "N-BEATS"], label="Select Model", value="Bi-iGRU")
#             variable_selector = gr.Dropdown(choices=[], label="Select Variables", multiselect=True, info="Select up to 7. Leave empty for all.")
            
            
#             with gr.Accordion("Advanced Parameters", open=False):
#                 forecast_horizon = gr.Number(label="Forecast Horizon", value=12, precision=0)
#                 input_length = gr.Number(label="Input Length (History)", value=36, precision=0)
#                 train_ratio = gr.Slider(minimum=0.1, maximum=0.9, step=0.05, label="Train/Test Split", value=0.8)
#                 sample_index_input = gr.Number(label="Sample Index to Plot", value=-1, info="Which test sample to visualize (-1 for last).", precision=0)
#                 normalize_checkbox = gr.Checkbox(label="Normalize Data", value=True)
            
#             with gr.Accordion("Deep Learning Parameters", open=True):
#                 learning_rate = gr.Number(label="Learning Rate", value=0.001)
#                 epochs = gr.Number(label="Epochs", value=10, precision=0)
#                 batch_size = gr.Number(label="Batch Size", value=32, precision=0)

#             run_button = gr.Button("📊 Run", variant="primary")

#         with gr.Column(scale=2):
#             status_msg = gr.Markdown("Status: Waiting for input...", label="Status")
#             training_log_display = gr.Markdown("### 📜 Training Log\n", label="Training Log")
#             with gr.Row():
#                 overall_mae = gr.Textbox(label="Overall MAE (all test samples)", interactive=False)
#                 overall_rmse = gr.Textbox(label="Overall RMSE (all test samples)", interactive=False)
#             with gr.Tabs():
#                 with gr.TabItem("Forecast Plot"):
#                     forecast_plot = gr.Image(label="Forecast vs. Ground Truth (for selected sample)", type="pil")
#                 with gr.TabItem("Input Data"):
#                     time_series_plot = gr.Image(label="Input Time Series", type="pil")

#     file_input.change(fn=handle_file_input, inputs=file_input, outputs=[time_series_plot, variable_selector])
#     variable_selector.change(fn=update_plot, inputs=[file_input, variable_selector], outputs=time_series_plot)
    
#     run_button.click(
#         fn=forecast_app,
#         inputs=[
#             file_input, model_selector, forecast_horizon, input_length,
#             learning_rate, epochs, batch_size, normalize_checkbox,
#             variable_selector, train_ratio, sample_index_input
#         ],
#         outputs=[
#             forecast_plot, overall_mae, overall_rmse, 
#             time_series_plot, status_msg, training_log_display
#         ]
#     )

with gr.Blocks(theme=gr.themes.Glass()) as app:
    gr.Markdown("## 📊 Time Series Forecasting App")

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload your CSV file")
            model_selector = gr.Radio(["Bi-iGRU", "TimeMixer++", "ARIMA", "N-BEATS"], label="Select Model", value="Bi-iGRU")
            variable_selector = gr.Dropdown(choices=[], label="Select Variables", multiselect=True, info="Select up to 7. Leave empty for all.")
            
            
            with gr.Accordion("Advanced Parameters", open=False):
                forecast_horizon = gr.Number(label="Forecast Horizon", value=12, precision=0)
                input_length = gr.Number(label="Input Length (History)", value=36, precision=0)
                train_ratio = gr.Slider(minimum=0.1, maximum=0.9, step=0.05, label="Train/Test Split", value=0.8)
                sample_index_input = gr.Number(label="Sample Index to Plot", value=-1, info="Which test sample to visualize (-1 for last).", precision=0)
                normalize_checkbox = gr.Checkbox(label="Normalize Data", value=True)
            
            with gr.Accordion("Deep Learning Parameters", open=True):
                learning_rate = gr.Number(label="Learning Rate", value=0.001)
                epochs = gr.Number(label="Epochs", value=10, precision=0)
                batch_size = gr.Number(label="Batch Size", value=32, precision=0)
            run_button = gr.Button("📊 Run", variant="primary")

        with gr.Column(scale=2):
            status_msg = gr.Markdown("Status: Waiting for input...")
            progress_bar = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.0, interactive=False, label="Training Progress")
            training_log_table = gr.Dataframe(headers=["Epoch", "Loss"], datatype=["number", "number"], row_count=5, col_count=2, label="Training Log", interactive=False)
            
            with gr.Row():
                overall_mae = gr.Textbox(label="Overall MAE", interactive=False)
                overall_rmse = gr.Textbox(label="Overall RMSE", interactive=False)

            decision_output = gr.Markdown("### 📊 Decision Summary will appear here")
            with gr.Tabs():
                with gr.TabItem("Forecast Plot"):
                    forecast_plot = gr.Image(label="Forecast vs. Ground Truth", type="pil")
                with gr.TabItem("Input Data"):
                    time_series_plot = gr.Image(label="Input Time Series", type="pil")

    file_input.change(fn=handle_file_input, inputs=file_input, outputs=[time_series_plot, variable_selector])
    variable_selector.change(fn=update_plot, inputs=[file_input, variable_selector], outputs=time_series_plot)

    run_button.click(
        fn=forecast_app,
        inputs=[file_input, model_selector, forecast_horizon, input_length,
                learning_rate, epochs, batch_size, normalize_checkbox,
                variable_selector, train_ratio, sample_index_input],
        outputs=[forecast_plot, overall_mae, overall_rmse,
                 time_series_plot, status_msg, training_log_table, progress_bar, decision_output]
    )
    

if __name__ == '__main__':
    logger.info("Starting Gradio app")
    app.launch(debug=True, share=False)