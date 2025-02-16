import os
import torch
from torch.utils.tensorboard import SummaryWriter

class ModelTensorBoardLogger:
    """
    TensorBoard logger for model training.
    Logs accuracy, loss, histograms, and hyperparameters.
    """
    def __init__(self, log_dir="logs/model_monitor"):
        self.writer = SummaryWriter(log_dir)
        print(f"TensorBoard model logging initialized at: {os.path.abspath(log_dir)}")

    def log_scalar(self, tag, value, step):
        """Logs a scalar metric (e.g., loss, accuracy)."""
        self.writer.add_scalar(tag, float(value), step)
        self.writer.flush()  # Force write to disk

    def log_histogram(self, tag, values, step):
        """Logs a histogram (e.g., weights, feature distributions)."""
        self.writer.add_histogram(tag, values.cpu().numpy(), step)

    def log_hparams(self, hparams, metrics):
        """Logs hyperparameters and their corresponding evaluation metrics."""
        self.writer.add_hparams(hparams, metrics)

    def close(self):
        """Closes the TensorBoard writer."""
        self.writer.close()
