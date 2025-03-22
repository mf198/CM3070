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
        print(
            f"✅ TensorBoard model logging initialized at: {os.path.abspath(log_dir)}"
        )

    def log_scalar(self, tag, value, step):
        """Logs a single scalar value."""
        self.writer.add_scalar(tag, float(value), step)
        self.writer.flush()

    def log_scalars(self, scalar_dict, step):
        """
        Logs multiple scalars at once.

        Args:
            scalar_dict (dict): Dictionary of {tag: value}.
            step (int): Current training step (usually epoch).
        """
        for tag, value in scalar_dict.items():
            self.writer.add_scalar(tag, float(value), step)
        self.writer.flush()

    def log_training_loss(self, loss, epoch):
        """Convenience method to log training loss."""
        self.log_scalar("Loss/train", loss, epoch)

    def log_histogram(self, tag, values, step):
        """Logs a histogram (e.g., weights or feature distributions)."""
        self.writer.add_histogram(tag, values.cpu().numpy(), step)

    def log_hparams(self, hparams, metrics):
        """Logs hyperparameters and final metrics."""
        self.writer.add_hparams(hparams, metrics)

    def close(self):
        self.writer.close()
