# ccfd/utils/tensorboard_gpu_logger.py
import pynvml
import torch
import time
from torch.utils.tensorboard import SummaryWriter

class TensorBoardGPUMonitor:
    def __init__(self, log_dir="runs/gpu_monitor"):
        """
        Initializes TensorBoard writer and NVML for GPU monitoring.
        
        Args:
            log_dir (str): Directory to store TensorBoard logs.
        """
        self.writer = SummaryWriter(log_dir)
        pynvml.nvmlInit()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming single GPU

    def get_gpu_stats(self):
        """
        Get GPU memory usage, utilization, and temperature using pynvml.
        
        Returns:
            Dictionary of GPU statistics.
        """        

        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
        gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
        temp_info = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, NVML_TEMPERATURE_GPU)

        return {
            f"Memory_Used_MB": mem_info.used / 1e6,
            f"Memory_Total_MB": mem_info.total / 1e6,
            f"Memory_Usage_Percent": (mem_info.used / mem_info.total) * 100,
            f"Utilization_Percent": gpu_util.gpu,
            f"Temperature_Celsius": temp_info
        }

    def log_gpu_metrics(self, step):
        """
        Logs GPU memory usage and utilization to TensorBoard.
        
        Args:
            step (int): Training step or epoch.
        """
        stats = self.get_gpu_stats()
        for key, value in stats.items():
            self.writer.add_scalar(key, value, step)

    def close(self):
        """Closes TensorBoard writer."""
        self.writer.close()
