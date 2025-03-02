import os
import time
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates, nvmlDeviceGetTemperature, NVML_TEMPERATURE_GPU
from torch.utils.tensorboard import SummaryWriter

class GPUTensorBoardLogger:
    """
    TensorBoard logger for GPU monitoring using pynvml.
    """
    def __init__(self, log_dir="logs/gpu_monitor"):
        self.writer = SummaryWriter(log_dir)
        nvmlInit()  # Initialize NVML
        print(f"TensorBoard GPU monitoring initialized at: {os.path.abspath(log_dir)}")

    def get_gpu_stats(self, gpu_index=0):
        """Fetches GPU memory usage, utilization, and temperature."""
        handle = nvmlDeviceGetHandleByIndex(gpu_index)
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        util_info = nvmlDeviceGetUtilizationRates(handle)
        temp_info = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)

        return {
            f"GPU_{gpu_index}/Memory_Used_MB": mem_info.used / 1e6,
            f"GPU_{gpu_index}/Memory_Total_MB": mem_info.total / 1e6,
            f"GPU_{gpu_index}/Memory_Usage_Percent": (mem_info.used / mem_info.total) * 100,
            f"GPU_{gpu_index}/Utilization_Percent": util_info.gpu,
            f"GPU_{gpu_index}/Temperature_Celsius": temp_info
        }

    def log_gpu_stats(self, step, gpu_index=0):
        """Logs GPU usage metrics to TensorBoard."""
        stats = self.get_gpu_stats(gpu_index)
        print(f"Stats: {stats}")
        for key, value in stats.items():
            self.writer.add_scalar(key, value, step)

    def close(self):
        """Closes the TensorBoard writer."""
        self.writer.close()
