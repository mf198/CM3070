import os
from pynvml import (
    nvmlInit,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetTemperature,
    NVML_TEMPERATURE_GPU,
)
from torch.utils.tensorboard import SummaryWriter

class GPUTensorBoardLogger:
    def __init__(self, log_dir="logs/gpu_monitor", gpu_index=0):
        self.writer = SummaryWriter(log_dir)
        self.gpu_index = gpu_index
        nvmlInit()
        self.handle = nvmlDeviceGetHandleByIndex(gpu_index)
        print(f"TensorBoard GPU monitoring initialized at: {os.path.abspath(log_dir)}")

    def get_gpu_stats(self):
        mem_info = nvmlDeviceGetMemoryInfo(self.handle)
        util_info = nvmlDeviceGetUtilizationRates(self.handle)
        temp = nvmlDeviceGetTemperature(self.handle, NVML_TEMPERATURE_GPU)

        return {
            "Memory_Used_MB": mem_info.used / 1e6,
            "Memory_Total_MB": mem_info.total / 1e6,
            "Memory_Usage_Percent": (mem_info.used / mem_info.total) * 100,
            "Utilization_Percent": util_info.gpu,
            "Temperature_Celsius": temp,
        }

    def log_gpu_stats(self, step):
        stats = self.get_gpu_stats()
        for key, value in stats.items():
            self.writer.add_scalar(f"GPU_{self.gpu_index}/{key}", value, step)

    def close(self):
        self.writer.close()

