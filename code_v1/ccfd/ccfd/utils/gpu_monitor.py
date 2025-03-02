# ccfd/utils/gpu_monitor.py
import time
import pynvml
import GPUtil

def get_gpu_info():
    """
    Returns the current GPU utilization and memory usage.

    Returns:
        dict: GPU usage details (utilization %, memory used, total memory).
    """
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assumes single GPU
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)

    return {
        "gpu_utilization": gpu_util.gpu,  # GPU usage in percentage
        "gpu_memory_used": mem_info.used / (1024 ** 2),  # Convert to MB
        "gpu_memory_total": mem_info.total / (1024 ** 2)  # Convert to MB
    }

def print_gpu_status():
    """
    Prints the current GPU status including utilization and memory usage.
    """
    gpu_info = get_gpu_info()
    print(f"🚀 GPU Status - Utilization: {gpu_info['gpu_utilization']}% | "
          f"Memory: {gpu_info['gpu_memory_used']:.2f}MB / {gpu_info['gpu_memory_total']:.2f}MB")

def track_gpu_during_training(model_function, X_train, y_train):
    """
    Tracks GPU performance before and after training.

    Args:
        model_function (function): The function that trains the model.
        X_train (cudf.DataFrame): Training features.
        y_train (cudf.Series): Training labels.

    Returns:
        model: The trained model.
    """
    #print("📊 Checking initial GPU status...")
    print_gpu_status()
    
    start_time = time.time()
    
    print(f"🛠️ Training {model_function.__name__} on GPU...")
    model = model_function(X_train, y_train)
    
    end_time = time.time()
    training_time = end_time - start_time

    print(f"✅ Training completed in {training_time:.2f} seconds.")
    #print("📊 Final GPU status:")
    print_gpu_status()

    return model, training_time
