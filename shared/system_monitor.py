import threading
import time
import psutil
import pynvml
import matplotlib.pyplot as plt
import numpy as np
import os

class SystemMonitor:
    """
    A class to monitor CPU, GPU, and Disk usage in a separate thread.
    """
    def __init__(self, sample_interval=1.0):
        self.interval = sample_interval
        self._running = False
        self.monitoring_thread = None

        # Data storage
        self.timestamps = []
        self.cpu_percent = []
        self.gpu_percent = []
        self.gpu_mem_percent = []
        self.disk_read_mb = []
        self.disk_write_mb = []
        
        # Initialize pynvml for GPU monitoring
        try:
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.gpu_available = True
        except pynvml.NVMLError:
            self.gpu_available = False
            print("NVIDIA GPU not found or pynvml failed to initialize.")

        # Get initial disk counters
        self.last_disk_io = psutil.disk_io_counters()

    def _monitor_loop(self):
        """The main monitoring loop that runs in the background."""
        start_time = time.time()
        while self._running:
            # Timestamp
            current_time = time.time() - start_time
            self.timestamps.append(current_time)

            # CPU Usage
            self.cpu_percent.append(psutil.cpu_percent(interval=None))

            # GPU Usage
            if self.gpu_available:
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                self.gpu_percent.append(gpu_util.gpu)
                self.gpu_mem_percent.append(100 * gpu_mem.used / gpu_mem.total)
            
            # Disk Usage (rate of change)
            current_disk_io = psutil.disk_io_counters()
            read_bytes = current_disk_io.read_bytes - self.last_disk_io.read_bytes
            write_bytes = current_disk_io.write_bytes - self.last_disk_io.write_bytes
            self.disk_read_mb.append(read_bytes / (1024**2) / self.interval)
            self.disk_write_mb.append(write_bytes / (1024**2) / self.interval)
            self.last_disk_io = current_disk_io

            # Wait for the next interval
            time.sleep(self.interval)

    def start(self):
        """Starts the monitoring thread."""
        if self.monitoring_thread is not None:
            print("Monitoring is already running.")
            return
        
        self._running = True
        self.monitoring_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitoring_thread.start()
        print("System monitor started.")

    def stop(self):
        """Stops the monitoring thread."""
        if self.monitoring_thread is None:
            print("Monitoring is not running.")
            return
            
        self._running = False
        self.monitoring_thread.join()
        self.monitoring_thread = None
        if self.gpu_available:
            pynvml.nvmlShutdown()
        print("System monitor stopped.")

    def plot_results(self):
        """Plots the collected system usage data."""
        if not self.timestamps:
            print("No data collected to plot.")
            return

        fig, axs = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
        fig.suptitle('System Resource Usage During Training', fontsize=16)

        # Plot CPU Usage
        axs[0].plot(self.timestamps, self.cpu_percent, label='CPU Usage', color='dodgerblue')
        axs[0].set_ylabel('Usage (%)')
        axs[0].set_title('CPU Utilization')
        axs[0].grid(True, linestyle='--', alpha=0.6)
        axs[0].legend()

        # Plot GPU Usage
        if self.gpu_available and self.gpu_percent:
            axs[1].plot(self.timestamps, self.gpu_percent, label='GPU Utilization', color='limegreen')
            axs[1].plot(self.timestamps, self.gpu_mem_percent, label='GPU Memory Usage', color='darkorange', linestyle='--')
            axs[1].set_ylabel('Usage (%)')
            axs[1].set_title('GPU Utilization & Memory')
            axs[1].grid(True, linestyle='--', alpha=0.6)
            axs[1].legend()
        else:
            axs[1].text(0.5, 0.5, 'GPU Data Not Available', ha='center', va='center', fontsize=12)
            axs[1].set_title('GPU Utilization & Memory')


        # Plot Disk I/O
        axs[2].plot(self.timestamps, self.disk_read_mb, label='Disk Read', color='crimson')
        axs[2].plot(self.timestamps, self.disk_write_mb, label='Disk Write', color='purple', linestyle='--')
        axs[2].set_xlabel('Time (s)')
        axs[2].set_ylabel('Throughput (MB/s)')
        axs[2].set_title('Disk I/O')
        axs[2].grid(True, linestyle='--', alpha=0.6)
        axs[2].legend()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()