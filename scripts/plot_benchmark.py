import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("logs", exist_ok=True)

log_path = 'logs/training_times.csv'
df = pd.read_csv(log_path, header=None, names=[
    "samples", "time", "cpu_start", "cpu_end", "mem_start", "mem_end",
    "gpu_start", "gpu_end", "gpumem_start", "gpumem_end"])

plt.figure()
df.plot(x="samples", y="time", marker='o', title="Training Time vs Sample Size")
plt.ylabel("Seconds")
plt.grid(True)
plt.savefig("logs/benchmark_training_time.png")

plt.figure()
df[["samples", "cpu_start", "cpu_end"]].plot(x="samples", marker='o', title="CPU Utilization (%)")
plt.ylabel("%")
plt.grid(True)
plt.savefig("logs/benchmark_cpu.png")

plt.figure()
df[["samples", "gpu_start", "gpu_end"]].plot(x="samples", marker='o', title="GPU Utilization (%)")
plt.ylabel("%")
plt.grid(True)
plt.savefig("logs/benchmark_gpu.png")

print("✅ Benchmark plots saved to logs/")
