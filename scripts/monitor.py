import time
import csv
import psutil
import subprocess
import os
import datetime

os.makedirs("logs", exist_ok=True)

log_path = "logs/usage_monitor.csv"

with open(log_path, 'a', newline='') as f:
    f.write(f",,,,START OF NEW RUN {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    writer = csv.writer(f)
    writer.writerow(['unix_timestamp', 'readable_time', 'cpu_percent', 'mem_percent', 'gpu_util', 'gpu_mem'])

    print("Resource monitoring started. Logging every 10s to logs/usage_monitor.csv...")

    try:
        while True:
            unix_time = time.time()
            readable_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
            cpu = psutil.cpu_percent()
            mem = psutil.virtual_memory().percent
            try:
                nvidia = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,nounits,noheader'])
                gpu_util, gpu_mem = map(int, nvidia.decode().strip().split(','))

            except:
                gpu_util, gpu_mem = 0, 0

            writer.writerow([unix_time, readable_time, cpu, mem, gpu_util, gpu_mem])
            f.flush()
            time.sleep(10)
    except KeyboardInterrupt:
        print("Monitoring stopped.")
