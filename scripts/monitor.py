import time
import csv
import psutil
import subprocess
import os
import datetime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--logfile', type=str, default='logs/usage_monitor.csv', help='Path to usage log CSV file')
args = parser.parse_args()

os.makedirs("logs", exist_ok=True)

with open(args.logfile, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['unix_timestamp', 'readable_time', 'cpu_percent', 'mem_percent', 'gpu_util', 'gpu_mem'])
    f.flush()


    print("Resource monitoring started. Logging every 10s to logs/usage_monitor.csv...")

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
        time.sleep(10)

