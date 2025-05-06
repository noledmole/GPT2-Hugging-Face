# GPT-2 Spotify Benchmark

## Setup
```
bash scripts/setup_environment.sh
```

## EBS Storage (Recommended for Large Datasets)
If your dataset is >100 GB (e.g. full 600 GB Spotify dataset), you should:

1. **Allocate larger EBS volume (e.g. 900 GB)** when launching EC2.
2. **After launch**, SSH into EC2 and format + mount it:
```bash
sudo mkfs -t ext4 /dev/nvme1n1
sudo mkdir /mnt/spotify_data
sudo mount /dev/nvme1n1 /mnt/spotify_data
```
3. **Make it persist on reboot** (optional):
```bash
echo '/dev/nvme1n1 /mnt/spotify_data ext4 defaults,nofail 0 2' | sudo tee -a /etc/fstab
```
4. **Put your CSV there**:
```bash
mv spotify_dataset.csv /mnt/spotify_data/
```

Then in your script, change the dataset path to:
```python
dataset_path = '/mnt/spotify_data/spotify_dataset.csv'
```

## Train GPT-2
```
python3 scripts/train.py --size 10000
```

## Monitor Resources (Optional)
```
python3 scripts/monitor.py
```

## Serve Model
```
MODEL_DIR=models/gpt2-spotify-10000 uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Output
- Model saved in `models/gpt2-spotify-<sample_size>`
- Training logs saved in `logs/training_times.csv`
- Resource usage in `logs/usage_monitor.csv`
- API available at POST `/generate` with body: `{ "text": "your prompt" }`
