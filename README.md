# GPT2-Hugging-Face
Training and deployment outline for GPT2 with Hugging Face

# GPT-2 Spotify Benchmark

## Setup
```
bash scripts/setup_environment.sh
```

## Train GPT-2
```
python3 scripts/train.py --size 10000
```

## Monitor Resources (Optional)
```
python3 scripts/monitor.py
```

## Output
- Model saved in `models/gpt2-spotify-<sample_size>`
- Training logs saved in `logs/training_times.csv`
- Resource usage in `logs/usage_monitor.csv`
