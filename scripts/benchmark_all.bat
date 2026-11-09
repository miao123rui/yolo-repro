@echo off
python src\benchmark_resources.py --config configs\benchmark_yolov8n.yaml
python src\benchmark_resources.py --config configs\benchmark_yolov8s.yaml
