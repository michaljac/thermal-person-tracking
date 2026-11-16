# Thermal Person Tracking

A Computer Vision project for person detection and identity-preserving tracking in thermal (white-hot) video, with a focus on occlusion recovery and inference performance.

## Objectives
- Detect people in thermal imagery using YOLO.
- Track individuals across frames using DeepSORT.
- Maintain target identity through occlusions using Re-Identification (ReID).
- Identify performance bottlenecks through latency profiling.
- Optimize inference for CPU-based environments.

## Tech Stack
Python · PyTorch · Ultralytics YOLO · OpenCV · DeepSORT · Docker · YAML · Matplotlib


## Getting Started

Build the Docker image:

```bash
docker build -t thermal-person-tracking:cpu .
```

Run the container with the project mounted:

```bash
docker run -it --rm -v "${PWD}:/workspace" -w /workspace thermal-person-tracking:cpu bash
```

Place an input video at `videos/input/input.mp4`, then run:

```bash
python -m detect
```

Detection results are saved to `videos/output/`.

## Configuration
Model settings, file paths, and detection parameters are managed through `config/config.yaml`.