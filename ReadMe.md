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

Place an input video at the path set in `paths.input_video` (default `videos/input/input_h264.mp4`), then run from the project root:

```bash
python src/detect.py
```

The annotated video is saved to `paths.output_dir` as `<model>_<backend>_img<size>_conf<conf>.mp4`. Test runs get a `_test<N>` suffix.

## Configuration
Model settings, file paths, detection and tracker parameters are managed through `config/config.yaml`.

For a quick check, `run.test_mode: true` processes only the first `run.test_num_frames` frames (default 50). Set `test_mode: false` to process the whole video.
