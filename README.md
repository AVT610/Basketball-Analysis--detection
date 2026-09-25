# Basketball Analysis - Detection

A computer-vision pipeline for detecting and tracking basketball players and the ball in game footage, then rendering annotated output video.

## What this project does

- Reads an input basketball video
- Detects **players** and **ball** using YOLO models
- Tracks players across frames with ByteTrack
- Keeps the most confident ball detection per frame
- Removes implausible ball detections and interpolates missing ball positions
- Draws player IDs and ball markers on frames
- Exports an annotated output video

## Project structure

- `main.py` - End-to-end inference pipeline
- `trackers/`
  - `player_tracker.py` - Player detection + tracking
  - `ball_tracker.py` - Ball detection + cleanup + interpolation
- `drawers/`
  - `player_tracks_drawer.py` - Player ellipse/ID overlays
  - `ball_tracks_drawer.py` - Ball marker overlays
- `utils/`
  - `video_utils.py` - Video read/write helpers
  - `stubs_utils.py` - Detection/track stub read/write helpers
  - `bbox_utils.py` - Bounding-box math utilities
- `training_notebooks/` - Notebook workflows for ball/player model training

## Requirements

Install dependencies in your environment:

```bash
pip install ultralytics supervision opencv-python numpy pandas
```

You also need trained model files (not committed in this repo):

- `models/player_detector.pt`
- `models/ball_detector_model.pt`

## Input/Output layout

Expected input and generated output paths used by `main.py`:

- Input video: `input_videos/video_1.mp4`
- Output video: `output_videos/output_video.avi`
- Optional cached tracks (stubs):
  - `stubs/player_track_stubs.pkl`
  - `stubs/ball_track_stubs.pkl`

> Note: video, model, and stub files are ignored by git in this repository.

## Run the pipeline

From repository root:

```bash
python main.py
```

## Notes

- The pipeline is configured for class names `Player` and `Ball` in the trained YOLO models.
- Track stubs can speed up repeat runs when using the same input video.
- Ball interpolation fills missing detections to produce smoother trajectories.
