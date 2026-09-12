# Test 3: Head pose test

## Goal

This is a brief test of the 6DRepNet model to obtain head rotation angles across three axes and subsequently work with them. The primary goal of this experiment is to obtain the head rotation angles.

This experiment forms the basis of a script for preparing data to test the response of facial recognition models to head rotation angles across various head positions.

## Methodology

- 177 frames extracted from a video
- `SixDRepNet` used to estimate pitch, yaw and roll
- Frames sorted by the tested head pose axis
- Frames grouped into approximately 1-degree angle intervals
- For each target angle, the frame with the lowest influence from the other two axes was selected
- Influence for each axis:
    - Pitch test: score = |yaw| + |roll|
    - Yaw test: score = |pitch| + |roll|
    - Roll test: score = |pitch| + |yaw|

## What is being tested

- Stability of head pose estimation with SixDRepNet
- Sorting algorithm for each axis with minimal influence from other two

`video_processing.py` processes a video, takes frames from video and saves it in output directory.
`position_processing.py` processes each frame, takes angles along the axes, then sorts it and selects best frames by axis for each target angle with minimal influence from other two.

## Results

### Pitch

Tested approximately from -24.6° to +42.7°

### Yaw

Tested approximately from -23.1° to +35.9°

### Roll

The current video did not contain enough roll movement for a meaningful test. A separate recording with deliberate head tilting is required.

Raw console logs with selected frames for each axis are in the txt document.