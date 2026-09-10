# Face Recognition Model Testing

A test repository for selecting face recognition models and comparison methods for an identity verification service (selfie video vs document photo or scan).

## Structure

- `test_1/` — pilot comparison of 5 DeepFace models on a small dataset
- `test_2/` — small model sensitivity test to head rotation/tilt angles 

Each test includes its own README describing the methodology and results.

## Tech Stack

Python, OpenCV, MTCNN, DeepFace (ArcFace, Facenet512, VGG-Face, Facenet, Dlib)
