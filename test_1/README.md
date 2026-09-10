# Test 1: Pilot comparison of 5 face recognition models

## Goal

This is a pilot comparison of 5 DeepFace models (Facenet512, ArcFace, Facenet, VGG-Face, Dlib) on a small dataset of my own photos. The main goal is to find out each model's basic reaction before making a full test with different groups of photos.

## Methodology

- 4 documents (photos and scans of documents)
- 12 photos of me in different conditions, named {type of light}{now/beard/doctime}{facial expression}, where doctime refers to photos from the time when I received the documents, so the face is the same.
- Each document is compared to each photo
- Metrics: cosine similarity on normalized embeddings (`raw` - raw similarity, `distance = 1 - raw`)

## What is being tested

- Basic stability for each model
- Sensitivity to time, type of light and facial features

## Results

Raw console logs are in the txt document.