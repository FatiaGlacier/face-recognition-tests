# Test 4: Pilot head pose reaction test

## Overview

This is a pilot test created to validate the head pose processing pipeline and the model's behavior before implementing the final version in Test 5.

The goal of this stage was to verify that the algorithms work correctly, generate the required intermediate data, and establish the workflow for future experiments.

## Current status

- This is an experimental implementation.
- The code is functional but not yet refactored.
- The implementation prioritizes validating the pipeline over clean architecture.
- The improved and refactored version will be developed in Test 5.

## Results

The first successful results are stored like graphs images and selected images data in results_me.json.

This file contains:

- selected best frames,
- estimated pitch, yaw, and roll angles,

This graphs contains:

- facenet512's response to a change in the angles of each axis