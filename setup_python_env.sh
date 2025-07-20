#!/bin/bash
# Python Environment Setup for Pupillary Distance Measurement Application
# This script ensures all required Python packages are installed for computer vision processing

echo "Setting up Python environment for PD measurement app..."

# Install required Python packages for computer vision
echo "Installing Python packages..."
uv add opencv-python mediapipe pupil-apriltags numpy

echo "Python environment setup complete!"
echo "Required packages:"
echo "  - opencv-python: Computer vision and image processing"
echo "  - mediapipe: Facial landmark detection and pupil detection"
echo "  - pupil-apriltags: AprilTag detection for scale reference"
echo "  - numpy: Numerical computing support"