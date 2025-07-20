# Setup Instructions for Pupillary Distance Measurement App

## Quick Setup for New Imports

When importing this project from GitHub to Replit, follow these steps:

### 1. Install Python Dependencies
Run the setup script to install required Python packages:
```bash
chmod +x setup_python_env.sh
./setup_python_env.sh
```

Or manually install packages:
```bash
uv add opencv-python mediapipe pupil-apriltags numpy
```

### 2. Required Python Packages
- **opencv-python**: Computer vision and image processing
- **mediapipe**: Facial landmark detection and pupil detection  
- **pupil-apriltags**: AprilTag detection for scale reference
- **numpy**: Numerical computing support

### 3. Start the Application
```bash
npm run dev
```

## Project Structure
- `client/`: React frontend with TypeScript
- `server/`: Express.js backend with TypeScript
- `server/python/`: Python scripts for computer vision processing
- `shared/`: Shared TypeScript schemas and types

## Features
- Image upload with drag-and-drop interface
- AprilTag detection for accurate scale reference
- MediaPipe-based pupil detection
- Precise pupillary distance measurement
- Real-time image processing with visual overlays

## Environment Requirements
- Node.js 20+
- Python 3.11+
- UV package manager (for Python dependencies)