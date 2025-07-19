#!/usr/bin/env python3
"""
Pupillary Distance Measurement using MediaPipe and AprilTag detection
"""

import cv2
import numpy as np
import mediapipe as mp
import json
import sys
import os
from pupil_apriltags import Detector

def process_image(image_path):
    """Process image to detect pupils and AprilTag, calculate PD"""
    try:
        # Initialize MediaPipe Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize AprilTag detector
        at_detector = Detector(families='tag36h11')
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return {"success": False, "error": "Could not read image file"}
        
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces and facial landmarks
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        ) as face_mesh:
            
            results = face_mesh.process(rgb_image)
            
            if not results.multi_face_landmarks:
                return {
                    "success": False, 
                    "error": "No face detected in the image",
                    "pupils_detected": False,
                    "apriltag_detected": False
                }
            
            # Get facial landmarks
            face_landmarks = results.multi_face_landmarks[0]
            h, w, c = image.shape
            
            # Extract pupil positions (approximate using eye landmarks)
            # Left eye center (from viewer's perspective - subject's right eye)
            left_eye_landmarks = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
            # Right eye center (from viewer's perspective - subject's left eye)
            right_eye_landmarks = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
            
            # Calculate eye centers
            left_eye_points = []
            right_eye_points = []
            
            for landmark in left_eye_landmarks:
                x = int(face_landmarks.landmark[landmark].x * w)
                y = int(face_landmarks.landmark[landmark].y * h)
                left_eye_points.append((x, y))
            
            for landmark in right_eye_landmarks:
                x = int(face_landmarks.landmark[landmark].x * w)
                y = int(face_landmarks.landmark[landmark].y * h)
                right_eye_points.append((x, y))
            
            # Calculate pupil centers (approximate as eye centers)
            left_pupil = np.mean(left_eye_points, axis=0).astype(int)
            right_pupil = np.mean(right_eye_points, axis=0).astype(int)
            
            # Detect AprilTags
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            tags = at_detector.detect(gray, estimate_tag_pose=False, camera_params=None, tag_size=None)
            
            if not tags:
                return {
                    "success": False,
                    "error": "No AprilTag detected in the image. Please ensure the AprilTag is visible and well-lit.",
                    "pupils_detected": True,
                    "apriltag_detected": False
                }
            
            # Use the first detected tag
            tag = tags[0]
            
            # Calculate AprilTag size in pixels (diagonal of the tag)
            tag_corners = tag.corners
            tag_width_pixels = np.linalg.norm(tag_corners[1] - tag_corners[0])
            tag_height_pixels = np.linalg.norm(tag_corners[2] - tag_corners[1])
            tag_size_pixels = (tag_width_pixels + tag_height_pixels) / 2  # Average width and height
            
            # Known AprilTag physical size (50mm)
            tag_physical_size_mm = 50.0
            
            # Calculate scale factor (mm per pixel)
            scale_factor = tag_physical_size_mm / tag_size_pixels
            
            # Calculate pupillary distance in pixels
            pixel_distance = np.linalg.norm(right_pupil - left_pupil)
            
            # Convert to millimeters
            pd_mm = pixel_distance * scale_factor
            
            # Create processed image with overlays
            processed_image = image.copy()
            
            # Draw pupil markers
            cv2.circle(processed_image, tuple(left_pupil), 5, (255, 0, 0), 2)  # Blue for left
            cv2.circle(processed_image, tuple(right_pupil), 5, (255, 0, 0), 2)  # Blue for right
            
            # Draw line between pupils
            cv2.line(processed_image, tuple(left_pupil), tuple(right_pupil), (255, 0, 0), 2)
            
            # Draw AprilTag outline
            tag_corners_int = tag.corners.astype(int)
            cv2.polylines(processed_image, [tag_corners_int], True, (0, 255, 255), 3)  # Yellow
            
            # Add text labels
            cv2.putText(processed_image, f"PD: {pd_mm:.1f}mm", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(processed_image, "Pupils Detected", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(processed_image, "AprilTag Found", 
                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Save processed image
            processed_dir = "server/processed_images"
            os.makedirs(processed_dir, exist_ok=True)
            
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            processed_filename = f"{base_name}_processed.jpg"
            processed_path = os.path.join(processed_dir, processed_filename)
            
            cv2.imwrite(processed_path, processed_image)
            
            return {
                "success": True,
                "pd_value": round(pd_mm, 1),
                "left_pupil": {"x": int(left_pupil[0]), "y": int(left_pupil[1])},
                "right_pupil": {"x": int(right_pupil[0]), "y": int(right_pupil[1])},
                "pixel_distance": round(float(pixel_distance), 1),
                "scale_factor": round(scale_factor, 3),
                "processed_image_path": processed_filename,
                "apriltag_detected": True,
                "pupils_detected": True
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"Processing error: {str(e)}",
            "apriltag_detected": False,
            "pupils_detected": False
        }

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(json.dumps({"success": False, "error": "Usage: python pd_processor.py <image_path>"}))
        sys.exit(1)
    
    image_path = sys.argv[1]
    result = process_image(image_path)
    print(json.dumps(result))
