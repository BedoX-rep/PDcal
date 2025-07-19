
#!/usr/bin/env python3
"""
Final Pupillary Distance Measurement
"""

import cv2
import numpy as np
import json
import sys
import os
import mediapipe as mp
from pupil_apriltags import Detector

def detect_glasses_and_lenses(image):
    """Placeholder function - glasses detection removed"""
    return {
        'glasses_detected': False,
        'confidence': 0.0,
        'frame_bbox': None,
        'lens_areas': []
    }

def detect_face_landmarks(image):
    """Detect face landmarks and pupils using MediaPipe Face Mesh"""
    
    # Convert BGR to RGB for MediaPipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = rgb_image.shape[:2]
    
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    
    print("Using MediaPipe for pupil detection...", file=sys.stderr)
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        
        results = face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            print("No face detected by MediaPipe, using fallback method", file=sys.stderr)
            # Fallback to center estimation
            face_center_x = w // 2
            eye_y = int(h * 0.4)  # Eyes typically at 40% down from top
            eye_separation = int(w * 0.12)  # Approximate eye separation
            
            left_pupil = (face_center_x - eye_separation, eye_y)
            right_pupil = (face_center_x + eye_separation, eye_y)
            return left_pupil, right_pupil
        
        # Get the first (and likely only) face
        face_landmarks = results.multi_face_landmarks[0]
        
        # MediaPipe landmark indices for eye centers
        # Left eye center landmarks (viewer's left)
        LEFT_EYE_INDICES = [468, 469, 470, 471, 472]  # Left iris landmarks
        # Right eye center landmarks (viewer's right)  
        RIGHT_EYE_INDICES = [473, 474, 475, 476, 477]  # Right iris landmarks
        
        # Calculate pupil centers using iris landmarks
        def get_pupil_center(eye_indices, landmarks, image_width, image_height):
            eye_points = []
            for idx in eye_indices:
                if idx < len(landmarks.landmark):
                    landmark = landmarks.landmark[idx]
                    x = int(landmark.x * image_width)
                    y = int(landmark.y * image_height)
                    eye_points.append((x, y))
            
            if eye_points:
                # Calculate centroid of iris landmarks
                avg_x = sum(p[0] for p in eye_points) / len(eye_points)
                avg_y = sum(p[1] for p in eye_points) / len(eye_points)
                return (int(avg_x), int(avg_y))
            return None
        
        # Get pupil centers
        left_pupil = get_pupil_center(LEFT_EYE_INDICES, face_landmarks, w, h)
        right_pupil = get_pupil_center(RIGHT_EYE_INDICES, face_landmarks, w, h)
        
        # Fallback to eye corner landmarks if iris detection fails
        if left_pupil is None or right_pupil is None:
            print("Iris landmarks not available, using eye corner landmarks", file=sys.stderr)
            
            # Eye corner landmarks
            LEFT_EYE_CORNERS = [33, 7, 163, 144, 145, 153, 154, 155, 133]  # Left eye outline
            RIGHT_EYE_CORNERS = [362, 382, 381, 380, 374, 373, 390, 249, 263]  # Right eye outline
            
            left_pupil = get_pupil_center(LEFT_EYE_CORNERS, face_landmarks, w, h)
            right_pupil = get_pupil_center(RIGHT_EYE_CORNERS, face_landmarks, w, h)
        
        # Final fallback if MediaPipe fails
        if left_pupil is None or right_pupil is None:
            print("MediaPipe landmark detection failed, using geometric fallback", file=sys.stderr)
            face_center_x = w // 2
            eye_y = int(h * 0.4)
            eye_separation = int(w * 0.12)
            
            left_pupil = (face_center_x - eye_separation, eye_y)
            right_pupil = (face_center_x + eye_separation, eye_y)
        
        # Ensure pupils are within image bounds
        left_pupil = (max(0, min(w-1, left_pupil[0])), max(0, min(h-1, left_pupil[1])))
        right_pupil = (max(0, min(w-1, right_pupil[0])), max(0, min(h-1, right_pupil[1])))
        
        # Validate pupil distance is reasonable
        pixel_distance = np.sqrt((right_pupil[0] - left_pupil[0])**2 + (right_pupil[1] - left_pupil[1])**2)
        
        # If pupils are too close or too far apart, use proportional estimation
        if pixel_distance < 20 or pixel_distance > w * 0.8:
            print("MediaPipe pupil distance seems unreasonable, using proportional fallback", file=sys.stderr)
            face_center_x = w // 2
            eye_y = int(h * 0.4)
            eye_separation = int(w * 0.12)
            
            left_pupil = (face_center_x - eye_separation, eye_y)
            right_pupil = (face_center_x + eye_separation, eye_y)
        
        # Ensure left pupil is to the left of right pupil
        if left_pupil[0] > right_pupil[0]:
            print("Swapping left/right pupils for correct orientation", file=sys.stderr)
            left_pupil, right_pupil = right_pupil, left_pupil
        
        print(f"MediaPipe detected pupils - Left: {left_pupil}, Right: {right_pupil}", file=sys.stderr)
        return left_pupil, right_pupil

def detect_apriltags(image):
    """Detect AprilTags using the proper pupil-apriltags library"""
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Process in smaller resolution to avoid memory issues
    h, w = gray.shape
    if max(h, w) > 512:
        scale = 512 / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        gray_small = cv2.resize(gray, (new_w, new_h))
        scale_back = max(h, w) / 512
    else:
        gray_small = gray
        scale_back = 1.0
    
    # Try AprilTag families one by one to reduce memory usage
    tag_families = ['tag25h9', 'tag36h11', 'tag16h5']  # Most common families first
    
    best_tag = None
    best_confidence = 0
    
    for family in tag_families:
        try:
            # Create detector for this family only
            detector = Detector(families=family)
            tags = detector.detect(gray_small, estimate_tag_pose=False, camera_params=None, tag_size=None)
            
            for tag in tags:
                # Scale coordinates back if we resized
                if scale_back != 1.0:
                    tag.corners = tag.corners * scale_back
                    tag.center = tag.center * scale_back
                
                # Accept tags with reasonable confidence (lowered threshold)
                if tag.decision_margin > best_confidence and tag.decision_margin > 5.0:
                    best_tag = tag
                    best_confidence = tag.decision_margin
                    print(f"Found {family} tag with confidence {tag.decision_margin}", file=sys.stderr)
            
            # Clean up detector to free memory
            del detector
            
            # If we found a good tag, stop searching
            if best_tag and best_confidence > 20.0:
                break
                
        except Exception as e:
            print(f"Error with {family}: {e}", file=sys.stderr)
            continue
    
    return best_tag

def process_image(image_path):
    """Process image to detect pupils, AprilTag, and glasses, calculate PD"""
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return {"success": False, "error": "Could not read image file"}
        
        # Resize if too large
        h, w = image.shape[:2]
        scale_factor = 1.0
        
        if max(h, w) > 1024:
            scale_factor = 1024 / max(h, w)
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # STEP 1: Detect AprilTags FIRST - this is mandatory
        print("Detecting AprilTags...", file=sys.stderr)
        apriltag = detect_apriltags(image)
        
        if apriltag is None:
            return {
                "success": False,
                "error": "No AprilTag detected. A valid AprilTag is required for accurate PD measurement. Please ensure an AprilTag is clearly visible in the image.",
                "apriltag_detected": False,
                "pupils_detected": False,
                "glasses_detected": False
            }
        
        print(f"AprilTag detected with confidence: {apriltag.decision_margin}", file=sys.stderr)
        
        # STEP 2: Skip glasses detection (removed)
        glasses_info = {'glasses_detected': False, 'confidence': 0.0, 'frame_bbox': None, 'lens_areas': []}
        
        # STEP 3: Detect face landmarks and pupils
        print("Detecting face landmarks...", file=sys.stderr)
        left_eye, right_eye = detect_face_landmarks(image)
        
        # Debug: Print pupil coordinates
        print(f"Detected pupils - Left: {left_eye}, Right: {right_eye}", file=sys.stderr)
        
        if left_eye is None or right_eye is None:
            return {
                "success": False,
                "error": "Could not detect pupils/eyes in the image. Please ensure your face is clearly visible and well-lit.",
                "pupils_detected": False,
                "apriltag_detected": True,
                "glasses_detected": glasses_info['glasses_detected']
            }
        
        # STEP 4: Calculate measurements
        print("Calculating PD...", file=sys.stderr)
        
        # Get AprilTag corners and calculate size
        tag_corners = apriltag.corners
        tag_width_pixels = np.linalg.norm(tag_corners[1] - tag_corners[0])
        tag_height_pixels = np.linalg.norm(tag_corners[2] - tag_corners[1])
        tag_size_pixels = (tag_width_pixels + tag_height_pixels) / 2
        
        # Known AprilTag physical size (50mm)
        tag_physical_size_mm = 50.0
        pixel_scale_factor = tag_physical_size_mm / tag_size_pixels
        
        # Calculate pupillary distance
        pixel_distance = np.sqrt((right_eye[0] - left_eye[0])**2 + (right_eye[1] - left_eye[1])**2)
        pd_mm = pixel_distance * pixel_scale_factor
        
        # STEP 5: Create processed image with accurate overlays
        processed_image = image.copy()
        
        # Ensure coordinates are integers
        left_eye = (int(left_eye[0]), int(left_eye[1]))
        right_eye = (int(right_eye[0]), int(right_eye[1]))
        
        # Glasses detection removed - no overlay needed
        
        # Draw pupil markers with better visibility
        # Draw outer circle (larger, green)
        cv2.circle(processed_image, left_eye, 6, (0, 255, 0), 2)
        cv2.circle(processed_image, right_eye, 6, (0, 255, 0), 2)
        
        # Draw inner circle (smaller, red for better contrast)
        cv2.circle(processed_image, left_eye, 3, (0, 0, 255), -1)
        cv2.circle(processed_image, right_eye, 3, (0, 0, 255), -1)
        
        # Draw precise center pixel (white dot)
        cv2.circle(processed_image, left_eye, 1, (255, 255, 255), -1)
        cv2.circle(processed_image, right_eye, 1, (255, 255, 255), -1)
        
        # Draw line between pupils with better visibility
        cv2.line(processed_image, left_eye, right_eye, (0, 255, 0), 3)
        
        # Draw AprilTag outline using actual detected corners
        tag_corners_int = tag_corners.astype(int)
        cv2.polylines(processed_image, [tag_corners_int], True, (0, 255, 255), 3)  # Yellow outline
        
        # Add corner markers
        for corner in tag_corners_int:
            cv2.circle(processed_image, tuple(corner), 3, (255, 0, 0), -1)  # Blue corners
        
        # Add AprilTag ID and confidence
        tag_center = np.mean(tag_corners, axis=0).astype(int)
        cv2.putText(processed_image, f"Tag ID: {apriltag.tag_id}", 
                   (tag_center[0] - 30, tag_center[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(processed_image, f"Conf: {apriltag.decision_margin:.1f}", 
                   (tag_center[0] - 30, tag_center[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add measurement text
        cv2.putText(processed_image, f"PD: {pd_mm:.1f}mm", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(processed_image, "AprilTag Detected", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(processed_image, "Eyes Detected", 
                   (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add glasses detection status
        if glasses_info['glasses_detected']:
            cv2.putText(processed_image, "Glasses Detected", 
                       (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 0, 128), 2)
        else:
            cv2.putText(processed_image, "No Glasses", 
                       (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
        
        cv2.putText(processed_image, f"Scale: {pixel_scale_factor:.3f}mm/px", 
                   (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Save processed image
        processed_dir = "server/processed_images"
        os.makedirs(processed_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        processed_filename = f"{base_name}_processed.jpg"
        processed_path = os.path.join(processed_dir, processed_filename)
        
        cv2.imwrite(processed_path, processed_image)
        
        # Calculate bounding box for the tag
        x_coords = tag_corners[:, 0]
        y_coords = tag_corners[:, 1]
        min_x, max_x = int(np.min(x_coords)), int(np.max(x_coords))
        min_y, max_y = int(np.min(y_coords)), int(np.max(y_coords))
        
        # Validate and ensure integer coordinates
        left_eye_coords = (int(round(left_eye[0])), int(round(left_eye[1])))
        right_eye_coords = (int(round(right_eye[0])), int(round(right_eye[1])))
        
        return {
            "success": True,
            "pd_value": round(pd_mm, 1),
            "left_pupil": {"x": left_eye_coords[0], "y": left_eye_coords[1]},
            "right_pupil": {"x": right_eye_coords[0], "y": right_eye_coords[1]},
            "pixel_distance": round(float(pixel_distance), 1),
            "scale_factor": round(pixel_scale_factor, 3),
            "processed_image_path": processed_filename,
            "apriltag_detected": True,
            "pupils_detected": True,
            "apriltag_id": int(apriltag.tag_id),
            "apriltag_confidence": round(apriltag.decision_margin, 1),
            "tag_bbox": {"x": min_x, "y": min_y, "width": max_x - min_x, "height": max_y - min_y},
            "tag_corners": tag_corners.tolist(),
            "glasses_detected": glasses_info['glasses_detected'],
            "glasses_confidence": round(glasses_info['confidence'], 2),
            "glasses_frame_bbox": glasses_info['frame_bbox'],
            "lens_areas": glasses_info['lens_areas']
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Processing error: {str(e)}",
            "apriltag_detected": False,
            "pupils_detected": False,
            "glasses_detected": False
        }

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(json.dumps({"success": False, "error": "Usage: python pd_processor_final.py <image_path>"}))
        sys.exit(1)
    
    image_path = sys.argv[1]
    result = process_image(image_path)
    print(json.dumps(result))
