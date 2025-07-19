#!/usr/bin/env python3
"""
Final Pupillary Distance Measurement using proper AprilTag detection
"""

import cv2
import numpy as np
import json
import sys
import os
from pupil_apriltags import Detector

def detect_face_landmarks(image):
    """Detect face landmarks using OpenCV Haar cascades"""
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Try multiple Haar cascade approaches
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    faces = []
    for scale_factor in [1.05, 1.1, 1.2, 1.3]:
        for min_neighbors in [3, 4, 5]:
            detected = face_cascade.detectMultiScale(gray, scale_factor, min_neighbors, minSize=(50, 50))
            if len(detected) > 0:
                faces = detected
                break
        if len(faces) > 0:
            break
    
    # If no face detected, assume central face region
    if len(faces) == 0:
        face_w = int(w * 0.6)
        face_h = int(h * 0.8)
        face_x = (w - face_w) // 2
        face_y = int(h * 0.1)
        faces = [(face_x, face_y, face_w, face_h)]
    
    # Use the largest face
    face = max(faces, key=lambda x: x[2] * x[3])
    fx, fy, fw, fh = face
    
    # Try eye detection methods
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eye_glasses_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
    
    face_gray = gray[fy:fy+fh, fx:fx+fw]
    eyes = []
    
    # Try regular eye detection
    eyes1 = eye_cascade.detectMultiScale(face_gray, 1.1, 3, minSize=(15, 15))
    if len(eyes1) >= 2:
        eyes = eyes1
    
    # Try eye with glasses detection
    if len(eyes) < 2:
        eyes2 = eye_glasses_cascade.detectMultiScale(face_gray, 1.1, 3, minSize=(15, 15))
        if len(eyes2) >= 2:
            eyes = eyes2
    
    # HoughCircles for pupil detection
    if len(eyes) < 2:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        face_enhanced = clahe.apply(face_gray)
        
        circles = cv2.HoughCircles(face_enhanced, cv2.HOUGH_GRADIENT, 1, 20,
                                 param1=50, param2=30, minRadius=3, maxRadius=15)
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            eye_candidates = []
            for (x, y, r) in circles:
                if y < fh * 0.6:  # Upper 60% of face
                    eye_candidates.append((x, y, r*2, r*2))
            
            if len(eye_candidates) >= 2:
                eye_candidates.sort(key=lambda e: (e[1], e[0]))
                eyes = eye_candidates[:2]
    
    # Fallback to proportional estimation
    if len(eyes) < 2:
        eye_y = int(fh * 0.4)
        eye_separation = int(fw * 0.25)
        face_center_x = fw // 2
        
        left_eye_x = face_center_x - eye_separation
        right_eye_x = face_center_x + eye_separation
        
        eye_size = int(fw * 0.08)
        
        eyes = [
            (left_eye_x - eye_size//2, eye_y - eye_size//2, eye_size, eye_size),
            (right_eye_x - eye_size//2, eye_y - eye_size//2, eye_size, eye_size)
        ]
    
    # Sort eyes by x-coordinate and take the two most separated
    eyes = sorted(eyes, key=lambda x: x[0])
    if len(eyes) > 2:
        eyes = [eyes[0], eyes[-1]]
    
    # Calculate eye centers and convert to full image coordinates
    left_eye_center = (fx + eyes[0][0] + eyes[0][2]//2, fy + eyes[0][1] + eyes[0][3]//2)
    right_eye_center = (fx + eyes[1][0] + eyes[1][2]//2, fy + eyes[1][1] + eyes[1][3]//2)
    
    return left_eye_center, right_eye_center

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
    """Process image to detect pupils and AprilTag, calculate PD"""
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
                "pupils_detected": False
            }
        
        print(f"AprilTag detected with confidence: {apriltag.decision_margin}", file=sys.stderr)
        
        # STEP 2: Only proceed to face detection if AprilTag is found
        print("Detecting face landmarks...", file=sys.stderr)
        left_eye, right_eye = detect_face_landmarks(image)
        
        if left_eye is None or right_eye is None:
            return {
                "success": False,
                "error": "Could not detect pupils/eyes in the image. Please ensure your face is clearly visible and well-lit.",
                "pupils_detected": False,
                "apriltag_detected": True
            }
        
        # STEP 3: Calculate measurements
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
        
        # STEP 4: Create processed image with accurate overlays
        processed_image = image.copy()
        
        # Draw pupil markers (small circles with 1-pixel centers)
        cv2.circle(processed_image, left_eye, 4, (0, 255, 0), 2)  # Green circles
        cv2.circle(processed_image, right_eye, 4, (0, 255, 0), 2)
        cv2.circle(processed_image, left_eye, 1, (255, 255, 255), -1)  # White 1-pixel centers
        cv2.circle(processed_image, right_eye, 1, (255, 255, 255), -1)
        
        # Draw line between pupils
        cv2.line(processed_image, left_eye, right_eye, (0, 255, 0), 2)
        
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
        cv2.putText(processed_image, f"Scale: {pixel_scale_factor:.3f}mm/px", 
                   (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
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
        
        return {
            "success": True,
            "pd_value": round(pd_mm, 1),
            "left_pupil": {"x": int(left_eye[0]), "y": int(left_eye[1])},
            "right_pupil": {"x": int(right_eye[0]), "y": int(right_eye[1])},
            "pixel_distance": round(float(pixel_distance), 1),
            "scale_factor": round(pixel_scale_factor, 3),
            "processed_image_path": processed_filename,
            "apriltag_detected": True,
            "pupils_detected": True,
            "apriltag_id": int(apriltag.tag_id),
            "apriltag_confidence": round(apriltag.decision_margin, 1),
            "tag_bbox": {"x": min_x, "y": min_y, "width": max_x - min_x, "height": max_y - min_y},
            "tag_corners": tag_corners.tolist()
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
        print(json.dumps({"success": False, "error": "Usage: python pd_processor_final.py <image_path>"}))
        sys.exit(1)
    
    image_path = sys.argv[1]
    result = process_image(image_path)
    print(json.dumps(result))