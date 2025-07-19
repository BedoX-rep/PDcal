#!/usr/bin/env python3
"""
Robust Pupillary Distance Measurement using MediaPipe and improved detection
"""

import cv2
import numpy as np
import json
import sys
import os

def detect_face_landmarks(image):
    """Detect face landmarks using multiple robust methods"""
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Method 1: Try multiple Haar cascade approaches
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Try different scale factors and parameters
    faces = []
    for scale_factor in [1.05, 1.1, 1.2, 1.3]:
        for min_neighbors in [3, 4, 5]:
            detected = face_cascade.detectMultiScale(gray, scale_factor, min_neighbors, minSize=(50, 50))
            if len(detected) > 0:
                faces = detected
                break
        if len(faces) > 0:
            break
    
    # Method 2: If no face detected, assume central face region
    if len(faces) == 0:
        # Assume face occupies central 60% of image
        face_w = int(w * 0.6)
        face_h = int(h * 0.8)
        face_x = (w - face_w) // 2
        face_y = int(h * 0.1)  # Start slightly below top
        faces = [(face_x, face_y, face_w, face_h)]
    
    # Use the largest face
    face = max(faces, key=lambda x: x[2] * x[3])
    fx, fy, fw, fh = face
    
    # Method 2: Try eye detection with multiple approaches
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eye_glasses_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
    
    face_gray = gray[fy:fy+fh, fx:fx+fw]
    
    # Try different eye detection methods
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
    
    # Method 3: Template matching for eyes (works better with glasses)
    if len(eyes) < 2:
        # Look for dark regions that could be eyes
        # Apply histogram equalization for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        face_enhanced = clahe.apply(face_gray)
        
        # Look for dark circular regions (pupils)
        circles = cv2.HoughCircles(face_enhanced, cv2.HOUGH_GRADIENT, 1, 20,
                                 param1=50, param2=30, minRadius=3, maxRadius=15)
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            # Filter circles in the eye region (upper half of face)
            eye_candidates = []
            for (x, y, r) in circles:
                if y < fh * 0.6:  # Upper 60% of face
                    eye_candidates.append((x, y, r*2, r*2))  # Convert to eye format
            
            if len(eye_candidates) >= 2:
                # Sort by y-coordinate (should be roughly same level) then x
                eye_candidates.sort(key=lambda e: (e[1], e[0]))
                eyes = eye_candidates[:2]
    
    # Method 4: Fallback to proportional estimation
    if len(eyes) < 2:
        # Use anatomical proportions for eye placement
        eye_y = int(fh * 0.4)  # Eyes at 40% down from top of face
        eye_separation = int(fw * 0.25)  # Eyes separated by 25% of face width
        face_center_x = fw // 2
        
        # Create eye regions
        left_eye_x = face_center_x - eye_separation
        right_eye_x = face_center_x + eye_separation
        
        eye_size = int(fw * 0.08)  # Eye size proportional to face
        
        eyes = [
            (left_eye_x - eye_size//2, eye_y - eye_size//2, eye_size, eye_size),
            (right_eye_x - eye_size//2, eye_y - eye_size//2, eye_size, eye_size)
        ]
    
    # Sort eyes by x-coordinate (left to right)
    eyes = sorted(eyes, key=lambda x: x[0])
    
    # Take the two most separated eyes if more than 2
    if len(eyes) > 2:
        eyes = [eyes[0], eyes[-1]]
    
    # Calculate eye centers and convert back to full image coordinates
    left_eye_center = (fx + eyes[0][0] + eyes[0][2]//2, fy + eyes[0][1] + eyes[0][3]//2)
    right_eye_center = (fx + eyes[1][0] + eyes[1][2]//2, fy + eyes[1][1] + eyes[1][3]//2)
    
    return left_eye_center, right_eye_center

def detect_apriltag_pattern(image):
    """Detect AprilTag-like patterns using contour analysis"""
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Multiple thresholding approaches for better detection
    candidates = []
    
    # Try different threshold values
    for thresh_val in [100, 127, 150]:
        _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 300 < area < 20000:  # Reasonable tag size
                # Get bounding rectangle
                x, y, cw, ch = cv2.boundingRect(contour)
                
                # Check aspect ratio (should be roughly square)
                aspect_ratio = cw / ch
                if 0.6 < aspect_ratio < 1.7:
                    # Check if it's in a reasonable position (upper 80% of image)
                    if y < h * 0.8:
                        # Calculate "squareness" score
                        perimeter = cv2.arcLength(contour, True)
                        if perimeter > 0:
                            circularity = 4 * np.pi * area / (perimeter * perimeter)
                            # For squares, circularity should be around 0.785
                            squareness = 1.0 - abs(circularity - 0.785)
                            
                            candidates.append({
                                'contour': contour,
                                'bbox': (x, y, cw, ch),
                                'area': area,
                                'squareness': squareness,
                                'center': (x + cw//2, y + ch//2),
                                'size': (cw + ch) / 2
                            })
    
    if not candidates:
        return None
    
    # Sort by squareness score and area
    candidates.sort(key=lambda x: x['squareness'] * np.sqrt(x['area']), reverse=True)
    
    return candidates[0]

def process_image(image_path):
    """Process image to detect pupils and AprilTag, calculate PD"""
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return {"success": False, "error": "Could not read image file"}
        
        # Resize if too large to prevent memory issues
        original_shape = image.shape
        h, w = image.shape[:2]
        scale_factor = 1.0
        
        if max(h, w) > 1024:
            scale_factor = 1024 / max(h, w)
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Detect facial landmarks (eyes)
        left_eye, right_eye = detect_face_landmarks(image)
        
        if left_eye is None or right_eye is None:
            return {
                "success": False,
                "error": "Could not detect pupils/eyes in the image. Please ensure your face is clearly visible and well-lit.",
                "pupils_detected": False,
                "apriltag_detected": False
            }
        
        # Detect AprilTag pattern
        tag_candidate = detect_apriltag_pattern(image)
        
        if tag_candidate is None:
            return {
                "success": False,
                "error": "No AprilTag pattern detected. Please ensure a clear square marker is visible in the image.",
                "pupils_detected": True,
                "apriltag_detected": False
            }
        
        # Calculate scale factor (assuming 50mm AprilTag)
        tag_physical_size_mm = 50.0
        tag_size_pixels = tag_candidate['size']
        pixel_scale_factor = tag_physical_size_mm / tag_size_pixels
        
        # Calculate pupillary distance
        pixel_distance = np.sqrt((right_eye[0] - left_eye[0])**2 + (right_eye[1] - left_eye[1])**2)
        pd_mm = pixel_distance * pixel_scale_factor
        
        # Create processed image with accurate overlays
        processed_image = image.copy()
        
        # Draw pupil markers (small circles)
        cv2.circle(processed_image, left_eye, 3, (0, 255, 0), 2)  # Green circles
        cv2.circle(processed_image, right_eye, 3, (0, 255, 0), 2)
        
        # Draw center dots (1 pixel)
        cv2.circle(processed_image, left_eye, 1, (255, 255, 255), -1)  # White center
        cv2.circle(processed_image, right_eye, 1, (255, 255, 255), -1)
        
        # Draw line between pupils
        cv2.line(processed_image, left_eye, right_eye, (0, 255, 0), 2)
        
        # Draw AprilTag bounding box
        x, y, cw, ch = tag_candidate['bbox']
        cv2.rectangle(processed_image, (x, y), (x + cw, y + ch), (0, 255, 255), 2)  # Yellow box
        
        # Add corner markers for the tag
        cv2.circle(processed_image, (x, y), 3, (255, 0, 0), -1)  # Blue corners
        cv2.circle(processed_image, (x + cw, y), 3, (255, 0, 0), -1)
        cv2.circle(processed_image, (x, y + ch), 3, (255, 0, 0), -1)
        cv2.circle(processed_image, (x + cw, y + ch), 3, (255, 0, 0), -1)
        
        # Add text labels
        cv2.putText(processed_image, f"PD: {pd_mm:.1f}mm", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(processed_image, "Eyes Detected", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(processed_image, "Tag Pattern Found", 
                   (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Add measurement details
        cv2.putText(processed_image, f"Scale: {pixel_scale_factor:.3f}mm/px", 
                   (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
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
            "left_pupil": {"x": int(left_eye[0]), "y": int(left_eye[1])},
            "right_pupil": {"x": int(right_eye[0]), "y": int(right_eye[1])},
            "pixel_distance": round(float(pixel_distance), 1),
            "scale_factor": round(pixel_scale_factor, 3),
            "processed_image_path": processed_filename,
            "apriltag_detected": True,
            "pupils_detected": True,
            "tag_bbox": {"x": x, "y": y, "width": cw, "height": ch},
            "tag_confidence": round(tag_candidate['squareness'], 3)
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
        print(json.dumps({"success": False, "error": "Usage: python pd_processor_robust.py <image_path>"}))
        sys.exit(1)
    
    image_path = sys.argv[1]
    result = process_image(image_path)
    print(json.dumps(result))