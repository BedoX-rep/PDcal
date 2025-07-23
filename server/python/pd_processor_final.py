#!/usr/bin/env python3
"""
Final Pupillary Distance Measurement using proper AprilTag detection
"""

import cv2
import numpy as np
import json
import sys
import os
import argparse
import mediapipe as mp
from pupil_apriltags import Detector

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
            nose_bridge = (face_center_x, int(h * 0.35))  # Default nose bridge position
            return left_pupil, right_pupil, nose_bridge
        
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
        
        # Calculate nose bridge center using proper nasal landmarks
        # Use the nasal bridge landmark which is the center point between eyes
        NOSE_BRIDGE_INDICES = [6]  # Nasal bridge center - the key landmark between eyes
        nose_bridge = get_pupil_center(NOSE_BRIDGE_INDICES, face_landmarks, w, h)
        
        # If single landmark fails, use multiple nose bridge landmarks
        if nose_bridge is None:
            NOSE_BRIDGE_INDICES = [168, 8, 9, 10, 151]  # Multiple nose landmarks
            nose_bridge = get_pupil_center(NOSE_BRIDGE_INDICES, face_landmarks, w, h)
        
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
            if nose_bridge is None:
                nose_bridge = (face_center_x, int(h * 0.35))
        
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
            if nose_bridge is None:
                nose_bridge = (face_center_x, int(h * 0.35))
        
        # Ensure left pupil is to the left of right pupil
        if left_pupil[0] > right_pupil[0]:
            print("Swapping left/right pupils for correct orientation", file=sys.stderr)
            left_pupil, right_pupil = right_pupil, left_pupil
        
        print(f"MediaPipe detected pupils - Left: {left_pupil}, Right: {right_pupil}", file=sys.stderr)
        
        # If nose bridge detection failed, use fallback calculation
        if nose_bridge is None:
            # Calculate nose bridge as midpoint between inner eye corners
            if left_pupil and right_pupil:
                nose_bridge = ((left_pupil[0] + right_pupil[0]) // 2, 
                              min(left_pupil[1], right_pupil[1]) - 20)  # Slightly above pupil line
            else:
                nose_bridge = (w // 2, int(h * 0.35))  # Default nose bridge position
        
        print(f"Nose bridge center: {nose_bridge}", file=sys.stderr)
        return left_pupil, right_pupil, nose_bridge

def detect_apriltags(image):
    """Detect AprilTags using the proper pupil-apriltags library"""
    
    try:
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
            detector = None
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
                
                # If we found a good tag, stop searching
                if best_tag and best_confidence > 20.0:
                    break
                    
            except Exception as e:
                print(f"Error with {family}: {e}", file=sys.stderr)
                continue
            finally:
                # Always clean up detector to free memory
                if detector is not None:
                    try:
                        del detector
                    except:
                        pass
        
        return best_tag
        
    except Exception as e:
        print(f"AprilTag detection failed: {e}", file=sys.stderr)
        return None

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
        left_eye, right_eye, nose_bridge = detect_face_landmarks(image)
        
        # Debug: Print pupil coordinates
        print(f"Detected pupils - Left: {left_eye}, Right: {right_eye}", file=sys.stderr)
        
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
        
        # Calculate monocular PD distances (HORIZONTAL distance only from nose bridge center to each pupil)
        left_monocular_distance_pixels = abs(left_eye[0] - nose_bridge[0])  # Only horizontal distance
        right_monocular_distance_pixels = abs(right_eye[0] - nose_bridge[0])  # Only horizontal distance
        
        # Convert monocular distances to millimeters
        left_monocular_pd_mm = left_monocular_distance_pixels * pixel_scale_factor
        right_monocular_pd_mm = right_monocular_distance_pixels * pixel_scale_factor
        
        # STEP 4: Create processed image with accurate overlays
        processed_image = image.copy()
        
        # Ensure coordinates are integers
        left_eye = (int(left_eye[0]), int(left_eye[1]))
        right_eye = (int(right_eye[0]), int(right_eye[1]))
        nose_bridge = (int(nose_bridge[0]), int(nose_bridge[1]))
        
        # Draw small, thin pupil markers that don't obscure the pupils
        cv2.circle(processed_image, left_eye, 2, (0, 255, 0), 1)
        cv2.circle(processed_image, right_eye, 2, (0, 255, 0), 1)
        
        # Draw nose bridge center
        cv2.circle(processed_image, nose_bridge, 3, (255, 0, 255), 2)  # Magenta for nose bridge
        
        # Draw line between pupils with better visibility
        cv2.line(processed_image, left_eye, right_eye, (0, 255, 0), 3)
        
        # Draw monocular PD lines (horizontal only)
        cv2.line(processed_image, (nose_bridge[0], left_eye[1]), left_eye, (0, 255, 255), 2)  # Yellow for left monocular
        cv2.line(processed_image, (nose_bridge[0], right_eye[1]), right_eye, (0, 255, 255), 2)  # Yellow for right monocular
        
        # Draw red vertical line through nose bridge center
        image_height = processed_image.shape[0]
        cv2.line(processed_image, (nose_bridge[0], 0), (nose_bridge[0], image_height), (0, 0, 255), 2)  # Red vertical line
        
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
        cv2.putText(processed_image, f"Left monocular PD: {left_monocular_pd_mm:.1f}mm", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(processed_image, f"Right monocular PD: {right_monocular_pd_mm:.1f}mm", 
                   (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(processed_image, "AprilTag Detected", 
                   (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(processed_image, "Eyes Detected", 
                   (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(processed_image, f"Scale: {pixel_scale_factor:.3f}mm/px", 
                   (10, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
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
        nose_bridge_coords = (int(round(nose_bridge[0])), int(round(nose_bridge[1])))
        
        return {
            "success": True,
            "pd_value": round(pd_mm, 1),
            "left_pupil": {"x": left_eye_coords[0], "y": left_eye_coords[1]},
            "right_pupil": {"x": right_eye_coords[0], "y": right_eye_coords[1]},
            "nose_bridge": {"x": nose_bridge_coords[0], "y": nose_bridge_coords[1]},
            "left_monocular_pd": round(left_monocular_pd_mm, 1),
            "right_monocular_pd": round(right_monocular_pd_mm, 1),
            "pixel_distance": round(float(pixel_distance), 1),
            "scale_factor": round(pixel_scale_factor, 3),
            "processed_image_path": processed_path,
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

def process_image_framesize(image_path, left_line_x, right_line_x, frame_width_mm):
    """Process image using frame size analysis"""
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {"success": False, "error": "Could not load image"}
            
        print(f"Processing with frame size analysis: left={left_line_x}, right={right_line_x}, width={frame_width_mm}mm", file=sys.stderr)
        
        # Detect pupils using the same method as AprilTag analysis
        left_eye, right_eye, nose_bridge = detect_face_landmarks(image)
        
        if left_eye is None or right_eye is None:
            return {"success": False, "error": "Could not detect pupils"}
        
        print(f"Detected pupils - Left: {left_eye}, Right: {right_eye}", file=sys.stderr)
        
        # Calculate pixel distance between pupils
        pixel_distance = np.sqrt((right_eye[0] - left_eye[0])**2 + (right_eye[1] - left_eye[1])**2)
        
        # Calculate frame pixel width and scale factor
        frame_pixel_width = abs(right_line_x - left_line_x)
        pixel_scale_factor = frame_width_mm / frame_pixel_width  # mm per pixel
        
        print(f"Frame width: {frame_pixel_width} pixels = {frame_width_mm}mm", file=sys.stderr)
        print(f"Scale factor: {pixel_scale_factor:.3f} mm/pixel", file=sys.stderr)
        
        # Calculate PD in millimeters
        pd_mm = pixel_distance * pixel_scale_factor
        
        # Calculate monocular PDs using nose bridge as reference
        left_monocular_px = abs(left_eye[0] - nose_bridge[0])
        right_monocular_px = abs(right_eye[0] - nose_bridge[0])
        left_monocular_pd_mm = left_monocular_px * pixel_scale_factor
        right_monocular_pd_mm = right_monocular_px * pixel_scale_factor
        
        print(f"Measurements - PD: {pd_mm:.1f}mm, Left mono: {left_monocular_pd_mm:.1f}mm, Right mono: {right_monocular_pd_mm:.1f}mm", file=sys.stderr)
        
        # Create processed image with frame size annotations
        processed_image = image.copy()
        
        # Ensure coordinates are integers
        left_eye = (int(left_eye[0]), int(left_eye[1]))
        right_eye = (int(right_eye[0]), int(right_eye[1]))
        nose_bridge = (int(nose_bridge[0]), int(nose_bridge[1]))
        
        # Draw pupil markers
        cv2.circle(processed_image, left_eye, 2, (0, 255, 0), 1)
        cv2.circle(processed_image, right_eye, 2, (0, 255, 0), 1)
        cv2.circle(processed_image, nose_bridge, 3, (255, 0, 255), 2)
        
        # Draw line between pupils
        cv2.line(processed_image, left_eye, right_eye, (0, 255, 0), 3)
        
        # Draw monocular PD lines
        cv2.line(processed_image, (nose_bridge[0], left_eye[1]), left_eye, (0, 255, 255), 2)
        cv2.line(processed_image, (nose_bridge[0], right_eye[1]), right_eye, (0, 255, 255), 2)
        
        # Draw red vertical line through nose bridge center
        image_height = processed_image.shape[0]
        cv2.line(processed_image, (nose_bridge[0], 0), (nose_bridge[0], image_height), (0, 0, 255), 2)
        
        # Draw frame reference lines
        left_line_int = int(left_line_x)
        right_line_int = int(right_line_x)
        cv2.line(processed_image, (left_line_int, 0), (left_line_int, image_height), (255, 0, 0), 3)  # Blue for left frame edge
        cv2.line(processed_image, (right_line_int, 0), (right_line_int, image_height), (255, 0, 0), 3)  # Blue for right frame edge
        
        # Draw horizontal line between frame edges
        mid_y = image_height // 6
        cv2.line(processed_image, (left_line_int, mid_y), (right_line_int, mid_y), (255, 0, 0), 4)
        
        # Add text annotations
        cv2.putText(processed_image, f"PD: {pd_mm:.1f}mm", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(processed_image, f"Left monocular PD: {left_monocular_pd_mm:.1f}mm", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(processed_image, f"Right monocular PD: {right_monocular_pd_mm:.1f}mm", 
                   (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(processed_image, f"Frame Size Analysis ({frame_width_mm}mm)", 
                   (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(processed_image, "Eyes Detected", 
                   (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(processed_image, f"Scale: {pixel_scale_factor:.3f}mm/px", 
                   (10, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Save processed image
        processed_dir = "server/processed_images"
        os.makedirs(processed_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        processed_filename = f"{base_name}_processed.jpg"
        processed_path = os.path.join(processed_dir, processed_filename)
        
        cv2.imwrite(processed_path, processed_image)
        
        # Validate and ensure integer coordinates
        left_eye_coords = (int(round(left_eye[0])), int(round(left_eye[1])))
        right_eye_coords = (int(round(right_eye[0])), int(round(right_eye[1])))
        nose_bridge_coords = (int(round(nose_bridge[0])), int(round(nose_bridge[1])))
        
        return {
            "success": True,
            "pd_value": round(pd_mm, 1),
            "left_pupil": {"x": left_eye_coords[0], "y": left_eye_coords[1]},
            "right_pupil": {"x": right_eye_coords[0], "y": right_eye_coords[1]},
            "nose_bridge": {"x": nose_bridge_coords[0], "y": nose_bridge_coords[1]},
            "left_monocular_pd": round(left_monocular_pd_mm, 1),
            "right_monocular_pd": round(right_monocular_pd_mm, 1),
            "pixel_distance": round(float(pixel_distance), 1),
            "scale_factor": round(pixel_scale_factor, 3),
            "processed_image_path": processed_path,
            "apriltag_detected": False,
            "pupils_detected": True,
            "frame_analysis": True,
            "frame_width_mm": frame_width_mm,
            "frame_pixel_width": round(frame_pixel_width, 1)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Frame size processing error: {str(e)}",
            "apriltag_detected": False,
            "pupils_detected": False
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process image for pupillary distance measurement')
    parser.add_argument('image_path', help='Path to the image file')
    parser.add_argument('--mode', choices=['apriltag', 'framesize'], default='apriltag', 
                       help='Analysis mode: apriltag or framesize')
    parser.add_argument('--left-line', type=float, help='Left frame edge X coordinate (for framesize mode)')
    parser.add_argument('--right-line', type=float, help='Right frame edge X coordinate (for framesize mode)')
    parser.add_argument('--frame-width', type=float, help='Frame width in mm (for framesize mode)')
    
    args = parser.parse_args()
    
    print(f"Python script started with mode: {args.mode}", file=sys.stderr)
    
    if args.mode == 'framesize':
        if args.left_line is None or args.right_line is None or args.frame_width is None:
            print(json.dumps({"success": False, "error": "Frame size mode requires --left-line, --right-line, and --frame-width parameters"}))
            sys.exit(1)
        print(f"Using frame size analysis: left={args.left_line}, right={args.right_line}, width={args.frame_width}", file=sys.stderr)
        result = process_image_framesize(args.image_path, args.left_line, args.right_line, args.frame_width)
    else:
        print("Using AprilTag analysis", file=sys.stderr)
        result = process_image(args.image_path)
    
    print(json.dumps(result))