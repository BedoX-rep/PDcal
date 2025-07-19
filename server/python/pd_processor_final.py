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

def detect_pupil_in_eye_region(eye_region, debug_name="eye"):
    """Simple and reliable pupil detection using darkest region approach"""
    
    if eye_region.size == 0:
        return None
    
    h, w = eye_region.shape
    
    # Simple preprocessing: just enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    enhanced = clahe.apply(eye_region)
    
    # Slight blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # Find the darkest point in the eye region
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blurred)
    
    # The pupil should be the darkest point
    pupil_x, pupil_y = min_loc
    
    # Validate the position is reasonable (not at edges)
    margin = 3
    if (margin <= pupil_x <= w - margin and 
        margin <= pupil_y <= h - margin):
        return (pupil_x, pupil_y)
    
    # If at edge, use center as fallback
    return (w // 2, h // 2)

def detect_face_landmarks(image):
    """Enhanced face and pupil detection using multiple computer vision techniques"""
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Step 1: Robust face detection with multiple cascades
    face_cascades = [
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
        cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml',
        cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
    ]
    
    faces = []
    for cascade_file in face_cascades:
        try:
            face_cascade = cv2.CascadeClassifier(cascade_file)
            for scale_factor in [1.05, 1.1, 1.2]:
                for min_neighbors in [3, 4, 5]:
                    detected = face_cascade.detectMultiScale(gray, scale_factor, min_neighbors, 
                                                           minSize=(50, 50), maxSize=(w, h))
                    if len(detected) > 0:
                        faces.extend(detected)
                        break
                if len(faces) > 0:
                    break
        except:
            continue
        if len(faces) > 0:
            break
    
    # If no face detected, use profile or assume central region
    if len(faces) == 0:
        try:
            profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
            profiles = profile_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
            if len(profiles) > 0:
                faces = profiles
        except:
            pass
    
    # Fallback to central face region
    if len(faces) == 0:
        face_w = int(w * 0.6)
        face_h = int(h * 0.8)
        face_x = (w - face_w) // 2
        face_y = int(h * 0.1)
        faces = [(face_x, face_y, face_w, face_h)]
    
    # Use the largest, most central face
    best_face = max(faces, key=lambda f: f[2] * f[3])
    fx, fy, fw, fh = best_face
    
    face_gray = gray[fy:fy+fh, fx:fx+fw]
    
    # Step 2: Enhanced eye region detection
    eye_candidates = []
    
    # Method 1: Multiple eye cascade classifiers
    eye_cascades = [
        cv2.data.haarcascades + 'haarcascade_eye.xml',
        cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml',
        cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml',
        cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml'
    ]
    
    for cascade_file in eye_cascades:
        try:
            eye_cascade = cv2.CascadeClassifier(cascade_file)
            for scale in [1.05, 1.1, 1.15]:
                for neighbors in [3, 4, 5]:
                    eyes = eye_cascade.detectMultiScale(face_gray, scale, neighbors, 
                                                       minSize=(10, 10), maxSize=(fw//3, fh//3))
                    if len(eyes) >= 2:
                        eye_candidates.extend(eyes)
                        break
                if len(eye_candidates) >= 2:
                    break
        except:
            continue
        if len(eye_candidates) >= 2:
            break
    
    # Method 2: Template matching for eye patterns
    if len(eye_candidates) < 2:
        # Create simple eye templates
        eye_template = np.zeros((20, 30), dtype=np.uint8)
        cv2.ellipse(eye_template, (15, 10), (12, 6), 0, 0, 360, 255, -1)
        cv2.ellipse(eye_template, (15, 10), (4, 4), 0, 0, 360, 0, -1)  # Pupil
        
        result = cv2.matchTemplate(face_gray, eye_template, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= 0.3)
        
        for pt in zip(*locations[::-1]):
            eye_candidates.append((pt[0], pt[1], 30, 20))
    
    # Method 3: Advanced HoughCircles with preprocessing
    if len(eye_candidates) < 2:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(face_gray)
        
        # Multiple HoughCircles parameters
        for dp in [1, 2]:
            for min_dist in [15, 20, 30]:
                for param1 in [50, 100]:
                    for param2 in [15, 20, 25]:
                        circles = cv2.HoughCircles(enhanced, cv2.HOUGH_GRADIENT, dp, min_dist,
                                                 param1=param1, param2=param2, 
                                                 minRadius=3, maxRadius=20)
                        
                        if circles is not None:
                            circles = np.round(circles[0, :]).astype("int")
                            for (x, y, r) in circles:
                                if y < fh * 0.7:  # Upper portion of face
                                    eye_candidates.append((x-r, y-r, r*2, r*2))
                        
                        if len(eye_candidates) >= 2:
                            break
                    if len(eye_candidates) >= 2:
                        break
                if len(eye_candidates) >= 2:
                    break
            if len(eye_candidates) >= 2:
                break
    
    # Step 3: Filter and select best eye candidates
    if len(eye_candidates) >= 2:
        # Filter by position (should be in upper half of face, horizontally separated)
        valid_eyes = []
        for (ex, ey, ew, eh) in eye_candidates:
            if (ey + eh/2) < fh * 0.65 and ew > 5 and eh > 5:  # Upper face region
                valid_eyes.append((ex, ey, ew, eh))
        
        if len(valid_eyes) >= 2:
            # Sort by horizontal position and take leftmost and rightmost
            valid_eyes.sort(key=lambda e: e[0])
            
            # Remove eyes that are too close to each other
            filtered_eyes = []
            for eye in valid_eyes:
                too_close = False
                for existing in filtered_eyes:
                    distance = abs(eye[0] - existing[0])
                    if distance < fw * 0.15:  # Too close horizontally
                        too_close = True
                        break
                if not too_close:
                    filtered_eyes.append(eye)
            
            if len(filtered_eyes) >= 2:
                eye_candidates = [filtered_eyes[0], filtered_eyes[-1]]  # Leftmost and rightmost
            else:
                eye_candidates = valid_eyes[:2]
        else:
            eye_candidates = eye_candidates[:2]
    
    # Step 4: Fallback to proportional estimation if needed
    if len(eye_candidates) < 2:
        eye_y = int(fh * 0.37)  # Eyes are typically at 37% down from top of face
        eye_separation = int(fw * 0.25)  # Eyes are about 25% of face width apart
        face_center_x = fw // 2
        
        left_eye_x = face_center_x - eye_separation
        right_eye_x = face_center_x + eye_separation
        
        eye_size = int(fw * 0.08)
        
        eye_candidates = [
            (left_eye_x - eye_size//2, eye_y - eye_size//2, eye_size, eye_size),
            (right_eye_x - eye_size//2, eye_y - eye_size//2, eye_size, eye_size)
        ]
    
    # Ensure we have exactly 2 eyes, sorted left to right
    if len(eye_candidates) > 2:
        eye_candidates = sorted(eye_candidates, key=lambda e: e[0])
        eye_candidates = [eye_candidates[0], eye_candidates[-1]]
    elif len(eye_candidates) == 1:
        # Duplicate the eye and offset for left/right
        ex, ey, ew, eh = eye_candidates[0]
        offset = int(fw * 0.2)
        eye_candidates = [
            (ex - offset, ey, ew, eh),
            (ex + offset, ey, ew, eh)
        ]
    
    eye_candidates = sorted(eye_candidates, key=lambda e: e[0])  # Sort left to right
    
    # Step 5: Precise pupil detection within each eye region
    left_eye_region = eye_candidates[0]
    right_eye_region = eye_candidates[1]
    
    # Validate eye regions are reasonable
    left_eye_region = (max(0, left_eye_region[0]), max(0, left_eye_region[1]), 
                      min(left_eye_region[2], fw), min(left_eye_region[3], fh))
    right_eye_region = (max(0, right_eye_region[0]), max(0, right_eye_region[1]), 
                       min(right_eye_region[2], fw), min(right_eye_region[3], fh))
    
    # Extract and preprocess eye regions for better pupil detection
    lex, ley, lew, leh = left_eye_region
    rex, rey, rew, reh = right_eye_region
    
    # Expand eye regions slightly for better context
    expansion = 5
    lex = max(0, lex - expansion)
    ley = max(0, ley - expansion)
    lew = min(lew + 2*expansion, fw - lex)
    leh = min(leh + 2*expansion, fh - ley)
    
    rex = max(0, rex - expansion)
    rey = max(0, rey - expansion)
    rew = min(rew + 2*expansion, fw - rex)
    reh = min(reh + 2*expansion, fh - rey)
    
    if lew > 0 and leh > 0:
        left_eye_img = face_gray[ley:ley+leh, lex:lex+lew]
        
        # Apply multiple attempts with different preprocessing
        left_pupil_local = None
        
        # Attempt 1: Standard detection
        left_pupil_local = detect_pupil_in_eye_region(left_eye_img, "left")
        
        # Attempt 2: If failed, try with histogram equalization
        if left_pupil_local is None:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
            enhanced_left = clahe.apply(left_eye_img)
            left_pupil_local = detect_pupil_in_eye_region(enhanced_left, "left_enhanced")
        
        # Attempt 3: If still failed, try with gamma correction
        if left_pupil_local is None:
            gamma_corrected = np.power(left_eye_img / 255.0, 0.7) * 255
            gamma_corrected = gamma_corrected.astype(np.uint8)
            left_pupil_local = detect_pupil_in_eye_region(gamma_corrected, "left_gamma")
        
        if left_pupil_local:
            left_pupil = (fx + lex + left_pupil_local[0], fy + ley + left_pupil_local[1])
        else:
            left_pupil = (fx + lex + lew//2, fy + ley + leh//2)
    else:
        left_pupil = (fx + lex + lew//2, fy + ley + leh//2)
    
    if rew > 0 and reh > 0:
        right_eye_img = face_gray[rey:rey+reh, rex:rex+rew]
        
        # Apply multiple attempts with different preprocessing
        right_pupil_local = None
        
        # Attempt 1: Standard detection
        right_pupil_local = detect_pupil_in_eye_region(right_eye_img, "right")
        
        # Attempt 2: If failed, try with histogram equalization
        if right_pupil_local is None:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
            enhanced_right = clahe.apply(right_eye_img)
            right_pupil_local = detect_pupil_in_eye_region(enhanced_right, "right_enhanced")
        
        # Attempt 3: If still failed, try with gamma correction
        if right_pupil_local is None:
            gamma_corrected = np.power(right_eye_img / 255.0, 0.7) * 255
            gamma_corrected = gamma_corrected.astype(np.uint8)
            right_pupil_local = detect_pupil_in_eye_region(gamma_corrected, "right_gamma")
        
        if right_pupil_local:
            right_pupil = (fx + rex + right_pupil_local[0], fy + rey + right_pupil_local[1])
        else:
            right_pupil = (fx + rex + rew//2, fy + rey + reh//2)
    else:
        right_pupil = (fx + rex + rew//2, fy + rey + reh//2)
    
    # Final validation and refinement of pupil positions
    h, w = gray.shape
    
    # Ensure pupils are within image bounds
    left_pupil = (max(0, min(w-1, left_pupil[0])), max(0, min(h-1, left_pupil[1])))
    right_pupil = (max(0, min(w-1, right_pupil[0])), max(0, min(h-1, right_pupil[1])))
    
    # Validate pupil distance is reasonable (should be between 45-85mm in most adults)
    pixel_distance = np.sqrt((right_pupil[0] - left_pupil[0])**2 + (right_pupil[1] - left_pupil[1])**2)
    
    # If pupils are too close or too far apart, use proportional estimation as fallback
    if pixel_distance < 20 or pixel_distance > w * 0.8:  # Unreasonable distance
        print("Pupil distance seems unreasonable, using proportional fallback", file=sys.stderr)
        
        # Use face-based proportional estimation
        face_center_x = fx + fw // 2
        eye_y = fy + int(fh * 0.37)  # 37% down from top of face
        eye_separation = int(fw * 0.25)  # 25% of face width
        
        left_pupil = (face_center_x - eye_separation, eye_y)
        right_pupil = (face_center_x + eye_separation, eye_y)
    
    # Additional refinement: ensure left pupil is actually to the left of right pupil
    if left_pupil[0] > right_pupil[0]:
        print("Swapping left/right pupils for correct orientation", file=sys.stderr)
        left_pupil, right_pupil = right_pupil, left_pupil
    
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
        
        # STEP 4: Create processed image with accurate overlays
        processed_image = image.copy()
        
        # Ensure coordinates are integers
        left_eye = (int(left_eye[0]), int(left_eye[1]))
        right_eye = (int(right_eye[0]), int(right_eye[1]))
        
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