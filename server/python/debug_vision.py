#!/usr/bin/env python3
"""
Debug vision detection
"""
import cv2
import numpy as np
import json
import sys

def debug_detection(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return {"error": "Could not read image"}
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Test face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    print(f"Image size: {w}x{h}", file=sys.stderr)
    print(f"Found {len(faces)} faces", file=sys.stderr)
    
    if len(faces) > 0:
        for i, (x, y, fw, fh) in enumerate(faces):
            print(f"Face {i}: {x},{y} {fw}x{fh}", file=sys.stderr)
            
            # Try to detect eyes in this face
            face_gray = gray[y:y+fh, x:x+fw]
            
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            eyes = eye_cascade.detectMultiScale(face_gray, 1.1, 3, minSize=(10, 10))
            print(f"  Eyes found: {len(eyes)}", file=sys.stderr)
            
            # Try circles (pupils)
            circles = cv2.HoughCircles(face_gray, cv2.HOUGH_GRADIENT, 1, 20,
                                     param1=50, param2=30, minRadius=3, maxRadius=20)
            circle_count = len(circles[0]) if circles is not None else 0
            print(f"  Circles found: {circle_count}", file=sys.stderr)
            
            # For the largest face, use proportional approach
            if i == 0:  # Use first/largest face
                eye_y = y + int(fh * 0.4)
                eye_separation = int(fw * 0.25)
                face_center_x = x + fw // 2
                
                left_eye = (face_center_x - eye_separation, eye_y)
                right_eye = (face_center_x + eye_separation, eye_y)
                
                return {
                    "success": True,
                    "method": "proportional",
                    "left_eye": left_eye,
                    "right_eye": right_eye,
                    "face_bbox": [x, y, fw, fh]
                }
    
    return {"success": False, "error": "No face detected"}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        result = debug_detection(sys.argv[1])
        print(json.dumps(result))
    else:
        print(json.dumps({"error": "No image path"}))