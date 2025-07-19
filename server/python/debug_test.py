#!/usr/bin/env python3
"""
Debug test for the PD processor
"""
import cv2
import numpy as np
import json
import sys

try:
    import mediapipe as mp
    from pupil_apriltags import Detector
    
    print("Libraries loaded successfully", file=sys.stderr)
    
    # Test with a simple image
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        image = cv2.imread(image_path)
        
        if image is None:
            print(json.dumps({"success": False, "error": "Could not read image"}))
            sys.exit(1)
            
        print(f"Image loaded: {image.shape}", file=sys.stderr)
        
        # Try MediaPipe face detection
        mp_face_mesh = mp.solutions.face_mesh
        
        # Resize if needed
        h, w = image.shape[:2]
        if max(h, w) > 1024:
            scale = 1024 / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h))
            
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        ) as face_mesh:
            results = face_mesh.process(rgb_image)
            
            if results.multi_face_landmarks:
                print("Face detected!", file=sys.stderr)
                
                # Test AprilTag detection
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                detector = Detector(families='tag25h9')
                tags = detector.detect(gray)
                
                print(f"Found {len(tags)} AprilTags", file=sys.stderr)
                
                if tags:
                    tag = tags[0]
                    print(f"Tag confidence: {tag.decision_margin}", file=sys.stderr)
                    
                    print(json.dumps({
                        "success": True,
                        "message": "Test successful - face and tag detected",
                        "tag_confidence": float(tag.decision_margin)
                    }))
                else:
                    print(json.dumps({
                        "success": False,
                        "error": "No AprilTag found",
                        "face_detected": True
                    }))
            else:
                print(json.dumps({
                    "success": False, 
                    "error": "No face detected"
                }))
    else:
        print(json.dumps({"success": False, "error": "No image path provided"}))
        
except Exception as e:
    print(json.dumps({"success": False, "error": f"Exception: {str(e)}"}))