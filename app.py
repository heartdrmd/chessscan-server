#!/usr/bin/env python3
"""
ChessScan Server v2 - OpenCV Perspective Warp
Returns perfectly warped 800x800 board image
iOS client just splits into squares and runs ML
"""

from flask import Flask, request, jsonify
import requests
import base64
import numpy as np
from PIL import Image
from io import BytesIO
import cv2
import os

app = Flask(__name__)

# Roboflow settings (fallback)
ROBOFLOW_API_KEY = os.environ.get('ROBOFLOW_API_KEY', 'qPVZQqumEL1m2BzsXbV0')
BOARD_MODEL = "chessboard-detection-x5kxd/1"


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'chessscan-v2'})


@app.route('/detect', methods=['POST'])
def detect_board():
    """
    Receives image, returns perfectly warped 800x800 board
    
    Returns JSON:
    {
        'success': bool,
        'board_image': base64 JPEG (800x800 perfect square),
        'confidence': float,
        'method': 'opencv' or 'roboflow'
    }
    """
    try:
        # Get image from request
        image = get_image_from_request()
        if image is None:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        original_size = image.size
        print(f"📸 Received image: {original_size}")
        
        # Convert to numpy/OpenCV format
        img_rgb = np.array(image.convert('RGB'))
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        # Try OpenCV chessboard corner detection first
        board_img, confidence, method = None, 0, None
        
        board_img, confidence = try_opencv_warp(img_rgb, gray)
        if board_img is not None:
            method = 'opencv'
            print(f"   ✅ OpenCV perspective warp successful")
        else:
            # Fallback to Roboflow
            print(f"   OpenCV failed, trying Roboflow...")
            board_img, confidence = try_roboflow_warp(image)
            if board_img is not None:
                method = 'roboflow'
                print(f"   ✅ Roboflow detection successful")
        
        if board_img is None:
            return jsonify({
                'success': False,
                'error': 'No chessboard detected',
                'board_detected': False
            })
        
        # Convert to base64
        board_base64 = image_to_base64(board_img)
        
        print(f"   ✅ Returning perfect 800x800 board via {method}")
        
        return jsonify({
            'success': True,
            'board_image': board_base64,
            'offset_x': 0,  # No offsets needed - board is perfect!
            'offset_y': 0,
            'confidence': confidence,
            'method': method,
            'board_detected': True
        })
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


def try_opencv_warp(img_rgb, gray):
    """
    Try to find chessboard corners with OpenCV and warp to perfect square
    Returns (PIL Image 800x800, confidence) or (None, 0)
    """
    # Try different pattern sizes
    for pattern in [(7, 7), (6, 6), (5, 5)]:
        ret, corners = cv2.findChessboardCorners(
            gray, pattern,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        )
        
        if ret:
            print(f"   Found {pattern[0]}x{pattern[1]} chessboard corners")
            
            # Refine corners for sub-pixel accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Reshape to grid
            rows, cols = pattern
            corners = corners.reshape(rows, cols, 2)
            
            # Calculate square size from corner spacing
            square_w = (corners[0, cols-1] - corners[0, 0])[0] / (cols - 1)
            square_h = (corners[rows-1, 0] - corners[0, 0])[1] / (rows - 1)
            
            # Extrapolate to outer board corners
            # Inner corners are 1 square inside, so we need to go 1 square out
            extra_cols = (8 - cols) / 2 + 1  # squares to add on each side
            extra_rows = (8 - rows) / 2 + 1
            
            top_left = corners[0, 0] - [square_w * extra_cols, square_h * extra_rows]
            top_right = corners[0, cols-1] + [square_w * extra_cols, -square_h * extra_rows]
            bottom_left = corners[rows-1, 0] - [square_w * extra_cols, -square_h * extra_rows]
            bottom_right = corners[rows-1, cols-1] + [square_w * extra_cols, square_h * extra_rows]
            
            # Source points (detected corners)
            src_pts = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
            
            # Destination points (perfect 800x800 square)
            dst_pts = np.array([[0, 0], [800, 0], [800, 800], [0, 800]], dtype=np.float32)
            
            # Get perspective transform matrix
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            
            # Apply warp
            warped = cv2.warpPerspective(img_rgb, M, (800, 800))
            
            # Convert to PIL
            result = Image.fromarray(warped)
            
            # Confidence based on pattern size (7x7 = best)
            confidence = 0.95 if pattern == (7, 7) else 0.90 if pattern == (6, 6) else 0.85
            
            return result, confidence
    
    return None, 0


def try_roboflow_warp(image):
    """
    Fallback: Use Roboflow for detection, then apply tilt correction
    Returns (PIL Image 800x800, confidence) or (None, 0)
    """
    # Resize for API
    image = resize_image(image, max_dim=1500)
    
    # Call Roboflow
    detection = detect_board_roboflow(image)
    if detection is None:
        return None, 0
    
    # Crop board region
    cx, cy = detection['x'], detection['y']
    w, h = detection['width'], detection['height']
    
    size = max(w, h) * 1.05
    x1 = int(cx - size / 2)
    y1 = int(cy - size / 2)
    x2 = int(cx + size / 2)
    y2 = int(cy + size / 2)
    
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.size[0], x2)
    y2 = min(image.size[1], y2)
    
    cropped = image.crop((x1, y1, x2, y2))
    cropped = cropped.resize((800, 800), Image.LANCZOS)
    
    # Apply tilt correction
    tilt = detect_tilt_angle(cropped)
    if abs(tilt) > 0.3:
        print(f"   Tilt correction: {tilt:.1f}°")
        cropped = cropped.rotate(tilt, expand=True, resample=Image.BICUBIC)
        cropped = center_crop_square(cropped)
        cropped = cropped.resize((800, 800), Image.LANCZOS)
    
    return cropped, detection['confidence']


def detect_board_roboflow(image):
    """Call Roboflow API to detect chessboard"""
    img_str = image_to_base64(image, quality=85)
    
    url = f"https://detect.roboflow.com/{BOARD_MODEL}?api_key={ROBOFLOW_API_KEY}"
    response = requests.post(
        url,
        data=img_str,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30
    )
    
    result = response.json()
    
    if 'predictions' not in result or len(result['predictions']) == 0:
        return None
    
    pred = result['predictions'][0]
    return {
        'x': pred['x'],
        'y': pred['y'],
        'width': pred['width'],
        'height': pred['height'],
        'confidence': pred['confidence']
    }


def detect_tilt_angle(image):
    """Detect board tilt using Hough lines"""
    gray = np.array(image.convert('L'))
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(bw, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=150)
    
    if lines is None:
        return 0
    
    angles = []
    for line in lines:
        theta = line[0][1]
        angle = np.degrees(theta) - 90
        while angle > 45:
            angle -= 90
        while angle < -45:
            angle += 90
        angles.append(angle)
    
    return np.median(angles) if angles else 0


def center_crop_square(image):
    """Crop the largest centered square"""
    w, h = image.size
    size = min(w, h)
    left = (w - size) // 2
    top = (h - size) // 2
    return image.crop((left, top, left + size, top + size))


def get_image_from_request():
    """Extract image from request"""
    if 'image' in request.files:
        file = request.files['image']
        return Image.open(file.stream)
    
    if request.is_json:
        data = request.get_json()
        if 'image' in data:
            img_data = base64.b64decode(data['image'])
            return Image.open(BytesIO(img_data))
    
    return None


def resize_image(image, max_dim=1500):
    """Resize image if larger than max_dim"""
    if max(image.size) <= max_dim:
        return image
    ratio = max_dim / max(image.size)
    new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
    return image.resize(new_size, Image.LANCZOS)


def image_to_base64(image, quality=90):
    """Convert PIL Image to base64 JPEG"""
    buffered = BytesIO()
    image_rgb = image.convert("RGB")
    image_rgb.save(buffered, quality=quality, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("ascii")


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 ChessScan server v2 starting on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)
