#!/usr/bin/env python3
"""
ChessScan Server - Board Detection Only
Returns cropped 800x800 board image + calibration offsets
iOS/Mac client does piece classification locally
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

# Roboflow settings
ROBOFLOW_API_KEY = os.environ.get('ROBOFLOW_API_KEY', 'qPVZQqumEL1m2BzsXbV0')
BOARD_MODEL = "chessboard-detection-x5kxd/1"


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'chessscan'})


@app.route('/detect', methods=['POST'])
def detect_board():
    """
    Receives image, returns cropped board + calibration
    
    Returns JSON:
    {
        'success': bool,
        'board_image': base64 JPEG (800x800),
        'offset_x': int,
        'offset_y': int,
        'confidence': float
    }
    """
    try:
        # Get image from request
        image = get_image_from_request()
        if image is None:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        original_size = image.size
        print(f"📸 Received image: {original_size}")
        
        # Step 1: Resize if needed
        image = resize_image(image, max_dim=1500)
        print(f"   Resized to: {image.size}")
        
        # Step 2: Detect board with Roboflow
        detection = detect_board_roboflow(image)
        if detection is None:
            return jsonify({
                'success': False, 
                'error': 'No chessboard detected',
                'board_detected': False
            })
        
        print(f"   Board detected: conf={detection['confidence']:.2%}")
        
        # Step 3: Crop to board (800x800)
        board_img = crop_board(image, detection)
        if board_img is None:
            return jsonify({'success': False, 'error': 'Failed to crop board'})
        
        # Step 4: Calibrate grid alignment
        offset_x, offset_y = calibrate_grid(board_img)
        print(f"   Calibration offset: ({offset_x}, {offset_y})")
        
        # Step 5: Convert board to base64
        board_base64 = image_to_base64(board_img)
        
        print(f"   ✅ Returning 800x800 board")
        
        return jsonify({
            'success': True,
            'board_image': board_base64,
            'offset_x': offset_x,
            'offset_y': offset_y,
            'confidence': detection['confidence'],
            'board_detected': True
        })
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


def get_image_from_request():
    """Extract image from various request formats"""
    # Multipart form upload
    if 'image' in request.files:
        file = request.files['image']
        return Image.open(file.stream)
    
    # Base64 JSON
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
    """Convert PIL Image to base64 JPEG string"""
    buffered = BytesIO()
    image_rgb = image.convert("RGB")
    image_rgb.save(buffered, quality=quality, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("ascii")


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


def crop_board(image, detection):
    """Crop, rotate 90° CCW, correct tilt, and resize board to 800x800"""
    cx, cy = detection['x'], detection['y']
    w, h = detection['width'], detection['height']
    
    # Make square using smaller dimension
    size = min(w, h)
    x1 = int(cx - size / 2)
    y1 = int(cy - size / 2)
    x2 = int(cx + size / 2)
    y2 = int(cy + size / 2)
    
    # Ensure within bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.size[0], x2)
    y2 = min(image.size[1], y2)
    
    # Crop
    cropped = image.crop((x1, y1, x2, y2))
    
    # Rotate 90° counter-clockwise (to fix sideways board)
    rotated = cropped.rotate(90, expand=True)
    print(f"   Rotated 90° CCW")
    
    # Resize to 800x800 for tilt detection
    resized = rotated.resize((800, 800), Image.LANCZOS)
    
    # Detect and correct tilt
    tilt_angle = detect_tilt(resized)
    if abs(tilt_angle) > 0.5:  # Only correct if tilt > 0.5 degrees
        print(f"   Tilt detected: {tilt_angle:.2f}°, correcting...")
        resized = resized.rotate(tilt_angle, expand=False, fillcolor=(128, 128, 128))
        # Re-crop to 800x800 after rotation
        resized = center_crop(resized, 800)
    
    return resized.resize((800, 800), Image.LANCZOS)


def detect_tilt(image):
    """
    Detect board tilt angle using Hough line detection
    Converts to B/W, finds strongest horizontal lines, calculates angle
    Returns angle in degrees (positive = clockwise tilt)
    """
    # Convert to grayscale numpy array
    gray = np.array(image.convert('L'))
    
    # High contrast: threshold to black/white
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Edge detection
    edges = cv2.Canny(bw, 50, 150, apertureSize=3)
    
    # Hough line detection
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=200)
    
    if lines is None or len(lines) == 0:
        print(f"   No lines detected for tilt correction")
        return 0
    
    # Collect angles of near-horizontal lines (within 20° of horizontal)
    horizontal_angles = []
    for line in lines:
        rho, theta = line[0]
        angle_deg = np.degrees(theta) - 90  # Convert to degrees from horizontal
        
        # Only consider near-horizontal lines
        if abs(angle_deg) < 20:
            horizontal_angles.append(angle_deg)
    
    if not horizontal_angles:
        print(f"   No horizontal lines found")
        return 0
    
    # Return median angle
    tilt = np.median(horizontal_angles)
    print(f"   Found {len(horizontal_angles)} horizontal lines, median tilt: {tilt:.2f}°")
    return tilt


def center_crop(image, size):
    """Crop center square from image"""
    w, h = image.size
    left = (w - size) // 2
    top = (h - size) // 2
    right = left + size
    bottom = top + size
    return image.crop((left, top, right, bottom))


def calibrate_grid(board_img):
    """
    Find optimal grid offset using multiple algorithms
    Returns (offset_x, offset_y) in pixels for 800x800 board
    """
    gray = np.array(board_img.convert('L'))
    
    offsets = []
    
    # Algorithm 1: Edge-based
    try:
        ox, oy = calibrate_edges(gray)
        offsets.append((ox, oy))
    except:
        pass
    
    # Algorithm 2: Color variance
    try:
        ox, oy = calibrate_color_variance(gray)
        offsets.append((ox, oy))
    except:
        pass
    
    # Algorithm 3: Corner pattern
    try:
        ox, oy = calibrate_corners(gray)
        offsets.append((ox, oy))
    except:
        pass
    
    if not offsets:
        return 0, 0
    
    # Take median of all results
    xs = [o[0] for o in offsets]
    ys = [o[1] for o in offsets]
    return int(np.median(xs)), int(np.median(ys))


def calibrate_edges(gray):
    """Edge-based calibration using Sobel"""
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    col_sums = np.sum(np.abs(sobelx), axis=0)
    row_sums = np.sum(np.abs(sobely), axis=1)
    
    best_ox, best_oy = 0, 0
    best_score = 0
    
    for ox in range(-50, 50, 2):
        for oy in range(-50, 50, 2):
            score = 0
            for i in range(9):
                x = i * 100 + ox
                y = i * 100 + oy
                if 0 <= x < 800:
                    score += col_sums[x]
                if 0 <= y < 800:
                    score += row_sums[y]
            if score > best_score:
                best_score = score
                best_ox, best_oy = ox, oy
    
    return best_ox, best_oy


def calibrate_color_variance(gray):
    """Find offset that minimizes variance within each square"""
    best_ox, best_oy = 0, 0
    best_score = float('inf')
    
    for ox in range(-40, 40, 4):
        for oy in range(-40, 40, 4):
            total_variance = 0
            for row in range(8):
                for col in range(8):
                    x = col * 100 + ox + 10
                    y = row * 100 + oy + 10
                    x2 = x + 80
                    y2 = y + 80
                    
                    if x >= 0 and y >= 0 and x2 < 800 and y2 < 800:
                        square = gray[y:y2, x:x2]
                        total_variance += np.std(square)
            
            if total_variance < best_score:
                best_score = total_variance
                best_ox, best_oy = ox, oy
    
    return best_ox, best_oy


def calibrate_corners(gray):
    """Check that corners match expected light/dark pattern"""
    best_ox, best_oy = 0, 0
    best_score = float('inf')
    
    for ox in range(-40, 40, 4):
        for oy in range(-40, 40, 4):
            score = 0
            for row in range(8):
                for col in range(8):
                    x = col * 100 + ox + 50
                    y = row * 100 + oy + 50
                    
                    if 0 <= x < 800 and 0 <= y < 800:
                        is_light_square = (row + col) % 2 == 0
                        pixel_val = gray[y, x]
                        
                        if is_light_square:
                            score += (255 - pixel_val) ** 2
                        else:
                            score += (pixel_val - 80) ** 2
            
            if score < best_score:
                best_score = score
                best_ox, best_oy = ox, oy
    
    return best_ox, best_oy


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 ChessScan server starting on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)
