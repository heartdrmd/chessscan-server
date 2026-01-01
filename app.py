#!/usr/bin/env python3
"""
ChessScan Server v7 - With iOS ML Board Detector Support
=========================================================

MODE 1: Local ML Pipeline (NEW!)
  - iOS runs CB.mlmodel to detect board bounding box
  - iOS crops image and sends cropped board
  - Server: CV corner detection → 9x9 grid → extract 64 squares
  - Returns: cropped board, grid overlay, stitched, all squares

MODE 2: Roboflow Pipeline (existing)
  - Server calls Roboflow API for board detection
  - CV corner detection → extract squares

MODE 3: Both Pipelines
  - Run both and return results for iOS to pick

USAGE:
  POST /detect?mode=1&precropped=true  → iOS ML (board already cropped)
  POST /detect?mode=1                   → Color+CV (legacy)
  POST /detect?mode=2                   → Roboflow
  POST /detect?mode=3                   → Both

OUTPUT:
  - squares: dict of 64 base64 images (a8→h1)
  - stitched_image: 2400x2400 combined board
  - board_image: cropped/straightened board
  - grid_overlay_image: 9x9 grid drawn on board (NEW!)
  - confidence, method, corners_found
"""

from flask import Flask, request, jsonify
import requests
import base64
import numpy as np
from PIL import Image, ImageDraw
from io import BytesIO
import cv2
import os

app = Flask(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

ROBOFLOW_API_KEY = os.environ.get('ROBOFLOW_API_KEY', 'qPVZQqumEL1m2BzsXbV0')
BOARD_MODEL = "chessboard-detection-x5kxd/1"

SQUARE_SIZE = 300
BOARD_SIZE = SQUARE_SIZE * 8  # 2400x2400

FILES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
RANKS = ['8', '7', '6', '5', '4', '3', '2', '1']

# Color ranges for board detection (HSV)
TAN_LOW = np.array([10, 20, 100])
TAN_HIGH = np.array([40, 150, 255])
BROWN_LOW = np.array([5, 30, 50])
BROWN_HIGH = np.array([35, 200, 220])
GREEN_LOW = np.array([35, 40, 80])
GREEN_HIGH = np.array([85, 255, 220])
LIGHT_LOW = np.array([30, 10, 180])
LIGHT_HIGH = np.array([90, 80, 255])


# =============================================================================
# ROUTES
# =============================================================================

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'chessscan-v7-ios-ml'})


@app.route('/detect', methods=['POST'])
def detect_board():
    """Main endpoint - handles all modes"""
    try:
        mode = request.args.get('mode', '1')
        mode = int(mode) if mode in ['1', '2', '3'] else 1
        precropped = request.args.get('precropped', 'false').lower() == 'true'
        
        image = get_image_from_request()
        if image is None:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        print(f"📸 Received image: {image.size}, Mode: {mode}, Precropped: {precropped}")
        
        # Mode 1 with precropped = iOS ML pipeline
        if mode == 1 and precropped:
            result = precropped_cv_pipeline(image)
            if result is None:
                return jsonify({
                    'success': False,
                    'error': 'CV corner detection failed on cropped board',
                    'board_detected': False
                })
            return format_response_with_grid(result, 'ios_ml')
        
        # Mode 1 without precropped = Color+CV pipeline (legacy)
        elif mode == 1:
            result = color_cv_pipeline(image)
            if result is None:
                return jsonify({
                    'success': False,
                    'error': 'No chessboard detected (Color+CV)',
                    'board_detected': False
                })
            return format_response_with_grid(result, 'color_cv')
        
        # Mode 2 = Roboflow
        elif mode == 2:
            result = roboflow_pipeline(image)
            if result is None:
                print("  ⚠️ Roboflow failed, falling back to Color+CV...")
                result = color_cv_pipeline(image)
                if result is None:
                    return jsonify({
                        'success': False,
                        'error': 'No chessboard detected',
                        'board_detected': False
                    })
                return format_response_with_grid(result, 'fallback_color_cv')
            return format_response_with_grid(result, 'roboflow')
        
        # Mode 3 = Both
        elif mode == 3:
            color_result = color_cv_pipeline(image)
            robo_result = roboflow_pipeline(image)
            return format_both_response(color_result, robo_result)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


def format_response_with_grid(result, method_name):
    """Format response with grid overlay image"""
    squares, stitched_img, board_img, grid_overlay_img, confidence, method, corners_found = result
    
    squares_base64 = {label: image_to_base64(img, quality=90) for label, img in squares.items()}
    
    response = {
        'success': True,
        'method': method_name,
        'squares': squares_base64,
        'stitched_image': image_to_base64(stitched_img, quality=90),
        'board_image': image_to_base64(board_img, quality=90),
        'confidence': confidence,
        'corners_found': corners_found,
        'board_detected': True
    }
    
    # Add grid overlay if available
    if grid_overlay_img is not None:
        response['grid_overlay_image'] = image_to_base64(grid_overlay_img, quality=90)
    
    print(f"✅ Returning 64 squares via {method_name} ({corners_found})")
    return jsonify(response)


def format_both_response(color_result, robo_result):
    """Format response for both pipelines"""
    response = {
        'success': True,
        'method': 'both',
        'board_detected': True
    }
    
    if color_result:
        squares, stitched_img, board_img, grid_overlay_img, confidence, method, corners_found = color_result
        squares_base64 = {label: image_to_base64(img, quality=90) for label, img in squares.items()}
        response['color_cv'] = {
            'squares': squares_base64,
            'stitched_image': image_to_base64(stitched_img, quality=90),
            'board_image': image_to_base64(board_img, quality=90),
            'grid_overlay_image': image_to_base64(grid_overlay_img, quality=90) if grid_overlay_img else None,
            'confidence': confidence,
            'corners_found': corners_found
        }
    else:
        response['color_cv'] = None
    
    if robo_result:
        squares, stitched_img, board_img, grid_overlay_img, confidence, method, corners_found = robo_result
        squares_base64 = {label: image_to_base64(img, quality=90) for label, img in squares.items()}
        response['roboflow'] = {
            'squares': squares_base64,
            'stitched_image': image_to_base64(stitched_img, quality=90),
            'board_image': image_to_base64(board_img, quality=90),
            'grid_overlay_image': image_to_base64(grid_overlay_img, quality=90) if grid_overlay_img else None,
            'confidence': confidence,
            'corners_found': corners_found
        }
    else:
        response['roboflow'] = None
    
    return jsonify(response)


# =============================================================================
# PIPELINE: PRECROPPED (iOS ML Board Detector)
# =============================================================================

def precropped_cv_pipeline(image):
    """
    Process a PRE-CROPPED board image from iOS ML detector.
    
    iOS has already:
    1. Run CB.mlmodel to detect board bounding box
    2. Cropped the image to just the board
    
    Server does:
    1. CV corner detection (findChessboardCorners)
    2. Extrapolate to 9x9 grid
    3. Extract 64 squares
    4. Return with debug images
    
    Returns: (squares, stitched, board_img, grid_overlay, confidence, method, corners_str) or None
    """
    print("  [PRECROPPED CV PIPELINE - iOS ML]")
    
    img_rgb = np.array(image.convert('RGB'))
    img_h, img_w = img_rgb.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    print(f"    Received cropped board: {img_w}x{img_h}")
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Try multiple methods to find CV corners
    methods = [
        ("clahe", enhanced),
        ("gray", gray),
        ("blur5", cv2.GaussianBlur(gray, (5, 5), 0)),
        ("blur7", cv2.GaussianBlur(gray, (7, 7), 0)),
        ("adaptive", cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),
    ]
    
    best_corners = None
    best_pattern = None
    best_method = None
    
    for method_name, processed in methods:
        for pattern in [(7, 7), (6, 6), (5, 5), (4, 4), (3, 3)]:
            ret, corners = cv2.findChessboardCorners(
                processed, pattern,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            if ret:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                print(f"    ✅ {method_name}: found {pattern}")
                best_corners = refined
                best_pattern = pattern
                best_method = method_name
                break
        if best_corners is not None:
            break
    
    # If no CV corners found, divide by 8
    if best_corners is None:
        print("    ❌ No CV corners found, dividing by 8...")
        squares, stitched, grid_overlay = extract_squares_divide8_with_grid(img_rgb)
        return squares, stitched, image, grid_overlay, 0.7, "divide8", "divide8"
    
    # Extrapolate to 9x9 and extract squares
    squares, stitched, grid_overlay = extract_with_grid_overlay(
        img_rgb, gray, best_corners, best_pattern, img_w, img_h
    )
    
    corners_str = f"{best_pattern[0]}x{best_pattern[1]}"
    print(f"  ✅ Precropped pipeline complete! ({corners_str})")
    
    return squares, stitched, image, grid_overlay, 0.9, best_method, corners_str


def extract_with_grid_overlay(img_rgb, gray, corners, pattern, img_w, img_h):
    """
    Extract 64 squares AND create grid overlay image for debugging.
    """
    rows, cols = pattern
    corners_grid = corners.reshape(rows, cols, 2)
    
    # Calculate step sizes
    h_steps = [corners_grid[i, j+1] - corners_grid[i, j] for i in range(rows) for j in range(cols - 1)]
    v_steps = [corners_grid[i+1, j] - corners_grid[i, j] for i in range(rows - 1) for j in range(cols)]
    avg_h = np.median(h_steps, axis=0)
    avg_v = np.median(v_steps, axis=0)
    step_h = np.linalg.norm(avg_h)
    step_v = np.linalg.norm(avg_v)
    
    # Find bounds
    det_min = corners_grid.min(axis=(0, 1))
    det_max = corners_grid.max(axis=(0, 1))
    
    # Room to expand
    margin = 2
    steps_left = int((det_min[0] - margin) / step_h) if step_h > 0 else 0
    steps_right = int((img_w - margin - det_max[0]) / step_h) if step_h > 0 else 0
    steps_up = int((det_min[1] - margin) / step_v) if step_v > 0 else 0
    steps_down = int((img_h - margin - det_max[1]) / step_v) if step_v > 0 else 0
    
    detected_sq = rows - 1
    need_total = 8
    
    add_left = min(steps_left, need_total - detected_sq)
    add_right = min(steps_right, need_total - detected_sq - add_left)
    add_up = min(steps_up, need_total - detected_sq)
    add_down = min(steps_down, need_total - detected_sq - add_up)
    
    # Ensure we get 8 squares
    total_h = detected_sq + add_left + add_right
    total_v = detected_sq + add_up + add_down
    
    # If not enough, center what we have
    if total_h < 8:
        add_left = (8 - detected_sq) // 2
        add_right = 8 - detected_sq - add_left
        total_h = 8
    if total_v < 8:
        add_up = (8 - detected_sq) // 2
        add_down = 8 - detected_sq - add_up
        total_v = 8
    
    print(f"    Expand: L={add_left}, R={add_right}, U={add_up}, D={add_down} → {total_h}x{total_v}")
    
    # Build full 9x9 grid
    tl_detected = corners_grid[0, 0]
    tl_full = tl_detected - add_left * avg_h - add_up * avg_v
    
    grid_rows = total_v + 1
    grid_cols = total_h + 1
    full_grid = np.zeros((grid_rows, grid_cols, 2), dtype=np.float32)
    
    for i in range(grid_rows):
        for j in range(grid_cols):
            full_grid[i, j] = tl_full + j * avg_h + i * avg_v
    
    # Create grid overlay image
    grid_overlay = img_rgb.copy()
    
    # Draw grid lines
    for i in range(grid_rows):
        for j in range(grid_cols - 1):
            pt1 = tuple(full_grid[i, j].astype(int))
            pt2 = tuple(full_grid[i, j + 1].astype(int))
            cv2.line(grid_overlay, pt1, pt2, (255, 0, 0), 2)
    for i in range(grid_rows - 1):
        for j in range(grid_cols):
            pt1 = tuple(full_grid[i, j].astype(int))
            pt2 = tuple(full_grid[i + 1, j].astype(int))
            cv2.line(grid_overlay, pt1, pt2, (255, 0, 0), 2)
    
    # Draw corner points
    for i in range(grid_rows):
        for j in range(grid_cols):
            pt = tuple(full_grid[i, j].astype(int))
            cv2.circle(grid_overlay, pt, 5, (0, 255, 0), -1)
    
    # Mark detected corners differently
    for i in range(rows):
        for j in range(cols):
            pt = tuple(corners_grid[i, j].astype(int))
            cv2.circle(grid_overlay, pt, 8, (255, 0, 255), 2)
    
    grid_overlay_pil = Image.fromarray(grid_overlay)
    
    # Extract squares
    squares, stitched = extract_squares_from_grid(img_rgb, full_grid, min(total_v, 8), min(total_h, 8))
    
    return squares, stitched, grid_overlay_pil


def extract_squares_divide8_with_grid(img_rgb):
    """Divide board by 8 and create grid overlay"""
    img_h, img_w = img_rgb.shape[:2]
    
    step_h = img_w / 8
    step_v = img_h / 8
    
    # Build 9x9 grid
    full_grid = np.zeros((9, 9, 2), dtype=np.float32)
    for i in range(9):
        for j in range(9):
            full_grid[i, j] = [j * step_h, i * step_v]
    
    # Create grid overlay
    grid_overlay = img_rgb.copy()
    for i in range(9):
        for j in range(8):
            pt1 = tuple(full_grid[i, j].astype(int))
            pt2 = tuple(full_grid[i, j + 1].astype(int))
            cv2.line(grid_overlay, pt1, pt2, (255, 0, 0), 2)
    for i in range(8):
        for j in range(9):
            pt1 = tuple(full_grid[i, j].astype(int))
            pt2 = tuple(full_grid[i + 1, j].astype(int))
            cv2.line(grid_overlay, pt1, pt2, (255, 0, 0), 2)
    
    for i in range(9):
        for j in range(9):
            pt = tuple(full_grid[i, j].astype(int))
            cv2.circle(grid_overlay, pt, 5, (0, 255, 0), -1)
    
    grid_overlay_pil = Image.fromarray(grid_overlay)
    
    # Extract squares
    squares, stitched = extract_squares_from_grid(img_rgb, full_grid, 8, 8)
    
    return squares, stitched, grid_overlay_pil


def extract_squares_from_grid(img_rgb, full_grid, num_rows, num_cols):
    """Extract squares using perspective transform for each cell"""
    squares = {}
    stitched = np.zeros((BOARD_SIZE, BOARD_SIZE, 3), dtype=np.uint8)
    
    for row in range(min(num_rows, 8)):
        for col in range(min(num_cols, 8)):
            tl = full_grid[row, col]
            tr = full_grid[row, col + 1]
            bl = full_grid[row + 1, col]
            br = full_grid[row + 1, col + 1]
            
            src_pts = np.array([tl, tr, br, bl], dtype=np.float32)
            dst_pts = np.array([
                [0, 0], [SQUARE_SIZE, 0],
                [SQUARE_SIZE, SQUARE_SIZE], [0, SQUARE_SIZE]
            ], dtype=np.float32)
            
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            square_img = cv2.warpPerspective(img_rgb, M, (SQUARE_SIZE, SQUARE_SIZE))
            
            label = f"{FILES[col]}{RANKS[row]}"
            squares[label] = Image.fromarray(square_img)
            
            y_out = row * SQUARE_SIZE
            x_out = col * SQUARE_SIZE
            stitched[y_out:y_out+SQUARE_SIZE, x_out:x_out+SQUARE_SIZE] = square_img
    
    return squares, Image.fromarray(stitched)


# =============================================================================
# PIPELINE: COLOR + CV (Legacy Mode 1)
# =============================================================================

def color_cv_pipeline(image):
    """Legacy Color+CV pipeline - returns 7-tuple with grid_overlay"""
    print("  [COLOR+CV PIPELINE]")
    
    img_rgb = np.array(image.convert('RGB'))
    img_h, img_w = img_rgb.shape[:2]
    
    # Rough crop
    x1, y1 = int(img_w * 0.02), int(img_h * 0.02)
    x2, y2 = int(img_w * 0.98), int(img_h * 0.7)
    
    if y2 - y1 < 500 or x2 - x1 < 500:
        cropped_rgb = img_rgb
    else:
        cropped_rgb = img_rgb[y1:y2, x1:x2]
    
    img_hsv = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2HSV)
    
    # Color masks
    mask_tan = cv2.inRange(img_hsv, TAN_LOW, TAN_HIGH)
    mask_brown = cv2.inRange(img_hsv, BROWN_LOW, BROWN_HIGH)
    mask_green = cv2.inRange(img_hsv, GREEN_LOW, GREEN_HIGH)
    mask_light = cv2.inRange(img_hsv, LIGHT_LOW, LIGHT_HIGH)
    
    combined_mask = cv2.bitwise_or(mask_tan, mask_brown)
    combined_mask = cv2.bitwise_or(combined_mask, mask_green)
    combined_mask = cv2.bitwise_or(combined_mask, mask_light)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined_mask)
    if num_labels < 2:
        return None
    
    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    if stats[largest_idx, cv2.CC_STAT_AREA] < 10000:
        return None
    
    component_mask = (labels == largest_idx).astype(np.uint8) * 255
    
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    if len(approx) < 4:
        return None
    
    points = approx.reshape(-1, 2)
    sum_pts = points[:, 0] + points[:, 1]
    diff_pts = points[:, 0] - points[:, 1]
    
    tl = points[np.argmin(sum_pts)]
    br = points[np.argmax(sum_pts)]
    tr = points[np.argmax(diff_pts)]
    bl = points[np.argmin(diff_pts)]
    
    # Perspective transform
    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    board_size = int(max(width, height))
    margin = int(board_size * 0.10)
    output_size = board_size + 2 * margin
    
    src_pts = np.array([tl, tr, br, bl], dtype=np.float32)
    dst_pts = np.array([
        [margin, margin],
        [margin + board_size, margin],
        [margin + board_size, margin + board_size],
        [margin, margin + board_size]
    ], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(cropped_rgb, M, (output_size, output_size))
    warped_pil = Image.fromarray(warped)
    gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
    
    # CV corner detection
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    methods = [
        ("clahe", enhanced),
        ("gray", gray),
        ("blur5", cv2.GaussianBlur(gray, (5, 5), 0)),
    ]
    
    best_corners = None
    best_pattern = None
    
    for method_name, processed in methods:
        for pattern in [(7, 7), (6, 6), (5, 5), (4, 4), (3, 3)]:
            ret, corners = cv2.findChessboardCorners(
                processed, pattern,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            if ret:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                best_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                best_pattern = pattern
                break
        if best_corners is not None:
            break
    
    crop_h, crop_w = gray.shape
    
    if best_corners is not None:
        squares, stitched, grid_overlay = extract_with_grid_overlay(
            warped, gray, best_corners, best_pattern, crop_w, crop_h
        )
        corners_str = f"{best_pattern[0]}x{best_pattern[1]}"
        confidence = 0.9
    else:
        squares, stitched, grid_overlay = extract_squares_divide8_with_grid(warped)
        corners_str = "divide8"
        confidence = 0.7
    
    return squares, stitched, warped_pil, grid_overlay, confidence, "color_cv", corners_str


# =============================================================================
# PIPELINE: ROBOFLOW (Mode 2) - Simplified
# =============================================================================

def roboflow_pipeline(image):
    """Roboflow pipeline - returns 7-tuple with grid_overlay"""
    print("  [ROBOFLOW PIPELINE]")
    
    # Call Roboflow API
    detection = detect_board_roboflow(image)
    if detection is None:
        print("    ❌ Roboflow detection failed")
        return None
    
    # Crop board with expansion
    img_np = np.array(image.convert('RGB'))
    img_h, img_w = img_np.shape[:2]
    
    cx, cy = detection['x'], detection['y']
    w, h = detection['width'], detection['height']
    
    expand = 0.10
    x1 = max(0, int(cx - w/2 - w * expand))
    y1 = max(0, int(cy - h/2 - h * expand))
    x2 = min(img_w, int(cx + w/2 + w * expand))
    y2 = min(img_h, int(cy + h/2 + h * expand))
    
    cropped = img_np[y1:y2, x1:x2]
    cropped_pil = Image.fromarray(cropped)
    gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
    crop_h, crop_w = gray.shape
    
    # CV corner detection
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    best_corners = None
    best_pattern = None
    
    for processed in [enhanced, gray, cv2.GaussianBlur(gray, (5, 5), 0)]:
        for pattern in [(7, 7), (6, 6), (5, 5), (4, 4), (3, 3)]:
            ret, corners = cv2.findChessboardCorners(
                processed, pattern,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            if ret:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                best_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                best_pattern = pattern
                break
        if best_corners is not None:
            break
    
    if best_corners is not None:
        squares, stitched, grid_overlay = extract_with_grid_overlay(
            cropped, gray, best_corners, best_pattern, crop_w, crop_h
        )
        corners_str = f"{best_pattern[0]}x{best_pattern[1]}"
    else:
        squares, stitched, grid_overlay = extract_squares_divide8_with_grid(cropped)
        corners_str = "divide8"
    
    return squares, stitched, cropped_pil, grid_overlay, detection['confidence'], "roboflow", corners_str


def detect_board_roboflow(image):
    """Call Roboflow API"""
    img_str = image_to_base64(image, quality=85)
    url = f"https://detect.roboflow.com/{BOARD_MODEL}?api_key={ROBOFLOW_API_KEY}"
    
    try:
        response = requests.post(
            url, data=img_str,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30
        )
        result = response.json()
    except Exception as e:
        print(f"    Roboflow API error: {e}")
        return None
    
    if 'predictions' not in result or not result['predictions']:
        return None
    
    pred = result['predictions'][0]
    return {
        'x': pred['x'], 'y': pred['y'],
        'width': pred['width'], 'height': pred['height'],
        'confidence': pred['confidence']
    }


# =============================================================================
# HELPERS
# =============================================================================

def get_image_from_request():
    if 'image' in request.files:
        return Image.open(request.files['image'].stream)
    if request.is_json:
        data = request.get_json()
        if 'image' in data:
            return Image.open(BytesIO(base64.b64decode(data['image'])))
    return None


def image_to_base64(image, quality=90):
    buffered = BytesIO()
    image.convert("RGB").save(buffered, quality=quality, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("ascii")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 ChessScan Server v7 (iOS ML Support) on port {port}")
    print(f"")
    print(f"   MODES:")
    print(f"     mode=1&precropped=true  → iOS ML (board already cropped)")
    print(f"     mode=1                   → Color+CV (legacy)")
    print(f"     mode=2                   → Roboflow")
    print(f"     mode=3                   → Both")
    print(f"")
    print(f"   NEW: Returns grid_overlay_image for debugging!")
    print(f"")
    app.run(host='0.0.0.0', port=port, debug=True)
