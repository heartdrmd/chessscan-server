#!/usr/bin/env python3
"""
ChessScan Server v6 - DUAL PIPELINE with 3 TOGGLE MODES
=========================================================

MODE 1: Color+CV Pipeline (DEFAULT)
  Step 1: Color mask → find tan/brown board colors (HSV)
  Step 2: Contour + approxPolyDP → find 4 board corners
  Step 3: getPerspectiveTransform → straighten with 10% margin
  Step 4: CLAHE + findChessboardCorners → try 7x7 down to 3x3
          ├── IF found → extrapolate to 9x9
          └── IF nothing found → divide board_size/8 → 9x9 grid
  Step 5: Extract 64 squares (300x300 each) → return

MODE 2: Roboflow Pipeline (with fallback to Mode 1)
  Step 1: Roboflow API → crop board with +10% expansion
  Step 2: 8 threshold methods, patterns 7x7→4x4
          ├── IF found → expand to 9x9 → extract 64 squares
          └── IF ALL FAIL → **RUN MODE 1 (Color+CV)**
  Step 3: Return

MODE 3: Both Pipelines (iOS user picks)
  Run Mode 1 → get color_cv result
  Run Mode 2 → get roboflow result
  Return BOTH → iOS displays both, user picks which to accept

USAGE:
  POST /detect?mode=1  → Color+CV only
  POST /detect?mode=2  → Roboflow (fallback to Color+CV)
  POST /detect?mode=3  → Both pipelines

OUTPUT (300x300 squares, 2400x2400 stitched):
  - squares: dict of 64 base64 images (a8→h1)
  - stitched_image: 2400x2400 combined board
  - board_image: cropped/straightened before grid
  - confidence, method, corners_found
"""

from flask import Flask, request, jsonify
import requests
import base64
import numpy as np
from PIL import Image
from io import BytesIO
import cv2
import math
import os

app = Flask(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Roboflow settings
ROBOFLOW_API_KEY = os.environ.get('ROBOFLOW_API_KEY', 'qPVZQqumEL1m2BzsXbV0')
BOARD_MODEL = "chessboard-detection-x5kxd/1"

# Expansion margins
ROBOFLOW_EXPAND = 0.10  # 10% expansion for Roboflow crop
COLOR_CV_MARGIN = 0.10  # 10% margin for perspective transform

# Square size (300x300 for better ML accuracy)
SQUARE_SIZE = 300
BOARD_SIZE = SQUARE_SIZE * 8  # 2400x2400

# Chess square labels
FILES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
RANKS = ['8', '7', '6', '5', '4', '3', '2', '1']  # top to bottom

# Color ranges for board detection (HSV) - BROADER RANGES
# Tan/beige squares
TAN_LOW = np.array([10, 20, 100])
TAN_HIGH = np.array([40, 150, 255])
# Brown/dark squares
BROWN_LOW = np.array([5, 30, 50])
BROWN_HIGH = np.array([35, 200, 220])
# Green squares (chess.com)
GREEN_LOW = np.array([35, 40, 80])
GREEN_HIGH = np.array([85, 255, 220])
# Light green/white squares
LIGHT_LOW = np.array([30, 10, 180])
LIGHT_HIGH = np.array([90, 80, 255])


# =============================================================================
# ROUTES
# =============================================================================

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'chessscan-v6-dual-pipeline'})


@app.route('/detect', methods=['POST'])
def detect_board():
    """
    Main endpoint - returns 64 labeled square images.
    
    Query params:
        mode: 1 (Color+CV), 2 (Roboflow), 3 (Both)
    
    Returns JSON based on mode.
    """
    try:
        # Get mode from query param (default: 1 = Color+CV)
        mode = request.args.get('mode', '1')
        mode = int(mode) if mode in ['1', '2', '3'] else 1
        
        # Get image from request
        image = get_image_from_request()
        if image is None:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        print(f"📸 Received image: {image.size}, Mode: {mode}")
        
        # Run appropriate pipeline(s)
        if mode == 1:
            # Color+CV only
            result = color_cv_pipeline(image)
            if result is None:
                return jsonify({
                    'success': False,
                    'error': 'No chessboard detected (Color+CV)',
                    'board_detected': False
                })
            return format_single_response(result, 'color_cv')
        
        elif mode == 2:
            # Roboflow with fallback to Color+CV
            result = roboflow_pipeline(image)
            if result is None:
                # Fallback to Color+CV
                print("  ⚠️ Roboflow pipeline failed, falling back to Color+CV...")
                result = color_cv_pipeline(image)
                if result is None:
                    return jsonify({
                        'success': False,
                        'error': 'No chessboard detected (Roboflow + Color+CV fallback)',
                        'board_detected': False
                    })
                return format_single_response(result, 'fallback_color_cv')
            return format_single_response(result, 'roboflow')
        
        elif mode == 3:
            # Both pipelines
            color_result = color_cv_pipeline(image)
            robo_result = roboflow_pipeline(image)
            
            if color_result is None and robo_result is None:
                return jsonify({
                    'success': False,
                    'error': 'No chessboard detected (both pipelines failed)',
                    'board_detected': False
                })
            
            return format_both_response(color_result, robo_result)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


def format_single_response(result, method_name):
    """Format response for single pipeline result"""
    squares, stitched_img, board_img, confidence, method, corners_found = result
    
    squares_base64 = {label: image_to_base64(img, quality=90) for label, img in squares.items()}
    
    print(f"✅ Returning 64 squares via {method_name} ({corners_found})")
    
    return jsonify({
        'success': True,
        'method': method_name,
        'squares': squares_base64,
        'stitched_image': image_to_base64(stitched_img, quality=90),
        'board_image': image_to_base64(board_img, quality=90),
        'confidence': confidence,
        'corners_found': corners_found,
        'board_detected': True
    })


def format_both_response(color_result, robo_result):
    """Format response for both pipelines"""
    response = {
        'success': True,
        'method': 'both',
        'board_detected': True
    }
    
    if color_result:
        squares, stitched_img, board_img, confidence, method, corners_found = color_result
        squares_base64 = {label: image_to_base64(img, quality=90) for label, img in squares.items()}
        response['color_cv'] = {
            'squares': squares_base64,
            'stitched_image': image_to_base64(stitched_img, quality=90),
            'board_image': image_to_base64(board_img, quality=90),
            'confidence': confidence,
            'corners_found': corners_found
        }
        print(f"✅ Color+CV: {corners_found}")
    else:
        response['color_cv'] = None
        print("❌ Color+CV: failed")
    
    if robo_result:
        squares, stitched_img, board_img, confidence, method, corners_found = robo_result
        squares_base64 = {label: image_to_base64(img, quality=90) for label, img in squares.items()}
        response['roboflow'] = {
            'squares': squares_base64,
            'stitched_image': image_to_base64(stitched_img, quality=90),
            'board_image': image_to_base64(board_img, quality=90),
            'confidence': confidence,
            'corners_found': corners_found
        }
        print(f"✅ Roboflow: {corners_found}")
    else:
        response['roboflow'] = None
        print("❌ Roboflow: failed")
    
    return jsonify(response)


# =============================================================================
# PIPELINE 1: COLOR + CV (Toggle 1 - DEFAULT)
# =============================================================================

def color_cv_pipeline(image):
    """
    COLOR + CV PIPELINE
    
    Step 0: Rough crop to focus on board area (removes excess background)
    Step 1: Color mask → find tan/brown/green board colors
    Step 2: Contour + approxPolyDP → find 4 board corners
    Step 3: getPerspectiveTransform → straighten with 10% margin
    Step 4: CLAHE + findChessboardCorners → try 7x7 down to 3x3
            ├── IF found → extrapolate to 9x9
            └── IF nothing found → divide board_size/8 → 9x9 grid
    Step 5: Extract 64 squares → return
    
    Returns: (squares_dict, stitched_img, board_img, confidence, method, corners_str) or None
    """
    print("  [COLOR+CV PIPELINE]")
    
    # Convert to numpy
    img_rgb = np.array(image.convert('RGB'))
    img_h, img_w = img_rgb.shape[:2]
    
    # =========================================================
    # STEP 0: Rough crop to focus on board area
    # =========================================================
    print("  Step 0: Rough crop...")
    
    # Crop to upper/center portion where board is likely to be
    # This removes excess background that confuses color detection
    x1 = int(img_w * 0.02)  # 2% margin
    y1 = int(img_h * 0.02)
    x2 = int(img_w * 0.98)
    y2 = int(img_h * 0.7)   # Upper 70% typically contains the board
    
    # Make sure crop is reasonable
    if y2 - y1 < 500 or x2 - x1 < 500:
        # Image too small, use full image
        cropped_rgb = img_rgb
        crop_offset = (0, 0)
    else:
        cropped_rgb = img_rgb[y1:y2, x1:x2]
        crop_offset = (x1, y1)
    
    print(f"    Rough crop: {cropped_rgb.shape[1]}x{cropped_rgb.shape[0]}")
    
    img_hsv = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2HSV)
    
    # =========================================================
    # STEP 1: Color mask to find board region
    # =========================================================
    print("  Step 1: Color mask...")
    
    # Create masks for different board colors
    mask_tan = cv2.inRange(img_hsv, TAN_LOW, TAN_HIGH)
    mask_brown = cv2.inRange(img_hsv, BROWN_LOW, BROWN_HIGH)
    mask_green = cv2.inRange(img_hsv, GREEN_LOW, GREEN_HIGH)
    mask_light = cv2.inRange(img_hsv, LIGHT_LOW, LIGHT_HIGH)
    
    # Combine masks
    combined_mask = cv2.bitwise_or(mask_tan, mask_brown)
    combined_mask = cv2.bitwise_or(combined_mask, mask_green)
    combined_mask = cv2.bitwise_or(combined_mask, mask_light)
    
    # Morphological operations to clean up - LARGER KERNEL
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined_mask)
    
    if num_labels < 2:
        print("    ❌ No board region found")
        return None
    
    # Find largest component (excluding background)
    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_area = stats[largest_idx, cv2.CC_STAT_AREA]
    
    if largest_area < 10000:
        print(f"    ❌ Board region too small: {largest_area}")
        return None
    
    component_mask = (labels == largest_idx).astype(np.uint8) * 255
    print(f"    Found board region: {largest_area} pixels")
    
    # =========================================================
    # STEP 2: Find 4 corners with contour + approxPolyDP
    # =========================================================
    print("  Step 2: Find corners...")
    
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("    ❌ No contours found")
        return None
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Approximate to polygon
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    if len(approx) < 4:
        print(f"    ❌ Not enough corners: {len(approx)}")
        return None
    
    # Get 4 extreme corners
    points = approx.reshape(-1, 2)
    sum_pts = points[:, 0] + points[:, 1]
    diff_pts = points[:, 0] - points[:, 1]
    
    tl = points[np.argmin(sum_pts)]
    br = points[np.argmax(sum_pts)]
    tr = points[np.argmax(diff_pts)]
    bl = points[np.argmin(diff_pts)]
    
    print(f"    Corners: TL={tl}, TR={tr}, BL={bl}, BR={br}")
    
    # =========================================================
    # STEP 3: Perspective transform to straighten
    # =========================================================
    print("  Step 3: Perspective transform...")
    
    # Calculate board dimensions
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    
    board_size = int(max(width_top, width_bottom, height_left, height_right))
    
    # Add margin for CV to find corners at edges
    margin = int(board_size * COLOR_CV_MARGIN)
    output_size = board_size + 2 * margin
    
    # Source points (detected corners)
    src_pts = np.array([tl, tr, br, bl], dtype=np.float32)
    
    # Destination points (with margin)
    dst_pts = np.array([
        [margin, margin],
        [margin + board_size, margin],
        [margin + board_size, margin + board_size],
        [margin, margin + board_size]
    ], dtype=np.float32)
    
    # Perspective transform
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(cropped_rgb, M, (output_size, output_size))
    warped_pil = Image.fromarray(warped)
    
    gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
    crop_h, crop_w = gray.shape
    
    print(f"    Warped to {output_size}x{output_size} (with {margin}px margin)")
    
    # =========================================================
    # STEP 4: Try to find CV corners, else divide by 8
    # =========================================================
    print("  Step 4: Find CV corners or divide by 8...")
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Try multiple threshold methods
    methods = [
        ("clahe", enhanced),
        ("adaptive_gaussian", cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),
        ("adaptive_mean", cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)),
        ("otsu", cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
        ("blur5_otsu", cv2.threshold(cv2.GaussianBlur(gray, (5,5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
        ("blur11_gray", cv2.GaussianBlur(gray, (11,11), 0)),
        ("blur31_gray", cv2.GaussianBlur(gray, (31,31), 0)),
    ]
    
    # Try to find corners with CV
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
                # Refine corners
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                print(f"    {method_name}: {pattern} ✅")
                best_corners = refined
                best_pattern = pattern
                best_method = method_name
                break
        if best_corners is not None:
            break
    
    # Decide: use CV corners or divide by 8
    if best_corners is not None:
        # For small patterns (3x3, 4x4), the CV corners might be in a weird location
        # Use the step size from CV but align to known board bounds
        if best_pattern[0] <= 4:
            print(f"    Small pattern {best_pattern}, using step size but aligning to board bounds...")
            squares, stitched = extract_squares_from_small_pattern(
                warped, gray, best_corners, best_pattern, margin, board_size, crop_w, crop_h
            )
        else:
            # Larger patterns (5x5+) - use normal extrapolation
            squares, stitched = extract_squares_from_corners(
                warped, gray, best_corners, best_pattern, crop_w, crop_h
            )
        corners_str = f"{best_pattern[0]}x{best_pattern[1]}"
        confidence = 0.9
    else:
        # Divide by 8 (use the margin-adjusted region)
        print("    ❌ No CV corners found, dividing by 8...")
        squares, stitched = extract_squares_divide8(warped, margin, board_size)
        corners_str = "divide8"
        best_method = "divide8"
        confidence = 0.7
    
    print(f"  ✅ Color+CV pipeline complete! ({corners_str})")
    
    return squares, stitched, warped_pil, confidence, best_method, corners_str


def extract_squares_from_corners(warped_np, gray, corners, pattern, crop_w, crop_h):
    """Extract 64 squares using CV-detected corners + extrapolation"""
    rows, cols = pattern
    corners_grid = corners.reshape(rows, cols, 2)
    
    # Calculate average step sizes
    h_steps = []
    for i in range(rows):
        for j in range(cols - 1):
            h_steps.append(corners_grid[i, j+1] - corners_grid[i, j])
    avg_h = np.median(h_steps, axis=0)
    
    v_steps = []
    for i in range(rows - 1):
        for j in range(cols):
            v_steps.append(corners_grid[i+1, j] - corners_grid[i, j])
    avg_v = np.median(v_steps, axis=0)
    
    step_h = abs(avg_h[0])
    step_v = abs(avg_v[1])
    
    # Find bounds of detected corners
    detected_min_x = corners_grid[:, :, 0].min()
    detected_max_x = corners_grid[:, :, 0].max()
    detected_min_y = corners_grid[:, :, 1].min()
    detected_max_y = corners_grid[:, :, 1].max()
    
    # Calculate room to expand in each direction
    rows_up = int(detected_min_y / step_v) if step_v > 0 else 0
    rows_down = int((crop_h - detected_max_y) / step_v) if step_v > 0 else 0
    cols_left = int(detected_min_x / step_h) if step_h > 0 else 0
    cols_right = int((crop_w - detected_max_x) / step_h) if step_h > 0 else 0
    
    # We need 9 rows and 9 cols total
    rows_needed = 9 - rows
    cols_needed = 9 - cols
    
    # Distribute expansion toward edges
    actual_rows_up = min(rows_up, rows_needed)
    actual_rows_down = min(rows_down, rows_needed - actual_rows_up)
    if actual_rows_up + actual_rows_down < rows_needed:
        remaining = rows_needed - actual_rows_up - actual_rows_down
        if rows_up > actual_rows_up:
            extra = min(rows_up - actual_rows_up, remaining)
            actual_rows_up += extra
            remaining -= extra
        if remaining > 0 and rows_down > actual_rows_down:
            actual_rows_down += min(rows_down - actual_rows_down, remaining)
    
    actual_cols_left = min(cols_left, cols_needed)
    actual_cols_right = min(cols_right, cols_needed - actual_cols_left)
    if actual_cols_left + actual_cols_right < cols_needed:
        remaining = cols_needed - actual_cols_left - actual_cols_right
        if cols_left > actual_cols_left:
            extra = min(cols_left - actual_cols_left, remaining)
            actual_cols_left += extra
            remaining -= extra
        if remaining > 0 and cols_right > actual_cols_right:
            actual_cols_right += min(cols_right - actual_cols_right, remaining)
    
    total_rows = rows + actual_rows_up + actual_rows_down
    total_cols = cols + actual_cols_left + actual_cols_right
    
    # Build the full grid
    full_grid = np.zeros((total_rows, total_cols, 2), dtype=np.float32)
    
    # Copy detected corners
    for i in range(rows):
        for j in range(cols):
            full_grid[actual_rows_up + i, actual_cols_left + j] = corners_grid[i, j]
    
    # Expand UP
    for i in range(actual_rows_up - 1, -1, -1):
        for j in range(actual_cols_left, actual_cols_left + cols):
            full_grid[i, j] = full_grid[i + 1, j] - avg_v
    
    # Expand DOWN
    for i in range(actual_rows_up + rows, total_rows):
        for j in range(actual_cols_left, actual_cols_left + cols):
            full_grid[i, j] = full_grid[i - 1, j] + avg_v
    
    # Expand LEFT (for all rows)
    for j in range(actual_cols_left - 1, -1, -1):
        for i in range(total_rows):
            full_grid[i, j] = full_grid[i, j + 1] - avg_h
    
    # Expand RIGHT (for all rows)
    for j in range(actual_cols_left + cols, total_cols):
        for i in range(total_rows):
            full_grid[i, j] = full_grid[i, j - 1] + avg_h
    
    # Clip to image bounds
    for i in range(total_rows):
        for j in range(total_cols):
            full_grid[i, j, 0] = np.clip(full_grid[i, j, 0], 0, crop_w - 1)
            full_grid[i, j, 1] = np.clip(full_grid[i, j, 1], 0, crop_h - 1)
    
    # Ensure we have 9x9 - pad if needed
    if total_rows < 9 or total_cols < 9:
        padded_grid = np.zeros((9, 9, 2), dtype=np.float32)
        pad_top = (9 - total_rows) // 2
        pad_left = (9 - total_cols) // 2
        for i in range(total_rows):
            for j in range(total_cols):
                padded_grid[pad_top + i, pad_left + j] = full_grid[i, j]
        # Fill missing by extrapolation
        for i in range(pad_top - 1, -1, -1):
            for j in range(9):
                if padded_grid[i+1, j, 0] > 0 or padded_grid[i+1, j, 1] > 0:
                    padded_grid[i, j] = padded_grid[i + 1, j] - avg_v
        for i in range(pad_top + total_rows, 9):
            for j in range(9):
                if padded_grid[i-1, j, 0] > 0 or padded_grid[i-1, j, 1] > 0:
                    padded_grid[i, j] = padded_grid[i - 1, j] + avg_v
        for j in range(pad_left - 1, -1, -1):
            for i in range(9):
                if padded_grid[i, j+1, 0] > 0 or padded_grid[i, j+1, 1] > 0:
                    padded_grid[i, j] = padded_grid[i, j + 1] - avg_h
        for j in range(pad_left + total_cols, 9):
            for i in range(9):
                if padded_grid[i, j-1, 0] > 0 or padded_grid[i, j-1, 1] > 0:
                    padded_grid[i, j] = padded_grid[i, j - 1] + avg_h
        # Clip padded grid
        for i in range(9):
            for j in range(9):
                padded_grid[i, j, 0] = np.clip(padded_grid[i, j, 0], 0, crop_w - 1)
                padded_grid[i, j, 1] = np.clip(padded_grid[i, j, 1], 0, crop_h - 1)
        full_grid = padded_grid


def extract_squares_from_small_pattern(warped_np, gray, corners, pattern, margin, board_size, crop_w, crop_h):
    """
    For small patterns (3x3, 4x4), use the step size from CV corners
    but align the grid to the known board bounds from perspective transform.
    
    This is better than blind extrapolation because the small pattern
    might be found in a weird location on the board.
    """
    rows, cols = pattern
    corners_grid = corners.reshape(rows, cols, 2)
    
    # Calculate average step sizes from the detected corners
    h_steps = []
    for i in range(rows):
        for j in range(cols - 1):
            h_steps.append(corners_grid[i, j+1] - corners_grid[i, j])
    avg_h = np.median(h_steps, axis=0) if h_steps else np.array([board_size / 8, 0])
    
    v_steps = []
    for i in range(rows - 1):
        for j in range(cols):
            v_steps.append(corners_grid[i+1, j] - corners_grid[i, j])
    avg_v = np.median(v_steps, axis=0) if v_steps else np.array([0, board_size / 8])
    
    step_h = abs(avg_h[0]) if abs(avg_h[0]) > 10 else board_size / 8
    step_v = abs(avg_v[1]) if abs(avg_v[1]) > 10 else board_size / 8
    
    # Use the known board bounds
    board_left = margin
    board_top = margin
    board_right = margin + board_size
    board_bottom = margin + board_size
    
    # Build 9x9 grid aligned to board bounds, using CV step size
    # Average the CV step with the divide-by-8 step for robustness
    expected_step = board_size / 8
    final_step_h = (step_h + expected_step) / 2
    final_step_v = (step_v + expected_step) / 2
    
    full_grid = np.zeros((9, 9, 2), dtype=np.float32)
    for i in range(9):
        for j in range(9):
            x = board_left + j * final_step_h
            y = board_top + i * final_step_v
            # Clip to image bounds
            x = np.clip(x, 0, crop_w - 1)
            y = np.clip(y, 0, crop_h - 1)
            full_grid[i, j] = [x, y]
    
    return extract_squares_from_grid(warped_np, full_grid)
    
    # Extract 64 squares
    return extract_squares_from_grid(warped_np, full_grid)


def extract_squares_divide8(warped_np, margin, board_size):
    """Extract 64 squares by simply dividing the board into 8x8"""
    # Build a simple 9x9 grid based on dividing the board
    square_step = board_size / 8
    
    full_grid = np.zeros((9, 9, 2), dtype=np.float32)
    for i in range(9):
        for j in range(9):
            full_grid[i, j] = [margin + j * square_step, margin + i * square_step]
    
    return extract_squares_from_grid(warped_np, full_grid)


def extract_squares_from_grid(warped_np, full_grid):
    """Extract 64 squares from a 9x9 grid of corners"""
    squares = {}
    
    for row in range(8):
        for col in range(8):
            # Get 4 corners of this square
            tl = full_grid[row, col]
            tr = full_grid[row, col + 1]
            bl = full_grid[row + 1, col]
            br = full_grid[row + 1, col + 1]
            
            # Perspective warp to perfect square
            src_pts = np.array([tl, tr, br, bl], dtype=np.float32)
            dst_pts = np.array([
                [0, 0], [SQUARE_SIZE, 0],
                [SQUARE_SIZE, SQUARE_SIZE], [0, SQUARE_SIZE]
            ], dtype=np.float32)
            
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            square_img = cv2.warpPerspective(warped_np, M, (SQUARE_SIZE, SQUARE_SIZE))
            
            # Label in chess notation
            label = f"{FILES[col]}{RANKS[row]}"
            squares[label] = Image.fromarray(square_img)
    
    # Create stitched board image
    stitched = stitch_squares(squares)
    
    return squares, stitched


def extract_squares_roboflow_divide8(cropped_np, robo_w, robo_h, expand_w, expand_h, crop_w, crop_h):
    """
    Extract 64 squares using Roboflow bounds directly (divide by 8).
    Used when CV corner detection fails but Roboflow succeeded.
    """
    # The inner board is the Roboflow detection minus the expansion
    inner_x1 = expand_w
    inner_y1 = expand_h
    
    # Clamp to actual crop bounds
    inner_x1 = max(0, min(inner_x1, crop_w - robo_w))
    inner_y1 = max(0, min(inner_y1, crop_h - robo_h))
    
    # Build 9x9 grid by dividing the Roboflow detection area
    square_w = robo_w / 8
    square_h = robo_h / 8
    
    full_grid = np.zeros((9, 9, 2), dtype=np.float32)
    for i in range(9):
        for j in range(9):
            x = inner_x1 + j * square_w
            y = inner_y1 + i * square_h
            # Clip to bounds
            x = np.clip(x, 0, crop_w - 1)
            y = np.clip(y, 0, crop_h - 1)
            full_grid[i, j] = [x, y]
    
    return extract_squares_from_grid(cropped_np, full_grid)


def stitch_squares(squares):
    """Combine 64 squares into one board image"""
    board_img = np.zeros((BOARD_SIZE, BOARD_SIZE, 3), dtype=np.uint8)
    
    for row in range(8):
        for col in range(8):
            label = f"{FILES[col]}{RANKS[row]}"
            square_np = np.array(squares[label])
            
            y1, y2 = row * SQUARE_SIZE, (row + 1) * SQUARE_SIZE
            x1, x2 = col * SQUARE_SIZE, (col + 1) * SQUARE_SIZE
            board_img[y1:y2, x1:x2] = square_np
    
    return Image.fromarray(board_img)


# =============================================================================
# PIPELINE 2: ROBOFLOW (Toggle 2)
# =============================================================================

def roboflow_pipeline(image):
    """
    ROBOFLOW PIPELINE
    
    Step 1: Roboflow API → crop board with +10% expansion
    Step 2: 8 threshold methods, patterns 7x7→4x4
            ├── IF found → expand to 9x9 → extract 64 squares
            └── IF ALL FAIL → return None (caller will fallback to Color+CV)
    
    Returns: (squares_dict, stitched_img, board_img, confidence, method, corners_str) or None
    """
    print("  [ROBOFLOW PIPELINE]")
    
    # =========================================================
    # STEP 1: ROBOFLOW - Crop board with +10% expansion
    # =========================================================
    print("  Step 1: Roboflow crop (+10% expanded)...")
    
    max_dim = 1500
    ratio = max_dim / max(image.size)
    if ratio < 1:
        resized = image.resize(
            (int(image.size[0] * ratio), int(image.size[1] * ratio)), 
            Image.LANCZOS
        )
    else:
        resized = image
        ratio = 1.0
    
    img_w, img_h = resized.size
    
    detection = detect_board_roboflow(resized)
    if detection is None:
        print("    ❌ Roboflow failed to detect board")
        return None
    
    cx, cy = detection['x'], detection['y']
    w, h = detection['width'], detection['height']
    
    # EXPAND by 10% in all 4 directions
    expand_w = w * ROBOFLOW_EXPAND
    expand_h = h * ROBOFLOW_EXPAND
    
    x1 = max(0, int(cx - w/2 - expand_w))
    y1 = max(0, int(cy - h/2 - expand_h))
    x2 = min(img_w, int(cx + w/2 + expand_w))
    y2 = min(img_h, int(cy + h/2 + expand_h))
    
    cropped = resized.crop((x1, y1, x2, y2))
    cropped_np = np.array(cropped.convert('RGB'))
    cropped_pil = cropped.convert('RGB')
    gray = cv2.cvtColor(cropped_np, cv2.COLOR_RGB2GRAY)
    crop_h, crop_w = gray.shape
    
    print(f"    Roboflow: {detection['confidence']:.0%}, expanded to {cropped.size}")
    
    # =========================================================
    # STEP 2: TRY 8 THRESHOLD METHODS
    # =========================================================
    print("  Step 2: Try 8 threshold methods...")
    
    methods = [
        ("adaptive_gaussian", cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),
        ("adaptive_mean", cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)),
        ("otsu", cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
        ("blur5_otsu", cv2.threshold(cv2.GaussianBlur(gray, (5,5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
        ("grayscale", gray),
        ("blur5_gray", cv2.GaussianBlur(gray, (5,5), 0)),
        ("blur11_gray", cv2.GaussianBlur(gray, (11,11), 0)),
        ("blur31_gray", cv2.GaussianBlur(gray, (31,31), 0)),
    ]
    
    # Collect all successful detections with their tilt
    candidates = []
    
    for method_name, bw in methods:
        for pattern in [(7, 7), (6, 6), (5, 5), (4, 4)]:
            ret, corners = cv2.findChessboardCorners(
                bw, pattern,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            if ret:
                # Refine corners
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                rows, cols = pattern
                grid = refined.reshape(rows, cols, 2)
                tilt = math.degrees(math.atan2(
                    grid[0, cols-1, 1] - grid[0, 0, 1],
                    grid[0, cols-1, 0] - grid[0, 0, 0]
                ))
                
                print(f"    {method_name}: {pattern} ✅ (tilt: {tilt:.1f}°)")
                candidates.append((refined, pattern, method_name, tilt))
                break
        else:
            print(f"    {method_name}: ❌")
    
    if not candidates:
        print("    ❌ No corners found with any method, using Roboflow divide-by-8...")
        # Use Roboflow bounds directly and divide by 8
        # The Roboflow detection gives us the board - just divide it
        squares, stitched = extract_squares_roboflow_divide8(
            cropped_np, w, h, expand_w, expand_h, crop_w, crop_h
        )
        return squares, stitched, cropped_pil, detection['confidence'], "roboflow_divide8", "divide8"
    
    # Pick best: prefer larger pattern, then tilt closest to 0°
    candidates.sort(key=lambda x: (-x[1][0], abs(x[3])))
    
    best_corners, best_pattern, best_method, raw_tilt = candidates[0]
    
    # Clamp tilt to ±40° max
    tilt = max(-40, min(40, raw_tilt))
    
    print(f"    BEST: {best_method} {best_pattern}, tilt {tilt:.1f}°")
    
    # =========================================================
    # STEP 3: EXPAND TO 9x9 (toward edges)
    # =========================================================
    print("  Step 3: Expand to 9x9...")
    
    rows, cols = best_pattern
    corners_grid = best_corners.reshape(rows, cols, 2)
    
    # Calculate average step sizes
    h_steps = []
    for i in range(rows):
        for j in range(cols - 1):
            h_steps.append(corners_grid[i, j+1] - corners_grid[i, j])
    avg_h = np.median(h_steps, axis=0)
    
    v_steps = []
    for i in range(rows - 1):
        for j in range(cols):
            v_steps.append(corners_grid[i+1, j] - corners_grid[i, j])
    avg_v = np.median(v_steps, axis=0)
    
    step_h = abs(avg_h[0])
    step_v = abs(avg_v[1])
    
    # Find bounds of detected corners
    detected_min_x = corners_grid[:, :, 0].min()
    detected_max_x = corners_grid[:, :, 0].max()
    detected_min_y = corners_grid[:, :, 1].min()
    detected_max_y = corners_grid[:, :, 1].max()
    
    # Calculate room to expand in each direction
    rows_up = int(detected_min_y / step_v) if step_v > 0 else 0
    rows_down = int((crop_h - detected_max_y) / step_v) if step_v > 0 else 0
    cols_left = int(detected_min_x / step_h) if step_h > 0 else 0
    cols_right = int((crop_w - detected_max_x) / step_h) if step_h > 0 else 0
    
    print(f"    Room: up={rows_up}, down={rows_down}, left={cols_left}, right={cols_right}")
    
    # We need 9 rows and 9 cols total
    rows_needed = 9 - rows
    cols_needed = 9 - cols
    
    # Distribute expansion toward edges
    actual_rows_up = min(rows_up, rows_needed)
    actual_rows_down = min(rows_down, rows_needed - actual_rows_up)
    if actual_rows_up + actual_rows_down < rows_needed:
        remaining = rows_needed - actual_rows_up - actual_rows_down
        if rows_up > actual_rows_up:
            extra = min(rows_up - actual_rows_up, remaining)
            actual_rows_up += extra
            remaining -= extra
        if remaining > 0 and rows_down > actual_rows_down:
            actual_rows_down += min(rows_down - actual_rows_down, remaining)
    
    actual_cols_left = min(cols_left, cols_needed)
    actual_cols_right = min(cols_right, cols_needed - actual_cols_left)
    if actual_cols_left + actual_cols_right < cols_needed:
        remaining = cols_needed - actual_cols_left - actual_cols_right
        if cols_left > actual_cols_left:
            extra = min(cols_left - actual_cols_left, remaining)
            actual_cols_left += extra
            remaining -= extra
        if remaining > 0 and cols_right > actual_cols_right:
            actual_cols_right += min(cols_right - actual_cols_right, remaining)
    
    total_rows = rows + actual_rows_up + actual_rows_down
    total_cols = cols + actual_cols_left + actual_cols_right
    
    print(f"    Expansion: up={actual_rows_up}, down={actual_rows_down}, left={actual_cols_left}, right={actual_cols_right}")
    print(f"    Grid: {total_rows}x{total_cols}")
    
    # Build the full grid
    full_grid = np.zeros((total_rows, total_cols, 2), dtype=np.float32)
    
    # Copy detected corners
    for i in range(rows):
        for j in range(cols):
            full_grid[actual_rows_up + i, actual_cols_left + j] = corners_grid[i, j]
    
    # Expand UP
    for i in range(actual_rows_up - 1, -1, -1):
        for j in range(actual_cols_left, actual_cols_left + cols):
            full_grid[i, j] = full_grid[i + 1, j] - avg_v
    
    # Expand DOWN
    for i in range(actual_rows_up + rows, total_rows):
        for j in range(actual_cols_left, actual_cols_left + cols):
            full_grid[i, j] = full_grid[i - 1, j] + avg_v
    
    # Expand LEFT (for all rows)
    for j in range(actual_cols_left - 1, -1, -1):
        for i in range(total_rows):
            full_grid[i, j] = full_grid[i, j + 1] - avg_h
    
    # Expand RIGHT (for all rows)
    for j in range(actual_cols_left + cols, total_cols):
        for i in range(total_rows):
            full_grid[i, j] = full_grid[i, j - 1] + avg_h
    
    # Clip to image bounds
    for i in range(total_rows):
        for j in range(total_cols):
            full_grid[i, j, 0] = np.clip(full_grid[i, j, 0], 0, crop_w - 1)
            full_grid[i, j, 1] = np.clip(full_grid[i, j, 1], 0, crop_h - 1)
    
    # Ensure we have 9x9 - pad if needed
    if total_rows < 9 or total_cols < 9:
        print(f"    ⚠️ Only got {total_rows}x{total_cols}, padding to 9x9")
        padded_grid = np.zeros((9, 9, 2), dtype=np.float32)
        pad_top = (9 - total_rows) // 2
        pad_left = (9 - total_cols) // 2
        for i in range(total_rows):
            for j in range(total_cols):
                padded_grid[pad_top + i, pad_left + j] = full_grid[i, j]
        # Fill missing by extrapolation
        for i in range(pad_top - 1, -1, -1):
            for j in range(9):
                if padded_grid[i+1, j, 0] > 0 or padded_grid[i+1, j, 1] > 0:
                    padded_grid[i, j] = padded_grid[i + 1, j] - avg_v
        for i in range(pad_top + total_rows, 9):
            for j in range(9):
                if padded_grid[i-1, j, 0] > 0 or padded_grid[i-1, j, 1] > 0:
                    padded_grid[i, j] = padded_grid[i - 1, j] + avg_v
        for j in range(pad_left - 1, -1, -1):
            for i in range(9):
                if padded_grid[i, j+1, 0] > 0 or padded_grid[i, j+1, 1] > 0:
                    padded_grid[i, j] = padded_grid[i, j + 1] - avg_h
        for j in range(pad_left + total_cols, 9):
            for i in range(9):
                if padded_grid[i, j-1, 0] > 0 or padded_grid[i, j-1, 1] > 0:
                    padded_grid[i, j] = padded_grid[i, j - 1] + avg_h
        # Clip padded grid
        for i in range(9):
            for j in range(9):
                padded_grid[i, j, 0] = np.clip(padded_grid[i, j, 0], 0, crop_w - 1)
                padded_grid[i, j, 1] = np.clip(padded_grid[i, j, 1], 0, crop_h - 1)
        full_grid = padded_grid
    
    # =========================================================
    # STEP 4: EXTRACT 64 SQUARES
    # =========================================================
    print("  Step 4: Extract 64 squares...")
    
    squares, stitched = extract_squares_from_grid(cropped_np, full_grid)
    
    corners_str = f"{best_pattern[0]}x{best_pattern[1]}"
    confidence = detection['confidence']
    
    print(f"  ✅ Roboflow pipeline complete! ({corners_str})")
    
    return squares, stitched, cropped_pil, confidence, best_method, corners_str


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def detect_board_roboflow(image):
    """Call Roboflow API to detect chessboard region"""
    img_str = image_to_base64(image, quality=85)
    
    url = f"https://detect.roboflow.com/{BOARD_MODEL}?api_key={ROBOFLOW_API_KEY}"
    
    try:
        response = requests.post(
            url,
            data=img_str,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30
        )
        result = response.json()
    except Exception as e:
        print(f"    Roboflow API error: {e}")
        return None
    
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


def get_image_from_request():
    """Extract image from request (file upload or base64 JSON)"""
    if 'image' in request.files:
        file = request.files['image']
        return Image.open(file.stream)
    
    if request.is_json:
        data = request.get_json()
        if 'image' in data:
            img_data = base64.b64decode(data['image'])
            return Image.open(BytesIO(img_data))
    
    return None


def image_to_base64(image, quality=90):
    """Convert PIL Image to base64 JPEG string"""
    buffered = BytesIO()
    image_rgb = image.convert("RGB")
    image_rgb.save(buffered, quality=quality, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("ascii")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 ChessScan Server v6 (Dual Pipeline) starting on port {port}")
    print(f"")
    print(f"   MODES (query param: mode=1/2/3):")
    print(f"     1 (default): Color+CV Pipeline")
    print(f"     2: Roboflow Pipeline (fallback to Color+CV)")
    print(f"     3: Both Pipelines (iOS user picks)")
    print(f"")
    print(f"   SQUARE SIZE: {SQUARE_SIZE}x{SQUARE_SIZE} (board: {BOARD_SIZE}x{BOARD_SIZE})")
    print(f"")
    app.run(host='0.0.0.0', port=port, debug=True)
