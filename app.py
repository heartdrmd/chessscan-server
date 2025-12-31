#!/usr/bin/env python3
"""
ChessScan Server v5 - JIGSAW PUZZLE PIPELINE (IMPROVED)
========================================================

IMPROVEMENTS OVER v4:
  1. Roboflow +10% expansion in all 4 directions (more room to extrapolate)
  2. 8 threshold methods (was 3) with EARLY EXIT on 7x7
  3. Expand grid TOWARD EDGES (not symmetric) - fills Roboflow bounds
  4. Clip extrapolation to image bounds
  5. NOW RETURNS grid_image with 9x9 dots visualization!

PIPELINE:
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: ROBOFLOW (+10% expanded)                           │
│          → Crop board from screenshot/photo                 │
│          → Add 10% margin on all sides for extrapolation    │
│                                                             │
│  STEP 2: TRY 8 THRESHOLD METHODS (early exit on 7x7)        │
│          1. adaptive_gaussian                               │
│          2. adaptive_mean                                   │
│          3. otsu                                            │
│          4. blur5_otsu (5x5 blur + Otsu)                    │
│          5. grayscale                                       │
│          6. blur5_gray (5x5 blur)                           │
│          7. blur11_gray (11x11 blur)                        │
│          8. blur31_gray (31x31 blur) - LAST RESORT          │
│                                                             │
│  STEP 3: REFINE CORNERS                                     │
│          → cornerSubPix for sub-pixel accuracy              │
│                                                             │
│  STEP 4: EXPAND TO 9x9 (toward edges, not symmetric)        │
│          → Calculate room in each direction                 │
│          → Expand until hitting Roboflow bounds             │
│          → Result: 9x9 grid filling the crop                │
│                                                             │
│  STEP 5: EXTRACT 64 INDIVIDUAL SQUARES                      │
│          → Each square warped to perfect 100x100            │
│          → Labeled a8, b8, c8... h1 (chess notation)        │
│                                                             │
│  STEP 6: RETURN TO iOS                                      │
│          → 64 labeled square images                         │
│          → iOS runs ML on each → piece or empty             │
└─────────────────────────────────────────────────────────────┘

BOARD COMPATIBILITY:
  ✅ Chess.com (high contrast)
  ✅ Lichess (tan/beige low contrast)
  ✅ Puzzle diagrams (wood texture)
  ✅ Book diagrams with diagonal hatching (blur31 method)
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

# Roboflow settings
ROBOFLOW_API_KEY = os.environ.get('ROBOFLOW_API_KEY', 'qPVZQqumEL1m2BzsXbV0')
BOARD_MODEL = "chessboard-detection-x5kxd/1"

# Roboflow expansion (2.5% in each direction - minimal safety margin)
ROBOFLOW_EXPAND = 0.025

# Chess square labels
FILES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
RANKS = ['8', '7', '6', '5', '4', '3', '2', '1']  # top to bottom


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'chessscan-v5-jigsaw'})


@app.route('/detect', methods=['POST'])
def detect_board():
    """
    Main endpoint - returns 64 labeled square images.
    
    Returns JSON:
    {
        'success': bool,
        'squares': {
            'a8': 'base64...', 'b8': 'base64...', ... 'h1': 'base64...'
        },
        'board_image': 'base64...',  # Reconstructed 800x800
        'grid_image': 'base64...',   # 9x9 grid with dots on original crop
        'confidence': float,
        'method': str,  # Which threshold method worked
        'corners_found': str,  # e.g., "7x7"
    }
    """
    try:
        # Get image from request
        image = get_image_from_request()
        if image is None:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        print(f"📸 Received image: {image.size}")
        
        # Run the jigsaw pipeline
        result = jigsaw_pipeline(image)
        
        if result is None:
            return jsonify({
                'success': False,
                'error': 'No chessboard detected',
                'board_detected': False
            })
        
        squares, board_img, grid_img, confidence, method, corners_found = result
        
        # Convert squares to base64
        squares_base64 = {}
        for label, img in squares.items():
            squares_base64[label] = image_to_base64(img, quality=90)
        
        # Convert reconstructed board to base64
        board_base64 = image_to_base64(board_img, quality=90)
        
        # Convert grid image to base64
        grid_base64 = image_to_base64(grid_img, quality=90)
        
        print(f"✅ Returning 64 squares via {method} ({corners_found})")
        
        return jsonify({
            'success': True,
            'squares': squares_base64,
            'board_image': board_base64,
            'grid_image': grid_base64,
            'confidence': confidence,
            'method': method,
            'corners_found': corners_found,
            'board_detected': True
        })
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


def jigsaw_pipeline(image):
    """
    THE JIGSAW PUZZLE PIPELINE v5
    
    Returns: (squares_dict, board_image, grid_image, confidence, method, corners_str) or None
    
    squares_dict = {'a8': PIL.Image, 'b8': PIL.Image, ... 'h1': PIL.Image}
    """
    
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
        print("    ❌ Roboflow failed")
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
    gray = cv2.cvtColor(cropped_np, cv2.COLOR_RGB2GRAY)
    crop_h, crop_w = gray.shape
    
    print(f"    Roboflow: {detection['confidence']:.0%}, expanded to {cropped.size}")
    
    # =========================================================
    # STEP 2: TRY 8 THRESHOLD METHODS (early exit on 7x7)
    # =========================================================
    print("  Step 2: Try 8 threshold methods...")
    
    methods = [
        ('adaptive_gaussian', lambda g: cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),
        ('adaptive_mean', lambda g: cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)),
        ('otsu', lambda g: cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
        ('blur5_otsu', lambda g: cv2.threshold(cv2.GaussianBlur(g, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
        ('grayscale', lambda g: g),
        ('blur5_gray', lambda g: cv2.GaussianBlur(g, (5, 5), 0)),
        ('blur11_gray', lambda g: cv2.GaussianBlur(g, (11, 11), 0)),
        ('blur31_gray', lambda g: cv2.GaussianBlur(g, (31, 31), 0)),
    ]
    
    best_result = None
    best_method = None
    best_pattern = (0, 0)
    
    for method_name, method_func in methods:
        try:
            processed = method_func(gray)
            found, corners = cv2.findChessboardCorners(
                processed, (7, 7),
                cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
            )
            if found:
                print(f"    {method_name}: (7, 7) ✅")
                print(f"    → 7x7 found! EARLY EXIT")
                best_result = corners
                best_method = method_name
                best_pattern = (7, 7)
                break
        except:
            pass
        
        for pattern in [(6, 6), (5, 5), (4, 4)]:
            try:
                processed = method_func(gray)
                found, corners = cv2.findChessboardCorners(
                    processed, pattern,
                    cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
                )
                if found:
                    rows, cols = pattern
                    score = rows * cols
                    if score > best_pattern[0] * best_pattern[1]:
                        best_result = corners
                        best_method = method_name
                        best_pattern = pattern
                        print(f"    {method_name}: {pattern} ✅")
                    break
            except:
                pass
        else:
            print(f"    {method_name}: ❌")
            continue
        
        if best_pattern == (7, 7):
            break
    
    if best_result is None:
        print("    ❌ No corners found")
        return None
    
    print(f"    BEST: {best_method} with {best_pattern}")
    
    # =========================================================
    # STEP 3: REFINE CORNERS
    # =========================================================
    print("  Step 3: Refine corners...")
    
    corners = cv2.cornerSubPix(
        gray, best_result, (11, 11), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    )
    
    rows, cols = best_pattern
    corners_grid = corners.reshape(rows, cols, 2)
    
    # Check if we need to rotate (a8 should be top-left with black)
    top_left = corners_grid[0, 0]
    top_right = corners_grid[0, -1]
    bottom_left = corners_grid[-1, 0]
    
    # Determine orientation
    needs_rotate = False
    dx = top_right[0] - top_left[0]
    dy = bottom_left[1] - top_left[1]
    
    if dx < 0 or dy < 0:
        needs_rotate = True
        corners_grid = corners_grid[::-1, ::-1, :]
    
    # Calculate tilt
    angle = math.degrees(math.atan2(
        corners_grid[0, -1, 1] - corners_grid[0, 0, 1],
        corners_grid[0, -1, 0] - corners_grid[0, 0, 0]
    ))
    print(f"    Tilt: {angle:.1f}° → {'OK' if abs(angle) < 10 else 'Rotating...'}")
    
    # =========================================================
    # STEP 4: EXPAND TO 9x9 (toward edges)
    # =========================================================
    print("  Step 4: Expand to 9x9 (toward edges)...")
    
    # Calculate average step size
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
        total_rows, total_cols = 9, 9
    
    print(f"    Final grid: {total_rows}x{total_cols}")
    
    # =========================================================
    # STEP 4b: CREATE GRID IMAGE WITH DOTS (for debug)
    # =========================================================
    grid_vis = cropped_np.copy()
    
    # Draw all 9x9 = 81 corner points
    for i in range(9):
        for j in range(9):
            pt = (int(full_grid[i, j, 0]), int(full_grid[i, j, 1]))
            # Green for detected corners, Red for extrapolated
            if actual_rows_up <= i < actual_rows_up + rows and actual_cols_left <= j < actual_cols_left + cols:
                color = (0, 255, 0)  # Green - detected
                radius = 6
            else:
                color = (255, 0, 0)  # Red - extrapolated
                radius = 5
            cv2.circle(grid_vis, pt, radius, color, -1)
            cv2.circle(grid_vis, pt, radius, (255, 255, 255), 1)  # White border
    
    # Draw grid lines
    for i in range(9):
        for j in range(8):
            pt1 = (int(full_grid[i, j, 0]), int(full_grid[i, j, 1]))
            pt2 = (int(full_grid[i, j+1, 0]), int(full_grid[i, j+1, 1]))
            cv2.line(grid_vis, pt1, pt2, (255, 255, 0), 1)  # Yellow horizontal
    for i in range(8):
        for j in range(9):
            pt1 = (int(full_grid[i, j, 0]), int(full_grid[i, j, 1]))
            pt2 = (int(full_grid[i+1, j, 0]), int(full_grid[i+1, j, 1]))
            cv2.line(grid_vis, pt1, pt2, (255, 255, 0), 1)  # Yellow vertical
    
    grid_pil = Image.fromarray(grid_vis)
    
    # =========================================================
    # STEP 5: EXTRACT 64 INDIVIDUAL SQUARES
    # =========================================================
    print("  Step 5: Extract 64 squares...")
    
    squares = {}
    square_size = 300  # Larger squares = better ML accuracy
    
    for row in range(8):
        for col in range(8):
            # Get 4 corners of this square
            tl = full_grid[row, col]
            tr = full_grid[row, col + 1]
            bl = full_grid[row + 1, col]
            br = full_grid[row + 1, col + 1]
            
            # Perspective warp to perfect 100x100
            src_pts = np.array([tl, tr, br, bl], dtype=np.float32)
            dst_pts = np.array([
                [0, 0], [square_size, 0],
                [square_size, square_size], [0, square_size]
            ], dtype=np.float32)
            
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            square_img = cv2.warpPerspective(cropped_np, M, (square_size, square_size))
            
            # Handle rotation if needed
            if needs_rotate:
                square_img = cv2.rotate(square_img, cv2.ROTATE_180)
                actual_row = 7 - row
                actual_col = 7 - col
            else:
                actual_row = row
                actual_col = col
            
            # Label in chess notation
            label = f"{FILES[actual_col]}{RANKS[actual_row]}"
            squares[label] = Image.fromarray(square_img)
    
    print(f"    Extracted 64 squares (a8 to h1)")
    
    # =========================================================
    # STEP 6: RECONSTRUCT BOARD IMAGE (for verification)
    # =========================================================
    print("  Step 6: Reconstruct board...")
    
    board_img = np.zeros((2400, 2400, 3), dtype=np.uint8)
    
    for row in range(8):
        for col in range(8):
            label = f"{FILES[col]}{RANKS[row]}"
            square_np = np.array(squares[label])
            
            y1, y2 = row * 300, (row + 1) * 300
            x1, x2 = col * 300, (col + 1) * 300
            board_img[y1:y2, x1:x2] = square_np
    
    board_pil = Image.fromarray(board_img)
    
    # Calculate confidence
    confidence = detection['confidence']
    corners_str = f"{best_pattern[0]}x{best_pattern[1]}"
    
    print(f"  ✅ Pipeline complete!")
    
    return squares, board_pil, grid_pil, confidence, best_method, corners_str


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


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 ChessScan Server v5 (Jigsaw Pipeline) starting on port {port}")
    print(f"")
    print(f"   IMPROVEMENTS:")
    print(f"     • Roboflow +10% expansion for extrapolation room")
    print(f"     • 8 threshold methods (early exit on 7x7)")
    print(f"     • Expand grid toward edges (fills Roboflow bounds)")
    print(f"     • NOW RETURNS grid_image with 9x9 dots!")
    print(f"")
    print(f"   Pipeline: Roboflow(+10%) → 8 Thresholds → OpenCV corners")
    print(f"             → Expand to edges → Extract 64 squares")
    print(f"")
    app.run(host='0.0.0.0', port=port, debug=True)
