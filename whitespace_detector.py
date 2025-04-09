# whitespace_detector.py
import cv2
import numpy as np
import os
import random

class WhiteSpaceDetector:
    def __init__(self, debug_mode=True):
        # White space detection parameters
        self.WHITE_LOWER = np.array([200, 200, 200])  # Lower bound for white in BGR
        self.WHITE_UPPER = np.array([255, 255, 255])  # Upper bound for white in BGR
        
        # Size constraints
        self.MIN_SPACE_WIDTH = 50
        self.MIN_SPACE_HEIGHT = 30
        self.SAFETY_MARGIN = 10  # Safety margin around white spaces
        
        # Debug settings
        self.debug_mode = debug_mode
        self.debug_count = 0
        
        # Create debug directory
        os.makedirs("debug_images", exist_ok=True)
    
    def detect_white_spaces(self, image):
        """Detect white spaces in the image"""
        # Create mask for white pixels
        white_mask = cv2.inRange(image, self.WHITE_LOWER, self.WHITE_UPPER)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        
        if self.debug_mode:
            cv2.imwrite(f"debug_images/white_mask_{self.debug_count}.jpg", white_mask)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours to get only large enough white spaces
        white_spaces = []
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size
            if w > self.MIN_SPACE_WIDTH and h > self.MIN_SPACE_HEIGHT:
                # Calculate the percentage of white pixels in the bounding box
                roi = white_mask[y:y+h, x:x+w]
                white_pixel_percentage = cv2.countNonZero(roi) / (w * h)
                
                # Only consider it a white space if it has enough white pixels
                if white_pixel_percentage > 0.8:  # 80% or more white pixels
                    white_spaces.append((x, y, w, h))
        
        # If not enough white spaces found, try using Canny edge detection
        if len(white_spaces) < 3:
            # Create a grayscale version
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Detect edges
            edges = cv2.Canny(blurred, 50, 150)
            
            # Invert edges to get potential spaces
            inverted_edges = cv2.bitwise_not(edges)
            
            if self.debug_mode:
                cv2.imwrite(f"debug_images/edges_{self.debug_count}.jpg", edges)
                cv2.imwrite(f"debug_images/inverted_edges_{self.debug_count}.jpg", inverted_edges)
            
            # Apply distance transform to find centers of empty spaces
            dist_transform = cv2.distanceTransform(inverted_edges, cv2.DIST_L2, 5)
            
            # Threshold the distance transform to get potential white spaces
            _, dist_thresh = cv2.threshold(dist_transform, 30, 255, cv2.THRESH_BINARY)
            dist_thresh = dist_thresh.astype(np.uint8)
            
            # Find contours in the thresholded distance transform
            dist_contours, _ = cv2.findContours(dist_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter by size
            for contour in dist_contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > self.MIN_SPACE_WIDTH and h > self.MIN_SPACE_HEIGHT:
                    # Check if this is a duplicate of an existing white space
                    is_duplicate = False
                    for ws in white_spaces:
                        ws_x, ws_y, ws_w, ws_h = ws
                        # Check for significant overlap
                        if (abs(x - ws_x) < ws_w//2 and abs(y - ws_y) < ws_h//2):
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        white_spaces.append((x, y, w, h))
        
        # If still not enough white spaces found, create some default ones
        if len(white_spaces) < 3:
            # Get screen dimensions
            height, width = image.shape[:2]
            
            # Create white spaces in various regions
            default_spaces = [
                (width - 400, 100, 300, 200),  # Top right
                (width - 400, height - 300, 300, 200),  # Bottom right
                (100, height - 300, 300, 200),  # Bottom left
                (width//2 - 150, 100, 300, 200),  # Top center
                (width//2 - 150, height - 300, 300, 200),  # Bottom center
            ]
            
            for space in default_spaces:
                if space not in white_spaces:
                    white_spaces.append(space)
        
        # Create debug visualization
        if self.debug_mode:
            debug_image = image.copy()
            for i, space in enumerate(white_spaces):
                x, y, w, h = space
                cv2.rectangle(debug_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(debug_image, f"Space {i}", (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            cv2.imwrite(f"debug_images/white_spaces_{self.debug_count}.jpg", debug_image)
            
            print(f"Detected {len(white_spaces)} white spaces")
            
        self.debug_count += 1
        return white_spaces
    
    def find_suitable_position(self, white_spaces, box_width, box_height, existing_boxes):
        """Find a suitable position within white spaces for a box"""
        # Add safety margin to box dimensions
        width_with_margin = box_width + 2 * self.SAFETY_MARGIN
        height_with_margin = box_height + 2 * self.SAFETY_MARGIN
        
        suitable_positions = []
        
        for white_space in white_spaces:
            x, y, w, h = white_space
            
            # Skip white spaces that are too small
            if w < width_with_margin or h < height_with_margin:
                continue
            
            # Try different positions within this white space
            potential_positions = []
            
            # Center position
            potential_positions.append((x + (w - box_width) // 2, y + (h - box_height) // 2))
            
            # Top-left with margin
            potential_positions.append((x + self.SAFETY_MARGIN, y + self.SAFETY_MARGIN))
            
            # Top-right with margin
            potential_positions.append((x + w - box_width - self.SAFETY_MARGIN, y + self.SAFETY_MARGIN))
            
            # Bottom-left with margin
            potential_positions.append((x + self.SAFETY_MARGIN, y + h - box_height - self.SAFETY_MARGIN))
            
            # Bottom-right with margin
            potential_positions.append((x + w - box_width - self.SAFETY_MARGIN, y + h - box_height - self.SAFETY_MARGIN))
            
            # Random positions
            for _ in range(5):
                rand_x = x + self.SAFETY_MARGIN + random.randint(0, max(0, w - width_with_margin))
                rand_y = y + self.SAFETY_MARGIN + random.randint(0, max(0, h - height_with_margin))
                potential_positions.append((rand_x, rand_y))
            
            # Check each position for overlaps with existing boxes
            for pos_x, pos_y in potential_positions:
                overlap = False
                new_box = (pos_x, pos_y, box_width, box_height)
                
                for existing_box in existing_boxes:
                    if self._boxes_overlap(new_box, existing_box):
                        overlap = True
                        break
                
                if not overlap:
                    suitable_positions.append((pos_x, pos_y))
        
        if suitable_positions:
            # Sort by distance from the center of the screen (closer is better)
            # This helps ensure more natural placement
            screen_width, screen_height = image.shape[1], image.shape[0]
            screen_center_x, screen_center_y = screen_width // 2, screen_height // 2
            
            def distance_to_center(pos):
                x, y = pos
                return ((x + box_width/2 - screen_center_x) ** 2 + 
                        (y + box_height/2 - screen_center_y) ** 2) ** 0.5
            
            suitable_positions.sort(key=distance_to_center)
            
            # Choose a position (prefer positions closer to the center)
            if len(suitable_positions) > 3:
                # Choose randomly from the top 3 positions
                return random.choice(suitable_positions[:3])
            else:
                return suitable_positions[0]
        
        # If no suitable position found, use a default position
        # (right side of the screen, which is common for CAD labels)
        screen_width, screen_height = image.shape[1], image.shape[0]
        default_x = screen_width - box_width - 50  # 50 pixels from right edge
        default_y = 100 + random.randint(0, max(0, screen_height - 300))  # Random Y in the top area
        
        return (default_x, default_y)
    
    def _boxes_overlap(self, box1, box2, threshold=5):
        """Check if two boxes overlap (with threshold to avoid touching boxes)"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Add threshold to create a buffer zone
        x1 -= threshold
        y1 -= threshold
        w1 += 2 * threshold
        h1 += 2 * threshold
        
        # Calculate the coordinates of the corners
        left1, right1 = x1, x1 + w1
        top1, bottom1 = y1, y1 + h1
        left2, right2 = x2, x2 + w2
        top2, bottom2 = y2, y2 + h2
        
        # Check for overlap
        if right1 <= left2 or right2 <= left1 or bottom1 <= top2 or bottom2 <= top1:
            return False
        return True

# Test the detector directly
if __name__ == "__main__":
    detector = WhiteSpaceDetector(debug_mode=True)
    
    print("White Space Detector Test")
    print("=======================")
    print("Options:")
    print("1. Test with screen capture")
    print("2. Test with saved image")
    
    choice = input("Enter your choice (1/2): ").strip()
    
    if choice == "1":
        print("\nCapturing screen in 5 seconds...")
        
        import pyautogui
        import time
        
        for i in range(5, 0, -1):
            print(f"{i}...")
            time.sleep(1)
        
        # Capture and analyze screen
        screenshot = pyautogui.screenshot()
        screenshot = np.array(screenshot)
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
        
        white_spaces = detector.detect_white_spaces(screenshot)
        print(f"\nDetected {len(white_spaces)} white spaces")
# whitespace_detector.py 

    elif choice == "2":
        image_path = input("Enter path to image file: ").strip()
        
        if os.path.exists(image_path):
            # Load and analyze image
            image = cv2.imread(image_path)
            white_spaces = detector.detect_white_spaces(image)
            print(f"\nDetected {len(white_spaces)} white spaces")
            print("Check debug_images folder for results")
        else:
            print(f"Error: File {image_path} not found")
    else:
        print("Invalid choice")        