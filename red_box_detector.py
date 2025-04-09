import cv2
import numpy as np
from PIL import ImageGrab
import os
import time

class RedBoxDetector:
    def __init__(self, debug_mode=True):
        # Default color thresholds for red boxes (BGR format)
        # These are based on typical MicroStation red box colors
        self.RED_LOWER = np.array([0, 0, 150])
        self.RED_UPPER = np.array([100, 100, 255])
        
        # Size filters
        self.MIN_BOX_WIDTH = 20
        self.MIN_BOX_HEIGHT = 10
        self.MAX_BOX_WIDTH = 500
        self.MAX_BOX_HEIGHT = 100
        
        # Aspect ratio constraints for text boxes
        self.MIN_ASPECT_RATIO = 1.5  # Boxes are typically wider than tall
        self.MAX_ASPECT_RATIO = 15.0  # Allow for very wide boxes
        
        # Overlap detection parameters
        self.OVERLAP_THRESHOLD = 0.05
        self.PROXIMITY_THRESHOLD = 10  # Pixels
        
        # Debug settings
        self.debug_mode = debug_mode
        self.debug_count = 0
        
        # Create debug directory
        os.makedirs("debug_images", exist_ok=True)
    
    def capture_screen(self):
        """Capture the screen and convert to OpenCV format"""
        screenshot = ImageGrab.grab()
        screenshot = np.array(screenshot)
        # Convert RGB to BGR (OpenCV format)
        screenshot_bgr = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
        
        if self.debug_mode:
            cv2.imwrite(f"debug_images/screenshot_{self.debug_count}.jpg", screenshot_bgr)
            self.debug_count += 1
        
        return screenshot_bgr
    
    def detect_boxes(self, image):
        """Detect red boxes using multiple methods and combine results"""
        # Method 1: Standard color-based detection
        color_boxes = self._detect_by_color(image)
        
        # Method 2: Contour-based detection
        contour_boxes = self._detect_by_contours(image)
        
        # Method 3: Text label pattern detection
        text_boxes = self._detect_by_text_pattern(image)
        
        # Combine results, avoiding duplicates
        all_boxes = []
        
        # First add color-detected boxes
        for box in color_boxes:
            all_boxes.append(box)
        
        # Add unique contour boxes
        for box in contour_boxes:
            if not self._is_duplicate(box, all_boxes):
                all_boxes.append(box)
        
        # Add unique text boxes
        for box in text_boxes:
            if not self._is_duplicate(box, all_boxes):
                all_boxes.append(box)
        
        # Filter out UI elements and non-MicroStation boxes
        filtered_boxes = self.filter_ui_elements(all_boxes, image)
        
        if self.debug_mode:
            debug_image = image.copy()
            
            # Draw original boxes
            for i, box in enumerate(all_boxes):
                x, y, w, h = box
                cv2.rectangle(debug_image, (x, y), (x+w, y+h), (255, 0, 0), 1)  # Blue for all detected
            
            # Draw filtered boxes
            for i, box in enumerate(filtered_boxes):
                x, y, w, h = box
                cv2.rectangle(debug_image, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Green for filtered
                cv2.putText(debug_image, f"{i}: {w}x{h}", (x, y-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.imwrite(f"debug_images/detection_{self.debug_count}.jpg", debug_image)
            print(f"Detected {len(filtered_boxes)} red boxes (from {len(all_boxes)} candidates)")
        
        return filtered_boxes
    
    def _detect_by_color(self, image):
        """Detect red boxes using color thresholds"""
        # Create mask for red pixels
        red_mask = cv2.inRange(image, self.RED_LOWER, self.RED_UPPER)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        
        if self.debug_mode:
            cv2.imwrite(f"debug_images/red_mask_{self.debug_count}.jpg", red_mask)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours to get only rectangular shapes (red boxes)
        red_boxes = []
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size
            if (self.MIN_BOX_WIDTH <= w <= self.MAX_BOX_WIDTH and 
                self.MIN_BOX_HEIGHT <= h <= self.MAX_BOX_HEIGHT):
                
                # Calculate the percentage of red pixels in the bounding box
                roi = red_mask[y:y+h, x:x+w]
                red_pixel_percentage = cv2.countNonZero(roi) / (w * h)
                
                # Only consider it a red box if it has enough red pixels
                if red_pixel_percentage > 0.2:
                    aspect_ratio = w / h
                    if self.MIN_ASPECT_RATIO <= aspect_ratio <= self.MAX_ASPECT_RATIO:
                        red_boxes.append((x, y, w, h))
                        
                        if self.debug_mode:
                            print(f"Color detected box at ({x}, {y}), size {w}x{h}, red %: {red_pixel_percentage:.2f}")
        
        return red_boxes
    
    def _detect_by_contours(self, image):
        """Detect red boxes using contour detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY_INV)
        
        if self.debug_mode:
            cv2.imwrite(f"debug_images/binary_{self.debug_count}.jpg", binary)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours to get only rectangular shapes
        boxes = []
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size
            if (self.MIN_BOX_WIDTH <= w <= self.MAX_BOX_WIDTH and 
                self.MIN_BOX_HEIGHT <= h <= self.MAX_BOX_HEIGHT):
                
                # Check aspect ratio
                aspect_ratio = w / h
                if self.MIN_ASPECT_RATIO <= aspect_ratio <= self.MAX_ASPECT_RATIO:
                    # Check if it contains red pixels
                    roi = image[y:y+h, x:x+w]
                    red_mask = cv2.inRange(roi, self.RED_LOWER, self.RED_UPPER)
                    red_percentage = cv2.countNonZero(red_mask) / (w * h)
                    
                    if red_percentage > 0.15:  # Lower threshold for contour detection
                        boxes.append((x, y, w, h))
                        
                        if self.debug_mode:
                            print(f"Contour detected box at ({x}, {y}), size {w}x{h}, red %: {red_percentage:.2f}")
        
        return boxes
    
    def _detect_by_text_pattern(self, image):
        """Detect red boxes containing text label patterns (like E0_BS)"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        if self.debug_mode:
            cv2.imwrite(f"debug_images/text_thresh_{self.debug_count}.jpg", thresh)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by aspect ratio (labels are typically wider than tall)
            aspect_ratio = w / h
            if self.MIN_ASPECT_RATIO < aspect_ratio < self.MAX_ASPECT_RATIO and 10 < h < 40:
                # Check if there's red color nearby
                expanded_x = max(0, x - 10)
                expanded_y = max(0, y - 10)
                expanded_w = min(image.shape[1] - expanded_x, w + 20)
                expanded_h = min(image.shape[0] - expanded_y, h + 20)
                
                roi = image[expanded_y:expanded_y+expanded_h, expanded_x:expanded_x+expanded_w]
                
                # Check for red pixels in the ROI
                red_mask = cv2.inRange(roi, self.RED_LOWER, self.RED_UPPER)
                red_percentage = cv2.countNonZero(red_mask) / (expanded_w * expanded_h)
                
                if red_percentage > 0.1:  # At least 10% red pixels
                    text_boxes.append((expanded_x, expanded_y, expanded_w, expanded_h))
                    
                    if self.debug_mode:
                        print(f"Text detected box at ({expanded_x}, {expanded_y}), size {expanded_w}x{expanded_h}")
        
        return text_boxes
    
    def _is_duplicate(self, new_box, existing_boxes, tolerance=20):
        """Check if a box is a duplicate of any in the existing boxes list"""
        for box in existing_boxes:
            # Check if centers are close
            if self._is_same_box(new_box, box, tolerance):
                return True
        return False
    
    def _is_same_box(self, box1, box2, tolerance=20):
        """Check if two boxes are likely the same box"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate centers
        center1_x, center1_y = x1 + w1//2, y1 + h1//2
        center2_x, center2_y = x2 + w2//2, y2 + h2//2
        
        # Check if centers are close
        return (abs(center1_x - center2_x) < tolerance and 
                abs(center1_y - center2_y) < tolerance)
    
    # In the filter_ui_elements method, make this change:

    def filter_ui_elements(self, boxes, image):
        """Filter out UI elements like buttons, tabs, etc."""
        filtered_boxes = []
        height, width = image.shape[:2]
        
        # Define UI regions to exclude (typical locations of UI elements)
        ui_regions = [
            (0, 0, width, 50),               # Top bar (reduce height from 100 to 50)
            (0, 0, 180, height),             # Left sidebar (reduce width from 200 to 180)
            (width-30, 0, 30, height),       # Right edge (reduce width from 50 to 30)
            (0, height-30, width, 30)        # Bottom edge (reduce height from 50 to 30)
        ]
        
        for box in boxes:
            x, y, w, h = box
            
            # Check if box is part of a CAD label (typically contains "E0_" or similar)
            is_cad_label = True  # Default to True to be more permissive with box detection
            
            # Skip if it's in a UI region and not a CAD label
            in_ui_region = False
            for ui_x, ui_y, ui_w, ui_h in ui_regions:
                if (x >= ui_x and y >= ui_y and 
                    x + w <= ui_x + ui_w and y + h <= ui_y + ui_h):
                    # Box is entirely in UI region
                    if not is_cad_label:
                        in_ui_region = True
                        break
            
            if not in_ui_region:
                filtered_boxes.append(box)
        
        if self.debug_mode:
            print(f"Filtered out {len(boxes) - len(filtered_boxes)} UI elements")
        
        return filtered_boxes
    
    def is_overlapping(self, box1, box2):
        """Check if two boxes overlap"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate the coordinates of the corners
        left1, right1 = x1, x1 + w1
        top1, bottom1 = y1, y1 + h1
        left2, right2 = x2, x2 + w2
        top2, bottom2 = y2, y2 + h2
        
        # Check for overlap
        if right1 <= left2 or right2 <= left1 or bottom1 <= top2 or bottom2 <= top1:
            return False
        
        # Calculate overlap area
        overlap_width = min(right1, right2) - max(left1, left2)
        overlap_height = min(bottom1, bottom2) - max(top1, top2)
        overlap_area = overlap_width * overlap_height
        
        # Calculate minimum box area
        area1 = w1 * h1
        area2 = w2 * h2
        min_area = min(area1, area2)
        
        # Return True if overlap area is significant
        return overlap_area / min_area > self.OVERLAP_THRESHOLD
    
    def is_too_close(self, box1, box2):
        """Check if two boxes are too close to each other"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate centers
        center1_x, center1_y = x1 + w1//2, y1 + h1//2
        center2_x, center2_y = x2 + w2//2, y2 + h2//2
        
        # Calculate distance between centers
        distance = np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
        
        # Calculate minimum safe distance (sum of half widths and heights plus threshold)
        safe_distance = (w1 + w2) / 4 + (h1 + h2) / 4 + self.PROXIMITY_THRESHOLD
        
        # Consider boxes too close if distance is less than the safe distance
        return distance < safe_distance
    
    # In the find_overlapping_boxes method, make this change:

    def find_overlapping_boxes(self, red_boxes, image=None):
        """Find boxes that overlap with other boxes or are too close"""
        overlapping = []
        
        # First check for physical overlaps
        for i, box1 in enumerate(red_boxes):
            for j, box2 in enumerate(red_boxes):
                if i != j and self.is_overlapping(box1, box2):
                    if box1 not in overlapping:
                        overlapping.append(box1)
        
        # Then check for boxes that are too close
        if len(overlapping) == 0 and len(red_boxes) > 1:
            for i, box1 in enumerate(red_boxes):
                for j, box2 in enumerate(red_boxes):
                    if i != j and self.is_too_close(box1, box2):
                        if box1 not in overlapping:
                            overlapping.append(box1)
        
        # If still no overlaps, check for boxes that might be overlapping with drawing elements
        # This is a heuristic that can help move isolated boxes that are placed poorly
        if len(overlapping) == 0 and len(red_boxes) > 0:
            # Consider small boxes or boxes near the edges as potentially overlapping
            for box in red_boxes:
                x, y, w, h = box
                height, width = image.shape[:2] if image is not None else (1080, 1920)  # Default values
                
                # If the box is near the edge of the screen
                if x < 100 or y < 100 or x + w > width - 100 or y + h > height - 100:
                    overlapping.append(box)
                    if self.debug_mode:
                        print(f"Box at ({x},{y}) is near the edge - marking as overlapping")
                    continue
                
                # If the box is very small, it might need to be moved to be more visible
                if w < 30 or h < 15:
                    overlapping.append(box)
                    if self.debug_mode:
                        print(f"Box at ({x},{y}) is small - marking as overlapping")
                    continue
        
        if self.debug_mode:
            if image is not None:
                debug_image = image.copy()
                
                # Draw all boxes
                for box in red_boxes:
                    x, y, w, h = box
                    # Overlapping boxes in red, others in green
                    color = (0, 0, 255) if box in overlapping else (0, 255, 0)
                    cv2.rectangle(debug_image, (x, y), (x+w, y+h), color, 2)
                
                cv2.imwrite(f"debug_images/overlaps_{self.debug_count}.jpg", debug_image)
            
            print(f"Found {len(overlapping)} overlapping boxes out of {len(red_boxes)} total")
        
        return overlapping

# Test the detector directly
if __name__ == "__main__":
    detector = RedBoxDetector(debug_mode=True)
    
    print("Red Box Detector Test")
    print("====================")
    print("Options:")
    print("1. Test with screen capture")
    print("2. Test with saved image")
    
    choice = input("Enter your choice (1/2): ").strip()
    
    if choice == "1":
        print("\nCapturing screen in 5 seconds...")
        for i in range(5, 0, -1):
            print(f"{i}...")
            time.sleep(1)
        
        # Capture and analyze screen
        screen = detector.capture_screen()
        red_boxes = detector.detect_boxes(screen)
        overlapping_boxes = detector.find_overlapping_boxes(red_boxes, screen)
        
        print(f"\nDetected {len(red_boxes)} red boxes")
        print(f"Found {len(overlapping_boxes)} overlapping boxes")
        print("Check debug_images folder for results")
    
    elif choice == "2":
        image_path = input("Enter path to image file: ").strip()
        
        if os.path.exists(image_path):
            # Load and analyze image
            image = cv2.imread(image_path)
            red_boxes = detector.detect_boxes(image)
            overlapping_boxes = detector.find_overlapping_boxes(red_boxes, image)
            
            print(f"\nDetected {len(red_boxes)} red boxes")
            print(f"Found {len(overlapping_boxes)} overlapping boxes")
            print("Check debug_images folder for results")
        else:
            print(f"Error: File {image_path} not found")
    else:
        print("Invalid choice")