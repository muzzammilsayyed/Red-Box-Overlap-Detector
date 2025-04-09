# box_mover.py
import pyautogui
import time
import os
import cv2
import numpy as np
import random
from red_box_detector import RedBoxDetector

class BoxMover:
    def __init__(self, debug_mode=True, test_mode=True):
        self.red_detector = RedBoxDetector(debug_mode=debug_mode)
        
        self.debug_mode = debug_mode
        self.test_mode = test_mode  # When True, just simulate moves
        self.debug_count = 0
        
        # Movement parameters
        self.MOVE_DELAY = 0.5  # Seconds between moves
        self.DRAG_DURATION = 0.5  # Seconds for drag operation
        self.CLICK_RETRIES = 3  # Number of retries if movement fails
        
        # Box tracking
        self.moved_boxes = []  # Track already moved boxes
        self.failed_boxes = []  # Track boxes that failed to move
        
        # Configure PyAutoGUI
        pyautogui.PAUSE = 0.1  # Pause between commands
        pyautogui.FAILSAFE = True  # Move to corner to abort
        
        # Create debug directory
        os.makedirs("debug_images", exist_ok=True)
    
    def move_box(self, box, target_position, retry_count=0):
        """Move a box to target position using PyAutoGUI"""
        x, y, w, h = box
        target_x, target_y = target_position
        
        # Skip if box has already been moved
        for moved_box, _ in self.moved_boxes:
            if self._is_same_box(box, moved_box):
                if self.debug_mode:
                    print(f"Box at ({x}, {y}) was already moved, skipping")
                return True
        
        # Calculate center points
        center_x = x + w // 2
        center_y = y + h // 2
        
        if self.debug_mode:
            print(f"Moving box from ({center_x}, {center_y}) to ({target_x + w//2}, {target_y + h//2})")
        
        if self.test_mode:
            print(f"TEST MODE: Would move from ({center_x}, {center_y}) to ({target_x + w//2}, {target_y + h//2})")
            self.moved_boxes.append((box, target_position))
            return True
        
        try:
            # Method 1: Standard drag (which worked in calibration testing)
            pyautogui.moveTo(center_x, center_y, duration=0.3)
            time.sleep(0.2)
            
            # Using dragTo since it worked in the calibration
            success = self._perform_drag(center_x, center_y, target_x + w//2, target_y + h//2)
            
            if success:
                # Record the successful move
                self.moved_boxes.append((box, target_position))
                return True
            else:
                # Add to failed boxes list if this is the last retry
                if retry_count >= self.CLICK_RETRIES - 1:
                    self.failed_boxes.append(box)
                return False
                
        except Exception as e:
            if self.debug_mode:
                print(f"Error moving box: {e}")
            
            # Try alternate methods if we have retries left
            if retry_count < self.CLICK_RETRIES:
                if self.debug_mode:
                    print(f"Retrying with different approach... Attempt {retry_count + 2}/{self.CLICK_RETRIES + 1}")
                
                time.sleep(1)  # Wait before retry
                return self.move_box(box, target_position, retry_count + 1)
            else:
                # Add to failed boxes list
                self.failed_boxes.append(box)
                return False
    
    def _perform_drag(self, start_x, start_y, end_x, end_y):
        """Perform drag operation with specific MicroStation behavior"""
        try:
            # Standard dragTo method (worked in calibration)
            pyautogui.dragTo(end_x, end_y, duration=1, button='left')
            return True
        except:
            # Fallback to manual drag if standard method fails
            try:
                pyautogui.moveTo(start_x, start_y, duration=0.3)
                pyautogui.mouseDown(button='left')
                time.sleep(0.3)
                
                # Move in steps
                steps = 10
                for i in range(1, steps + 1):
                    progress = i / steps
                    curr_x = start_x + (end_x - start_x) * progress
                    curr_y = start_y + (end_y - start_y) * progress
                    pyautogui.moveTo(curr_x, curr_y, duration=0.1)
                
                time.sleep(0.3)
                pyautogui.mouseUp(button='left')
                return True
            except:
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
    
    def find_suitable_position(self, box, image):
        """Find a suitable position for a box in the image"""
        height, width = image.shape[:2]
        x, y, w, h = box
        
        # Strategy 1: Try moving it to the right side of the drawing
        # (a common convention in CAD drawings)
        right_side_x = width - w - 50  # 50 pixels from the right edge
        possible_y = y  # Keep the same vertical position
        
        # Ensure position is within screen bounds
        right_side_x = max(200, min(width - w - 20, right_side_x))
        possible_y = max(100, min(height - h - 50, possible_y))
        
        return (right_side_x, possible_y)
    
    def process_screen(self):
        """Process the current screen to find and move overlapping boxes"""
        # Capture screen
        if self.debug_mode:
            print("Capturing screen...")
        
        screen = self.red_detector.capture_screen()
        
        # Detect red boxes
        red_boxes = self.red_detector.detect_boxes(screen)
        
        if self.debug_mode:
            print(f"Detected {len(red_boxes)} red boxes")
        
        if not red_boxes:
            return 0, 0
        
        # Find overlapping boxes
        overlapping_boxes = self.red_detector.find_overlapping_boxes(red_boxes, screen)
        
        if self.debug_mode:
            print(f"Found {len(overlapping_boxes)} overlapping boxes")
        
        if not overlapping_boxes:
            return 0, 0
        
        # Filter out already processed boxes
        filtered_overlapping = []
        for box in overlapping_boxes:
            # Skip if box is in failed_boxes (already tried and failed)
            if any(self._is_same_box(box, failed) for failed in self.failed_boxes):
                continue
                
            # Skip if box has already been moved
            if any(self._is_same_box(box, moved[0]) for moved in self.moved_boxes):
                continue
                
            filtered_overlapping.append(box)
        
        if len(filtered_overlapping) < len(overlapping_boxes) and self.debug_mode:
            print(f"Filtered out {len(overlapping_boxes) - len(filtered_overlapping)} already processed boxes")
        
        # Process each overlapping box
        successful_moves = 0
        
        # Create debug visualization
        if self.debug_mode:
            debug_image = screen.copy()
        
        for box in filtered_overlapping:
            x, y, w, h = box
            
            # Find suitable position
            target_position = self.find_suitable_position(box, screen)
            
            if target_position is not None:
                if self.debug_mode:
                    print(f"Found suitable position ({target_position[0]}, {target_position[1]}) for box at ({x}, {y})")
                    # Draw movement plan on debug image
                    tx, ty = target_position
                    cv2.rectangle(debug_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.rectangle(debug_image, (tx, ty), (tx+w, ty+h), (0, 255, 0), 2)
                    # Draw arrow
                    center_x, center_y = x + w//2, y + h//2
                    cv2.arrowedLine(debug_image, (center_x, center_y), 
                                   (tx + w//2, ty + h//2), (0, 255, 255), 2)
                
                # Move the box
                success = self.move_box(box, target_position)
                
                if success:
                    successful_moves += 1
            else:
                if self.debug_mode:
                    print(f"No suitable position found for box at ({x}, {y})")
        
        # Save debug image
        if self.debug_mode and filtered_overlapping:
            cv2.imwrite(f"debug_images/movement_plan_{self.debug_count}.jpg", debug_image)
            self.debug_count += 1
        
        return successful_moves, len(filtered_overlapping)
    
    def run(self, iterations=5, delay=2):
        """Run the box mover for a specified number of iterations"""
        print("Starting Box Mover")
        print(f"Test Mode: {'ON' if self.test_mode else 'OFF'}")
        print(f"Running for {iterations} iterations with {delay}s delay\n")
        
        total_moved = 0
        
        # Add countdown before starting
        print("Starting in 5 seconds...")
        for i in range(5, 0, -1):
            print(f"{i}...")
            time.sleep(1)
        
        # Run for specified iterations
        for i in range(iterations):
            print(f"\nIteration {i+1}/{iterations}")
            
            # Process screen
            moved, total = self.process_screen()
            
            if total == 0:
                print("No overlapping boxes detected")
            else:
                print(f"Moved {moved}/{total} overlapping boxes")
                total_moved += moved
            
            # Wait before next iteration
            if i < iterations - 1:
                print(f"Waiting {delay} seconds...")
                time.sleep(delay)
        
        print(f"\nFinished! Moved {total_moved} boxes in total")
        
        # Print any failures
        if self.failed_boxes:
            print(f"Failed to move {len(self.failed_boxes)} boxes")

# Test the mover directly
if __name__ == "__main__":
    mover = BoxMover(debug_mode=True, test_mode=True)
    
    print("Box Mover Test")
    print("=============")
    print("Options:")
    print("1. Run in test mode (no actual movement)")
    print("2. Run with actual movement")
    
    choice = input("Enter your choice (1/2): ").strip()
    
    if choice == "1":
        print("\nRunning in TEST MODE (no actual movement)")
        mover.run(iterations=3, delay=2)
    elif choice == "2":
        print("\nWarning: This will perform actual mouse movements!")
        confirm = input("Are you sure you want to continue? (y/n): ").strip().lower()
        
        if confirm == 'y':
            mover.test_mode = False
            print("\nRunning with ACTUAL MOVEMENT")
            mover.run(iterations=3, delay=2)
        else:
            print("Operation cancelled")
    else:
        print("Invalid choice")