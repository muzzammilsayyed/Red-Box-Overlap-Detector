# main.py
import os
import time
import traceback
import keyboard
import cv2
import numpy as np
from red_box_detector import RedBoxDetector
from whitespace_detector import WhiteSpaceDetector
from box_mover import BoxMover

class RedBoxMovingApp:
    def __init__(self):
        # Create component instances
        self.red_detector = RedBoxDetector(debug_mode=True)
        self.white_detector = WhiteSpaceDetector(debug_mode=True)
        self.box_mover = BoxMover(debug_mode=True, test_mode=True)
        
        # Application state
        self.running = False
        self.paused = False
        
        # Create directories
        os.makedirs("debug_images", exist_ok=True)
        os.makedirs("results", exist_ok=True)
    
    def setup_keyboard_handlers(self):
        """Set up keyboard event handlers"""
        keyboard.add_hotkey('t', self.toggle_test_mode)
        keyboard.add_hotkey('p', self.toggle_pause)
        keyboard.add_hotkey('q', self.quit_app)
        keyboard.add_hotkey('r', self.reset_tracking)
    
    def toggle_test_mode(self):
        """Toggle test mode on/off"""
        self.box_mover.test_mode = not self.box_mover.test_mode
        print(f"Test mode: {'ON' if self.box_mover.test_mode else 'OFF'}")
    
    def toggle_pause(self):
        """Toggle pause state"""
        self.paused = not self.paused
        print(f"{'Paused' if self.paused else 'Resumed'}")
    
    def quit_app(self):
        """Quit the application"""
        self.running = False
        print("Quitting...")
    
    def reset_tracking(self):
        """Reset box tracking"""
        self.box_mover.moved_boxes = []
        self.box_mover.failed_boxes = []
        print("Reset box tracking - all boxes will be considered as new")
    
    def process_screen(self):
        """Process the current screen to detect and move boxes"""
        try:
            # Capture the screen
            screen = self.red_detector.capture_screen()
            
            # Detect red boxes
            red_boxes = self.red_detector.detect_boxes(screen)
            
            if not red_boxes:
                print("No red boxes detected")
                return 0, 0
            
            # Find overlapping boxes
            overlapping_boxes = self.red_detector.find_overlapping_boxes(red_boxes, screen)
            
            if not overlapping_boxes:
                print("No overlapping boxes detected")
                return 0, 0
            
            # Detect white spaces
            white_spaces = self.white_detector.detect_white_spaces(screen)
            
            # Filter out already processed boxes
            filtered_overlapping = []
            for box in overlapping_boxes:
                # Skip if box is in failed_boxes
                if any(self.box_mover._is_same_box(box, failed) for failed in self.box_mover.failed_boxes):
                    continue
                
                # Skip if box has already been moved
                if any(self.box_mover._is_same_box(box, moved[0]) for moved in self.box_mover.moved_boxes):
                    continue
                
                filtered_overlapping.append(box)
            
            if not filtered_overlapping:
                print("All overlapping boxes have been processed")
                return 0, 0
            
            # Create debug visualization
            debug_image = screen.copy()
            
            # Create a list of existing (non-overlapping) boxes
            non_overlapping = [box for box in red_boxes if box not in overlapping_boxes]
            
            # Add boxes that have already been moved to their new positions
            for moved_box, target_pos in self.box_mover.moved_boxes:
                non_overlapping.append((target_pos[0], target_pos[1], moved_box[2], moved_box[3]))
            
            # Process each overlapping box
            successful_moves = 0
            target_positions = {}
            
            for box in filtered_overlapping:
                x, y, w, h = box
                
                # Find suitable position
                target_position = self.white_detector.find_suitable_position(
                    white_spaces, w, h, non_overlapping
                )
                
                if target_position:
                    print(f"Found suitable position at {target_position} for box at ({x},{y})")
                    target_positions[box] = target_position
                    
                    # Move the box if we're running
                    if self.running:
                        success = self.box_mover.move_box(box, target_position)
                        if success:
                            successful_moves += 1
                            # Add to non-overlapping to avoid placing other boxes here
                            non_overlapping.append(
                                (target_position[0], target_position[1], w, h)
                            )
                else:
                    print(f"No suitable position found for box at ({x},{y})")
            
            # Draw the debug visualization
            # First draw white spaces
            for ws in white_spaces:
                x, y, w, h = ws
                cv2.rectangle(debug_image, (x, y), (x+w, y+h), (255, 0, 0), 1)
            
            # Draw all boxes
            for box in red_boxes:
                x, y, w, h = box
                color = (0, 0, 255) if box in overlapping_boxes else (0, 255, 0)
                cv2.rectangle(debug_image, (x, y), (x+w, y+h), color, 2)
            
            # Draw movement arrows
            for box, pos in target_positions.items():
                x, y, w, h = box
                tx, ty = pos
                
                # Draw arrow from center of box to target position
                center_x, center_y = x + w//2, y + h//2
                cv2.arrowedLine(debug_image, (center_x, center_y), 
                              (tx + w//2, ty + h//2), (0, 255, 255), 2)
                
                # Draw the box at the target position
                cv2.rectangle(debug_image, (tx, ty), (tx+w, ty+h), (0, 255, 255), 2)
            
            # Save the debug image
            timestamp = int(time.time())
            cv2.imwrite(f"results/movement_plan_{timestamp}.jpg", debug_image)
            
            return successful_moves, len(filtered_overlapping)
            
        except Exception as e:
            print(f"Error in process_screen: {e}")
            traceback.print_exc()
            return 0, 0
    
    # In the main.py file, update the run method:

    def run(self):
        """Run the main application loop"""
        self.running = True
        iteration = 0
        
        try:
            # Set up keyboard handlers
            self.setup_keyboard_handlers()
            
            print("\nRed Box Overlap Detector and Mover")
            print("=================================")
            print("Keyboard controls:")
            print("  t - Toggle test mode (currently ON)")
            print("  p - Pause/resume")
            print("  r - Reset box tracking")
            print("  q - Quit")
            print("\nStarting in TEST MODE (no actual movement)")
            print("Press 't' to toggle test mode OFF when ready for actual movement")
            
            # Countdown before starting
            print("\nStarting in 5 seconds...")
            for i in range(5, 0, -1):
                print(f"{i}...")
                time.sleep(1)
            
            # Main loop
            while self.running:
                if not self.paused:
                    iteration += 1
                    print(f"\n=== Iteration {iteration} ===")
                    print(f"Test mode: {'ON' if self.box_mover.test_mode else 'OFF'}")
                    
                    # Process screen
                    moved, total = self.process_screen()
                    
                    if total > 0:
                        print(f"Moved {moved}/{total} overlapping boxes")
                    
                    # Check if we need to continue
                    if total == 0 and iteration > 1:
                        print("No more overlapping boxes to process.")
                        if len(self.box_mover.moved_boxes) > 0:
                            print(f"Successfully moved {len(self.box_mover.moved_boxes)} boxes.")
                        if len(self.box_mover.failed_boxes) > 0:
                            print(f"Failed to move {len(self.box_mover.failed_boxes)} boxes.")
                        
                        # Ask if user wants to continue scanning
                        print("\nContinue scanning for more overlapping boxes? (y/n)")
                        # Use a different method for input to avoid blocking
                        continue_scanning = input().strip().lower()
                        if continue_scanning != 'y':
                            break
                    
                    # Wait before next scan
                    print("Waiting 3 seconds before next scan...")
                    for i in range(30):  # 30 x 0.1s = 3 seconds
                        if not self.running or self.paused:
                            break
                        time.sleep(0.1)
                else:
                    # When paused, just wait a bit
                    print("Paused. Press 'p' to resume.")
                    time.sleep(1)
        
        except KeyboardInterrupt:
            print("\nProgram interrupted by user (Ctrl+C)")
        except Exception as e:
            print(f"\nUnexpected error: {e}")
            traceback.print_exc()
        finally:
            print("\nProgram terminated.")
            # Print summary
            if hasattr(self, 'box_mover'):
                if self.box_mover.moved_boxes:
                    print(f"Successfully moved {len(self.box_mover.moved_boxes)} boxes")
                if self.box_mover.failed_boxes:
                    print(f"Failed to move {len(self.box_mover.failed_boxes)} boxes")

def test_on_sample_image(image_path):
    """Run a complete test on a sample image"""
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found")
        return
    
    # Create components
    red_detector = RedBoxDetector(debug_mode=True)
    white_detector = WhiteSpaceDetector(debug_mode=True)
    
    # Load image
    image = cv2.imread(image_path)
    
    # Detect red boxes
    red_boxes = red_detector.detect_boxes(image)
    print(f"Detected {len(red_boxes)} red boxes")
    
    # Find overlapping boxes
    overlapping_boxes = red_detector.find_overlapping_boxes(red_boxes, image)
    print(f"Found {len(overlapping_boxes)} overlapping boxes")
    
    # Detect white spaces
    white_spaces = white_detector.detect_white_spaces(image)
    print(f"Detected {len(white_spaces)} white spaces")
    
    # Create visualization
    non_overlapping = [box for box in red_boxes if box not in overlapping_boxes]
    target_positions = {}
    
    # Find suitable positions for overlapping boxes
    for box in overlapping_boxes:
        x, y, w, h = box
        target_position = white_detector.find_suitable_position(
            white_spaces, w, h, non_overlapping
        )
        
        if target_position:
            print(f"Found suitable position at {target_position} for box at ({x},{y})")
            target_positions[box] = target_position
            # Add to non-overlapping to avoid placing other boxes here
            non_overlapping.append((target_position[0], target_position[1], w, h))
        else:
            print(f"No suitable position found for box at ({x},{y})")
    
    # Create debug visualization
    debug_image = image.copy()
    
    # Draw white spaces
    for ws in white_spaces:
        x, y, w, h = ws
        cv2.rectangle(debug_image, (x, y), (x+w, y+h), (255, 0, 0), 1)
    
    # Draw all boxes
    for box in red_boxes:
        x, y, w, h = box
        color = (0, 0, 255) if box in overlapping_boxes else (0, 255, 0)
        cv2.rectangle(debug_image, (x, y), (x+w, y+h), color, 2)
    
    # Draw movement arrows
    for box, pos in target_positions.items():
        x, y, w, h = box
        tx, ty = pos
        
        # Draw arrow from center of box to target position
        center_x, center_y = x + w//2, y + h//2
        cv2.arrowedLine(debug_image, (center_x, center_y), 
                      (tx + w//2, ty + h//2), (0, 255, 255), 2)
        
        # Draw the box at the target position
        cv2.rectangle(debug_image, (tx, ty), (tx+w, ty+h), (0, 255, 255), 2)
    
    # Save the debug image
    os.makedirs("results", exist_ok=True)
    out_path = f"results/test_result_{os.path.basename(image_path)}"
    cv2.imwrite(out_path, debug_image)
    print(f"Test results saved to {out_path}")

def show_menu():
    """Show the main menu"""
    os.system('cls' if os.name == 'nt' else 'clear')
    print("Red Box Overlap Detector and Mover")
    print("=================================")
    print("1. Test on a sample image")
    print("2. Run the application (with actual mouse control)")
    print("3. Help and information")
    print("q. Quit")
    print()

def show_help():
    """Show help information"""
    os.system('cls' if os.name == 'nt' else 'clear')
    print("Red Box Overlap Detector and Mover - Help")
    print("=======================================")
    print("\nPurpose:")
    print("This application detects red boxes in MicroStation that")
    print("overlap with other elements, and moves them to suitable empty spaces.")
    print("\nHow to use:")
    print("1. Make sure MicroStation is open with the drawing visible")
    print("2. Run option 2 from the main menu")
    print("3. The program starts in TEST MODE (no actual movement)")
    print("4. Press 't' to toggle test mode OFF when you want actual movement")
    print("5. Press 'q' to quit when finished")
    print("\nKeyboard controls:")
    print("  t - Toggle test mode on/off")
    print("  p - Pause/resume the application")
    print("  r - Reset box tracking (forget moved/failed boxes)")
    print("  q - Quit the application")
    print("\nPress Enter to return to the menu...")
    input()

def main():
    """Main function"""
    while True:
        show_menu()
        choice = input("Enter your choice: ").strip().lower()
        
        if choice == '1':
            image_path = input("Enter path to sample image: ").strip()
            test_on_sample_image(image_path)
            print("\nPress Enter to continue...")
            input()
        
        elif choice == '2':
            print("\nWarning: This will perform actual mouse movements when test mode is disabled!")
            confirm = input("Are you sure you want to continue? (y/n): ").strip().lower()
            
            if confirm == 'y':
                app = RedBoxMovingApp()
                app.run()
        
        elif choice == '3':
            show_help()
        
        elif choice == 'q':
            print("Exiting...")
            break
        
        else:
            print("Invalid choice. Please try again.")
            time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Unhandled error: {e}")
        traceback.print_exc()
        print("\nThe program encountered an error and must exit.")
        print("Press Enter to continue...")
        input()