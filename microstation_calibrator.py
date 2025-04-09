import cv2
import numpy as np
import pyautogui
import time
import os

class MicrostationCalibrator:
    def __init__(self):
        # Create debug directory
        os.makedirs("calibration", exist_ok=True)
        
    def capture_screen(self):
        """Capture the current screen"""
        screenshot = pyautogui.screenshot()
        screenshot = np.array(screenshot)
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
        return screenshot
    
    def analyze_colors(self):
        """Analyze colors on screen to identify red boxes"""
        print("\nColor Analysis Tool")
        print("===================")
        print("This tool will help calibrate color detection for MicroStation's red boxes.")
        print("Instructions:")
        print("1. Open MicroStation with a drawing containing red boxes")
        print("2. Position a red box in the center of your screen")
        print("3. Press Enter to capture the screen")
        input("\nPress Enter when ready...")
        
        # Capture screen
        screen = self.capture_screen()
        cv2.imwrite("calibration/screen.jpg", screen)
        
        # Create interactive window
        cv2.namedWindow("Color Calibration")
        cv2.setMouseCallback("Color Calibration", self.mouse_callback, screen)
        
        # Show image
        cv2.imshow("Color Calibration", screen)
        print("\nClick on red boxes to analyze colors.")
        print("Press 'q' to quit calibration.")
        
        # Wait for key press
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cv2.destroyAllWindows()
    
    def mouse_callback(self, event, x, y, flags, image):
        """Handle mouse clicks on the calibration image"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Get color at clicked position
            color = image[y, x]
            b, g, r = color
            
            print(f"\nClicked position: ({x}, {y})")
            print(f"BGR color: {color}")
            print(f"RGB color: [{r}, {g}, {b}]")
            
            # Extract 20x20 region around click
            size = 20
            x1, y1 = max(0, x-size), max(0, y-size)
            x2, y2 = min(image.shape[1], x+size), min(image.shape[0], y+size)
            region = image[y1:y2, x1:x2]
            
            # Analyze region colors
            avg_color = np.mean(region, axis=(0, 1))
            b_avg, g_avg, r_avg = avg_color
            
            print(f"Average BGR in region: [{b_avg:.0f}, {g_avg:.0f}, {r_avg:.0f}]")
            
            # Save region for reference
            cv2.imwrite(f"calibration/region_{x}_{y}.jpg", region)
            
            # Update color thresholds based on this analysis
            suggested_lower = [max(0, b_avg-30), max(0, g_avg-30), max(0, r_avg-30)]
            suggested_upper = [min(255, b_avg+30), min(255, g_avg+30), min(255, r_avg+30)]
            
            print(f"Suggested RED_LOWER = np.array([{suggested_lower[0]:.0f}, {suggested_lower[1]:.0f}, {suggested_lower[2]:.0f}])")
            print(f"Suggested RED_UPPER = np.array([{suggested_upper[0]:.0f}, {suggested_upper[1]:.0f}, {suggested_upper[2]:.0f}])")
    
    def test_mouse_control(self):
        """Test mouse control functionality in MicroStation"""
        print("\nMouse Control Test")
        print("=================")
        print("This test will verify if PyAutoGUI can control the mouse in MicroStation.")
        print("Instructions:")
        print("1. Open MicroStation with a drawing containing red boxes")
        print("2. Keep MicroStation in focus")
        input("\nPress Enter when ready to start test (you'll have 5 seconds to switch back to MicroStation)...")
        
        # Wait for user to switch to MicroStation
        for i in range(5, 0, -1):
            print(f"{i}...")
            time.sleep(1)
        
        current_pos = pyautogui.position()
        print(f"Current mouse position: {current_pos}")
        
        try:
            # Test 1: Simple mouse movement
            print("\nTest 1: Moving mouse to center of screen")
            screen_width, screen_height = pyautogui.size()
            center_x, center_y = screen_width // 2, screen_height // 2
            
            pyautogui.moveTo(center_x, center_y, duration=1)
            time.sleep(1)
            
            # Test 2: Click and move
            print("Test 2: Clicking and moving mouse")
            pyautogui.click()
            time.sleep(0.5)
            pyautogui.moveRel(100, 0, duration=0.5)
            time.sleep(0.5)
            
            # Test 3: Drag operation
            print("Test 3: Testing drag operation")
            pyautogui.moveTo(center_x, center_y, duration=0.5)
            time.sleep(0.5)
            
            print("  Pressing mouse down...")
            pyautogui.mouseDown()
            time.sleep(0.5)
            
            print("  Dragging...")
            pyautogui.moveTo(center_x + 100, center_y + 100, duration=1)
            time.sleep(0.5)
            
            print("  Releasing mouse...")
            pyautogui.mouseUp()
            
            print("\nMouse control tests completed.")
            print("Did the mouse move correctly in MicroStation? (y/n)")
            result = input().strip().lower()
            
            if result == 'y':
                print("Great! PyAutoGUI can control the mouse in MicroStation.")
            else:
                print("There seems to be an issue with mouse control.")
                print("Possible solutions:")
                print("1. Ensure MicroStation is in focus")
                print("2. Check if your OS has accessibility permissions enabled")
                print("3. Try running the script with administrator privileges")
        except Exception as e:
            print(f"Error during mouse control test: {e}")
        finally:
            # Return mouse to original position
            pyautogui.moveTo(current_pos)
    
    def test_drag_and_drop(self):
        """Test specific drag and drop in MicroStation"""
        print("\nDrag and Drop Test")
        print("=================")
        print("This test will simulate dragging a red box in MicroStation.")
        print("Instructions:")
        print("1. Open MicroStation with a drawing containing red boxes")
        print("2. Position a red box that you want to test moving")
        print("3. We'll attempt to drag it to a new position")
        
        # Get starting coordinates
        print("\nClick on a red box when ready...")
        start_x, start_y = pyautogui.position()
        print(f"Starting position: ({start_x}, {start_y})")
        
        # Get target coordinates
        print("Now move to where you want to drag it and click...")
        target_x, target_y = pyautogui.position()
        print(f"Target position: ({target_x}, {target_y})")
        
        # Confirm before proceeding
        print(f"\nWill drag from ({start_x}, {start_y}) to ({target_x}, {target_y})")
        confirm = input("Proceed with drag test? (y/n): ").strip().lower()
        
        if confirm != 'y':
            print("Test cancelled.")
            return
        
        print("\nPerforming drag operation in 5 seconds...")
        for i in range(5, 0, -1):
            print(f"{i}...")
            time.sleep(1)
        
        try:
            # Move to start position
            pyautogui.moveTo(start_x, start_y, duration=0.5)
            time.sleep(0.5)
            
            # Standard drag approach
            print("\nMethod 1: Standard drag...")
            pyautogui.dragTo(target_x, target_y, duration=1, button='left')
            time.sleep(2)
            
            # More detailed approach
            print("\nMethod 2: Manual drag (click, hold, move, release)...")
            pyautogui.moveTo(start_x, start_y, duration=0.5)
            time.sleep(0.5)
            pyautogui.mouseDown(button='left')
            time.sleep(0.5)
            
            # Move in steps
            steps = 10
            for i in range(1, steps + 1):
                progress = i / steps
                curr_x = start_x + (target_x - start_x) * progress
                curr_y = start_y + (target_y - start_y) * progress
                pyautogui.moveTo(curr_x, curr_y, duration=0.1)
            
            time.sleep(0.5)
            pyautogui.mouseUp(button='left')
            
            print("\nDrag tests completed.")
            print("Did either method successfully drag the box? (y/n)")
            result = input().strip().lower()
            
            if result == 'y':
                print("Great! We can move boxes in MicroStation.")
                method = input("Which method worked? (1 or 2): ").strip()
                print(f"We'll use Method {method} in our implementation.")
            else:
                print("Troubleshooting needed for mouse drag operations.")
                print("Suggestions:")
                print("1. Try clicking first to ensure focus")
                print("2. Try slower drag operations")
                print("3. Make sure MicroStation allows dragging of these elements")
        except Exception as e:
            print(f"Error during drag test: {e}")

if __name__ == "__main__":
    calibrator = MicrostationCalibrator()
    
    while True:
        print("\nMicroStation Calibration Tool")
        print("============================")
        print("1. Analyze Red Box Colors")
        print("2. Test Mouse Control")
        print("3. Test Drag and Drop")
        print("q. Quit")
        
        choice = input("\nEnter your choice: ").strip().lower()
        
        if choice == '1':
            calibrator.analyze_colors()
        elif choice == '2':
            calibrator.test_mouse_control()
        elif choice == '3':
            calibrator.test_drag_and_drop()
        elif choice == 'q':
            break
        else:
            print("Invalid choice. Please try again.")