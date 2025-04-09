
# Red Box Overlap Detector and Mover for MicroStation

This application automatically detects red text boxes in MicroStation drawings, identifies overlapping boxes, and moves them to appropriate white spaces.

## Features

- Detects red boxes containing text labels (e.g., "E0_BS_0084-22_21")
- Identifies when red boxes overlap with other drawing elements
- Finds suitable white spaces for relocating boxes
- Uses mouse control to drag and drop boxes to new positions
- Includes test mode for safety (preview moves without mouse movement)
- Provides visualization of detected boxes and planned movements

## Installation

1. Make sure you have Python 3.7 or newer installed

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Open MicroStation with a drawing containing red text boxes

2. Run the application:
   ```
   python main.py
   ```

3. Choose Option 2 to run the application with mouse control

4. The application will:
   - Start in TEST MODE (no actual mouse movement)
   - Detect red boxes and identify overlaps
   - Find suitable positions for overlapping boxes
   - Show planned movements in the debug images

5. When you're ready for actual movement:
   - Press 'T' to toggle test mode OFF
   - The application will now move boxes using the mouse

6. Additional keyboard commands:
   - 'P': Pause/resume the application
   - 'R': Reset box tracking (forget previously moved boxes)
   - 'Q': Quit the application

## Testing on Sample Images

If you want to test the detection without running the full application:

1. Place sample images in the 'before_images' folder
2. Run the application and choose Option 1
3. Enter the path to your sample image when prompted
4. Check the 'results' folder for visualization of the detection

## Troubleshooting

- **No red boxes detected**: Make sure MicroStation is visible and the red boxes are on screen
- **Boxes not moving**: Ensure MicroStation has focus when the application is running
- **False positives**: If UI elements are being detected as red boxes, adjust the UI filter regions
- **Mouse movement issues**: Try running with administrator privileges

## Safety Features

- Always starts in TEST MODE (no movement by default)
- Move your mouse to the top-left corner to abort (PyAutoGUI failsafe)
- Press 'P' to pause at any time
- Visualization of planned moves before execution

## System Requirements

- Windows operating system
- Python 3.7 or newer
- MicroStation with visible red text boxes
- Screen resolution of at least 1024x768
