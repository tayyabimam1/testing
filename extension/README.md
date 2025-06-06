# DeepSight Browser Extension

A browser extension that detects AI-manipulated (deepfake) videos by analyzing video frames using a deep learning model.

## Setup Instructions

### 1. Install Python Requirements

Make sure you have Python 3.7+ installed. Then install the required packages:

```bash
pip install fastapi uvicorn torch torchvision numpy opencv-python face_recognition Pillow
```

### 2. Start the Python Backend

- Double-click `start_server.bat` file to start the backend server
- Or run manually: `python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload`
- The server should start at http://localhost:8000

### 3. Load the Extension in Chrome

1. Open Chrome and go to `chrome://extensions/`
2. Enable "Developer mode" (toggle in the top right)
3. Click "Load unpacked" and select the extension folder
4. The DeepSight extension should now appear in your extensions list

## Usage

1. Navigate to a website with videos (YouTube, Instagram, TikTok, etc.)
2. Click the DeepSight extension icon in your browser toolbar
3. If a video is detected, you'll see a preview in the popup
4. Click "Detect DeepFake" to analyze the video
5. Results will be displayed in the popup, showing whether the video is likely AI-generated or authentic

## How It Works

1. The extension captures frames from videos on the webpage
2. These frames are sent to the Python backend for analysis
3. The backend uses a ResNext-based deep learning model to analyze the frames
4. Results are sent back to the extension and displayed in the popup

## Troubleshooting

- **Server Not Running**: Make sure the Python backend is running before using the extension. A green dot in the popup indicates the server is running.
- **No Video Detected**: Try refreshing the page or clicking directly on a video to select it.
- **Analysis Fails**: Check the console for any error messages.

## Files

- `main.py`: Python backend with deepfake detection model
- `content.js`: Content script that interacts with web pages
- `popup.html/js`: Extension popup interface
- `background.js`: Background script for extension functionality
- `manifest.json`: Extension configuration 