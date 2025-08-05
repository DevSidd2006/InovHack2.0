# Real-Time AI Video Analytics Dashboard

This project is a real-time video analytics dashboard using YOLOv10 and LLM (Large Language Model) integration, built with Flask and Ultralytics YOLO. It provides live object detection, analytics, and AI-powered insights from any video source (webcam, IP camera, or video file).

## Images

<img width="2822" height="1572" alt="Screenshot 2025-07-27 191742" src="https://github.com/user-attachments/assets/3e739338-7579-48fa-ac79-dc72e597cf71" />
<img width="2320" height="1256" alt="Screenshot 2025-07-27 191812" src="https://github.com/user-attachments/assets/eaf775be-d494-46a4-a3ab-e225f1249e64" />



## Features
- **Live Video Feed**: View real-time video from webcam, IP stream, or video file.
- **Object Detection**: Uses YOLOv10 for detecting people, vehicles, weapons, and more.
- **Analytics Panels**: Live stats for queue, crowd, weapons, traffic, and parking.
- **Detection History**: Per-second detection graphs for all features.
- **LLM Insights**: AI-generated summaries and Q&A about the current video feed and analytics.
- **Alerts**: Real-time alerts for crowding, weapons, and traffic events.
- **Modern Dashboard UI**: Responsive, dark-mode enabled, with sidebar navigation.

## Getting Started

### Prerequisites
- Python 3.8+
- pip
- (Recommended) A CUDA-capable GPU for best YOLO performance

### Installation
1. **Clone the repository**
   ```sh
   git clone <your-repo-url>
   cd InovHack
   ```
2. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```
   (If `requirements.txt` is missing, install manually:)
   ```sh
   pip install flask opencv-python ultralytics torch requests
   ```
3. **Download YOLOv10 model weights**
   - Download `yolov10n.pt` from the [Ultralytics YOLOv10 release page](https://github.com/ultralytics/ultralytics/releases) and place it in the project root.

4. **Set your OpenRouter API key**
   - Get a free API key from [OpenRouter](https://openrouter.ai/).
   - Set it as an environment variable:
     ```sh
     set OPENROUTER_API_KEY=sk-or-...
     ```

### Running the App
```sh
python app.py
```
- The dashboard will be available at [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Usage
- **Switch Video Source**: Use the dropdown above the video feed to select webcam (Live Camera) or enter a video/IP stream URL.
- **View Analytics**: Use the sidebar to switch between Queue, Crowd, Weapons, Traffic, and Parking panels.
- **Ask the AI**: Use the LLM Insights panel to ask questions about the current analytics.

## File Structure
- `app.py` - Main Flask backend, YOLO/LLM logic, API endpoints
- `templates/dashboard.html` - Dashboard UI (Bootstrap 5, Chart.js)
- `yolov8n.pt` or `yolov10n.pt` - YOLO model weights
- `requirements.txt` - Python dependencies

## Notes
- For webcam, ensure your device has a camera and OpenCV can access it (index 0).
- For video files or IP streams, enter the full path or URL in the dashboard.
- LLM features require a valid OpenRouter API key and internet access.

## License
MIT License

---
Made for InovHack 2.0 by DevSidd2006
