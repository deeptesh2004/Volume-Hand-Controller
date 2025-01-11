# Volume Hand Controller  
The Volume Hand Controller is a Python-based project that utilizes the [cv2 (OpenCV)](https://github.com/opencv/opencv-python) 
and [mediapipe](https://github.com/google-ai-edge/mediapipe) libraries to capture hand gestures in real-time via a webcam. 
The system detects the position of the thumb and index finger, and based on their distance, it adjusts the audio volume. 
This gesture-based control provides an intuitive, touch-free way to manage volume levels, 
offering a seamless integration of computer vision and gesture recognition.  

## Landmarks
<div align="center">
  <img src="./resources/hand_landmarks.png" alt="Landmarks image" width="600"/>
</div>  

## Virtual environment  
To keep your project dependencies isolated and organized, it's recommended to use a virtual environment. 
This ensures that your system's global Python packages remain unaffected and that your project has 
the exact versions of libraries it needs, reducing the risk of version conflicts  
- Setup
  ```bash
  python -m venv venv_name
  ```
  
- Activation
  - Windows  
    ```bash
    .\venv_name\Scripts\activate
    ```
    
  - macOS / Linux
    ```bash
    source venv_name/bin/activate
    ```
## Install dependencies  
Install required dependencies for the project
```bash
pip install -r requirements.txt
```
If you experience an error `Failed building wheel for opencv-python` while installing `opencv-python` this might solve the problem for you:
- ```bash
  python -m pip install --upgrade pip setuptools wheel
  ```
- ```bash
  pip install -r requirements.txt
  ```

## Arguments 
- `-r` or `--reset-volume` - Restores the system volume to what it was before the script started

For all available arguments use `python main.py --help`
