## YOGA AI: Real-Time Pose Detection

This project is a real-time human pose detection application that uses MediaPipe and OpenCV to detect various yoga poses, including a complete guided Surya Namaskar sequence.

## Technologies Used

- **Language:** Python  
- **Libraries:**  
  - `OpenCV` – for video capture and drawing interface  
  - `MediaPipe` – for pose estimation  
  - `NumPy` – for numerical calculations  
  - `textwrap` – for formatting dynamic messages

## Features

- **Real-Time Pose Detection** via webcam.
- **Supports multiple poses** including:
  - T-Pose
  - Bicep Curl
  - Tricep Stretch
  - Tree Pose
  - Hands Up
  - Surya Namaskar (12 steps)
- **Visual Feedback**:  
  - Instructions displayed on screen.  
  - Dynamic pose validation feedback ("Perfect!" / "Incorrect Pose!").  
  - Pose step descriptions shown during Surya Namaskar.  
- **Fullscreen Window** for immersive interaction.
- **Keyboard Selection**: Select poses using number keys `1` to `6`.

## Pose Options

| Key | Pose Name               |
|-----|-------------------------|
| 1   | T-Pose                  |
| 2   | Bicep Curl              |
| 3   | Tricep Stretch          |
| 4   | Tree Pose               |
| 5   | Hands Up                |
| 6   | Surya Namaskar (Guided) |

## Surya Namaskar Steps

1. Pranamasana (Prayer Pose)  
2. Hasta Uttanasana (Raised Arms Pose)  
3. Padahastasana (Hand to Foot Pose)  
4. Ashwa Sanchalanasana (Equestrian Pose)  
5. Dandasana (Stick Pose)  
6. Ashtanga Namaskara (Salute with Eight Parts)  
7. Bhujangasana (Cobra Pose)  
8. Parvatasana (Mountain Pose)  
9. Ashwa Sanchalanasana (Opposite Leg)  
10. Padahastasana (Return)  
11. Hasta Uttanasana (Return)  
12. Pranamasana (End)

## Project Structure

```
project/
│
├── README                 # Project description
├── maincode.py            # Contains main python code
├── test.mp4               # Demo video of pose detection for suryanamaskar
├── Output Images/         # Folder for output screenshots and suryanamaskar video

```

## How to Run

1. **Install the required libraries**  
   ```bash
   pip install opencv-python mediapipe numpy
   ```

2. **Run the script**  
   ```bash
   python maincode.py
   ```

3. **Instructions**  
   - A window will open showing your webcam feed.  
   - Select a pose by pressing the corresponding key (1–6).  
   - Follow the on-screen instructions and check feedback.

## Output Preview

- **Real-time pose visualization**
- **On-screen messages for pose correctness**
- **Dynamic guidance for each Surya Namaskar step**
