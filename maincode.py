import cv2
import mediapipe as mp
import numpy as np
import textwrap

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

pose_options = {
    1: "T-Pose",
    2: "Bicep Curl",
    3: "Tricep Stretch",
    4: "Tree Pose",
    5: "Hands Up",
    6: "Surya Namaskar"
}
selected_pose = None

surya_namaskar_steps = [
    "Pranamasana (Prayer Pose)",
    "Hasta Uttanasana (Raised Arms Pose)",
    "Padahastasana (Hand to Foot Pose)",
    "Ashwa Sanchalanasana (Equestrian Pose)",
    "Dandasana (Stick Pose)",
    "Ashtanga Namaskara (Salute with Eight Parts or Points)",
    "Bhujangasana (Cobra Pose)",
    "Parvatasana (Mountain Pose)",
    "Ashwa Sanchalanasana (Equestrian Pose)",
    "Padahastasana (Hand to Foot Pose)",
    "Hasta Uttanasana (Raised Arms Pose)",
    "Pranamasana (Prayer Pose)"
]
current_step = 0

pose_descriptions = {
    0: "Pranamasana (Prayer Pose): Stand with feet together, palms joined in front of the chest.",
    1: "Hasta Uttanasana (Raised Arms Pose): Raise arms overhead, arch back slightly.",
    2: "Padahastasana (Hand to Foot Pose): Bend forward, touch hands to feet.",
    3: "Ashwa Sanchalanasana (Equestrian Pose): Step one leg back, bend the other knee.",
    4: "Dandasana (Stick Pose): Step both legs back, body in a straight line.",
    5: "Ashtanga Namaskara (Salute with Eight Parts or Points): Lower knees, chest, and chin to the floor.",
    6: "Bhujangasana (Cobra Pose): Lift chest off the floor, arch back.",
    7: "Parvatasana (Mountain Pose): Lift hips, form an inverted V shape.",
    8: "Ashwa Sanchalanasana (Equestrian Pose): Step the opposite leg forward, bend the knee.",
    9: "Padahastasana (Hand to Foot Pose): Bend forward, touch hands to feet.",
    10: "Hasta Uttanasana (Raised Arms Pose): Raise arms overhead, arch back slightly.",
    11: "Pranamasana (Prayer Pose): Stand with feet together, palms joined in front of the chest."
}

def is_t_pose(landmarks):
    left_wrist, right_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST], landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_shoulder, right_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    return abs(left_wrist.y - left_shoulder.y) < 0.1 and abs(right_wrist.y - right_shoulder.y) < 0.1

def is_bicep_curl(landmarks):
    left_wrist, right_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST], landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_elbow, right_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW], landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
    return left_wrist.y < left_elbow.y or right_wrist.y < right_elbow.y

def is_tricep_stretch(landmarks):
    left_wrist, right_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST], landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_elbow, right_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW], landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
    return left_wrist.y < left_elbow.y and left_wrist.x < left_elbow.x or right_wrist.y < right_elbow.y and right_wrist.x > right_elbow.x

def is_tree_pose(landmarks):
    left_ankle, right_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE], landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
    left_knee, right_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE], landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
    return abs(left_ankle.y - right_knee.y) < 0.1 or abs(right_ankle.y - left_knee.y) < 0.1

def is_hands_up(landmarks):
    left_wrist, right_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST], landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_shoulder, right_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    return left_wrist.y < left_shoulder.y and right_wrist.y < right_shoulder.y

def is_surya_namaskar_step(landmarks, step):
    if step == 0:  # Pranamasana (Prayer Pose)
        left_wrist, right_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST], landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_elbow, right_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW], landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
        left_hip, right_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP], landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        return abs(left_wrist.y - right_wrist.y) < 0.05 and abs(left_elbow.y - right_elbow.y) < 0.05 and abs(left_hip.y - right_hip.y) < 0.05 and abs(left_wrist.x - right_wrist.x) < 0.05
    elif step == 1:  # Hasta Uttanasana (Raised Arms Pose)
        left_wrist, right_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST], landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_shoulder, right_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        return left_wrist.y < left_shoulder.y and right_wrist.y < right_shoulder.y and abs(left_wrist.x - right_wrist.x) < 0.1
    elif step == 2:  # Padahastasana (Hand to Foot Pose)
        left_wrist, right_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST], landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_ankle, right_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE], landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
        return left_wrist.y > left_ankle.y and right_wrist.y > right_ankle.y and abs(left_wrist.x - right_wrist.x) < 0.1
    elif step == 3:  # Ashwa Sanchalanasana (Equestrian Pose)
        left_knee, right_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE], landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
        left_ankle, right_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE], landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
        return abs(left_knee.y - right_ankle.y) < 0.1 or abs(right_knee.y - left_ankle.y) < 0.1
    elif step == 4:  # Dandasana (Stick Pose)
        left_ankle, right_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE], landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
        left_shoulder, right_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        return abs(left_ankle.y - right_ankle.y) < 0.05 and abs(left_shoulder.y - right_shoulder.y) < 0.05
    elif step == 5:  # Ashtanga Namaskara (Salute with Eight Parts or Points)
        left_knee, right_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE], landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
        left_ankle, right_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE], landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
        left_elbow, right_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW], landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
        left_shoulder, right_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        return left_knee.y < left_ankle.y and right_knee.y < right_ankle.y and left_elbow.y < left_shoulder.y and right_elbow.y < right_shoulder.y
    elif step == 6:  # Bhujangasana (Cobra Pose)
        left_shoulder, right_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip, right_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP], landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        return left_shoulder.y < left_hip.y and right_shoulder.y < right_hip.y and abs(left_shoulder.x - right_shoulder.x) < 0.1
    elif step == 7:  # Parvatasana (Mountain Pose)
        left_wrist, right_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST], landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_ankle, right_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE], landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
        return left_wrist.y > left_ankle.y and right_wrist.y > right_ankle.y and abs(left_wrist.x - right_wrist.x) < 0.1
    elif step == 8:  # Ashwa Sanchalanasana (Equestrian Pose)
        left_knee, right_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE], landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
        left_ankle, right_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE], landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
        return abs(left_knee.y - right_ankle.y) < 0.1 or abs(right_knee.y - left_ankle.y) < 0.1
    elif step == 9:  # Padahastasana (Hand to Foot Pose)
        left_wrist, right_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST], landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_ankle, right_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE], landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
        return left_wrist.y > left_ankle.y and right_wrist.y > right_ankle.y and abs(left_wrist.x - right_wrist.x) < 0.1
    elif step == 10:  # Hasta Uttanasana (Raised Arms Pose)
        left_wrist, right_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST], landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_shoulder, right_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        return left_wrist.y < left_shoulder.y and right_wrist.y < right_shoulder.y and abs(left_wrist.x - right_wrist.x) < 0.1
    elif step == 11:  # Pranamasana (Prayer Pose)
        left_wrist, right_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST], landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_elbow, right_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW], landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
        left_hip, right_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP], landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        return abs(left_wrist.y - right_wrist.y) < 0.05 and abs(left_elbow.y - right_elbow.y) < 0.05 and abs(left_hip.y - right_hip.y) < 0.05
    return False

# For suryanamaskar pose detection(Video Sample):
#cap = cv2.VideoCapture("C:\\Users\\Anshul\\OneDrive\\Desktop\\YogaAI\\test.mp4")

#For live camera
cap = cv2.VideoCapture(0)
cv2.namedWindow('Pose Detection', cv2.WINDOW_NORMAL)  
cv2.setWindowProperty('Pose Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

retry_count = 0
max_retries = 5

with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:  # Increased confidence for more precise detection
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Failed to read from camera. Retrying...")
            retry_count += 1
            if retry_count >= max_retries:
                print("Max retries reached. Exiting...")
                break
            continue
        retry_count = 0  # Reset retry count on successful read

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw pose options with a 25% smaller gray box and a thin black border
        text_y = 30
        box_width = 187  # 25% smaller than 250
        box_height = int(text_y + len(pose_options) * 22.5)  # Calculate the height of the box
        cv2.rectangle(image, (10, 10), (box_width, box_height), (200, 200, 200), -1)  # Gray background for options
        cv2.rectangle(image, (10, 10), (box_width, box_height), (0, 0, 0), 1)  # Thin black border
        for key, val in pose_options.items():
            cv2.putText(image, f"Press {key}: {val}", (20, int(text_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)  # Blue color, smaller font
            text_y += 22.5  # 25% smaller than 30
        
        if selected_pose:
            cv2.putText(image, f"Selected: {pose_options[selected_pose]}", (20, int(box_height - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)  # Blue color, smaller font, just above the border

        message = ""
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                                      mp_drawing_styles.get_default_pose_landmarks_style())
            landmarks = results.pose_landmarks.landmark
            if selected_pose == 1:
                message = "Perfect!" if is_t_pose(landmarks) else "Incorrect T-Pose!"
            elif selected_pose == 2:
                message = "Perfect!" if is_bicep_curl(landmarks) else "Incorrect Bicep Curl!"
            elif selected_pose == 3:
                message = "Perfect!" if is_tricep_stretch(landmarks) else "Incorrect Tricep Stretch!"
            elif selected_pose == 4:
                message = "Perfect!" if is_tree_pose(landmarks) else "Incorrect Tree Pose!"
            elif selected_pose == 5:
                message = "Perfect!" if is_hands_up(landmarks) else "Incorrect Hands Up!"
            elif selected_pose == 6:
                if current_step < len(surya_namaskar_steps):
                    step_desc = surya_namaskar_steps[current_step]
                    pose_desc = pose_descriptions[current_step]
                    message = f"Step {current_step + 1}: {step_desc}"
                    if is_surya_namaskar_step(landmarks, current_step):
                        status = "Correct Position"
                        status_color = (0, 255, 0)  # Green
                        print(f"Step {current_step + 1} correct")
                    else:
                        status = "Incorrect Position"
                        status_color = (0, 0, 255)  # Red
                        print(f"Step {current_step + 1} incorrect")

                    # Display the message inside the grey box
                    y0, dy = box_height + 30, 15  # Move the text lower
                    max_width = box_width - 20  # Adjust max width to fit inside the box
                    wrapped_desc = textwrap.wrap(pose_desc, width=max_width // 7)  # Adjust width for wrapping
                    wrapped_message = textwrap.wrap(message, width=max_width // 7)  # Adjust width for wrapping
                    wrapped_status = textwrap.wrap(status, width=max_width // 7)  # Adjust width for wrapping

                    # Calculate the height needed for the text
                    total_lines = len(wrapped_message) + len(wrapped_desc) + len(wrapped_status)
                    box_height_steps = y0 + total_lines * dy + 10

                    # Add white highlight and thin black border
                    cv2.rectangle(image, (10, box_height + 5), (box_width, box_height_steps), (255, 255, 255), -1)  # White highlight
                    cv2.rectangle(image, (10, box_height + 5), (box_width, box_height_steps), (0, 0, 0), 1)  # Thin black border

                    # Draw text inside the box
                    y_text = y0
                    for line in wrapped_message:
                        cv2.putText(image, line, (20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 0), 1)  # Dark Green for step
                        y_text += dy
                    cv2.putText(image, "Description:", (20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)  # Black for description label
                    y_text += dy
                    for line in wrapped_desc:
                        cv2.putText(image, line, (20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)  # Black for description
                        y_text += dy
                    for line in wrapped_status:
                        cv2.putText(image, line, (20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)  # Red/Green for status

                    # Display "Correct Position" for a short duration before proceeding to the next step
                    if status == "Correct Position":
                        cv2.putText(image, status, (20, y0 + (len(wrapped_desc) + 1) * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)
                        cv2.imshow('Pose Detection', image)
                        cv2.waitKey(1000)  # Wait for 1 second
                        current_step += 1
                        print(f"Proceeding to step {current_step + 1}")
                else:
                    message = "Surya Namaskar Completed!"
                    cv2.putText(image, message, (20, box_height + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)  # White color for completion message
                    current_step = 0  # Reset step for Surya Namaskar

            if message and selected_pose != 6:
                color = (0, 255, 0) if "Perfect" in message else (0, 0, 255)
                text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                text_x = image.shape[1] - text_size[0] - 50
                text_y = 50
                # Add black highlight
                cv2.rectangle(image, (text_x - 10, text_y - 20), (text_x + text_size[0] + 10, text_y + 10), (0, 0, 0), -1)
                cv2.putText(image, message, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.imshow('Pose Detection', image)
        key = cv2.waitKey(5) & 0xFF
        if key in [ord(str(k)) for k in pose_options.keys()]:
            selected_pose = int(chr(key))
            current_step = 0  # Reset step for Surya Namaskar
        elif key == 27:
            break

cap.release()
cv2.destroyAllWindows()