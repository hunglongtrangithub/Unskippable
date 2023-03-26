from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()



app = Flask(__name__)

camera = cv2.VideoCapture(0) # 0 for first webcam

counter = 0
stage = "down"


# Update counter function
def UpdateCounter(angle):
    global counter
    global stage
    if angle > 90:
        stage = "down"
    if angle < 45 and stage == "down":
        stage = "up"
        counter += 1
        # print(counter)


# Calculate angle
def calculate_angle(first, mid, end):
    first = np.array(first)  # First
    mid = np.array(mid)  # Mid
    end = np.array(end)  # End

    radians = np.arctan2(end[1] - mid[1], end[0] - mid[0]) - np.arctan2(first[1] - mid[1], first[0] - mid[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def display_counter(exercise,frame):
    # Dictionary to pick the appropriate landmarks according to the exercise:
    exercise_dict = {
        "curl": [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_ELBOW.value,
                 mp_pose.PoseLandmark.LEFT_WRIST.value],
        "pushup": [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_ELBOW.value,
                   mp_pose.PoseLandmark.LEFT_WRIST.value],
        "squat": [mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_KNEE.value,
                  mp_pose.PoseLandmark.LEFT_ANKLE.value],
        "lunge": [mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_KNEE.value,
                  mp_pose.PoseLandmark.LEFT_ANKLE.value],
        "situp": [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_HIP.value,
                  mp_pose.PoseLandmark.LEFT_KNEE.value],
        "jumpingjack": [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_HIP.value,
                        mp_pose.PoseLandmark.LEFT_KNEE.value],
        "pullup": [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_ELBOW.value,
                   mp_pose.PoseLandmark.LEFT_WRIST.value],
    }
    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while camera.isOpened():
            # ret, frame = camera.read()

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates according to the exercise
                first = [landmarks[exercise_dict[exercise][0]].x, landmarks[exercise_dict[exercise][0]].y]
                mid = [landmarks[exercise_dict[exercise][1]].x, landmarks[exercise_dict[exercise][1]].y]
                end = [landmarks[exercise_dict[exercise][2]].x, landmarks[exercise_dict[exercise][2]].y]

                # Calculate angle
                angle = calculate_angle(first, mid, end)

                # Visualize angle


                # Update counter
                UpdateCounter(angle)

                # Render detections


            except:
                pass

            # Render curl counter
            # # Setup status box
            # cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
            #
            # # Rep data
            # cv2.putText(image, 'REPS', (15, 12),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            # cv2.putText(image, str(counter),
            #             (10, 60),
            #             cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            #
            # # Stage data
            # cv2.putText(image, 'STAGE', (65, 12),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            # cv2.putText(image, stage,
            #             (60, 60),
            #             cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            #
            # cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        camera.release()
        cv2.destroyAllWindows()





def generate_frames(excersise):
    global stage, pushup_count
    exercise_dict = {
        "curl": [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_ELBOW.value,
                 mp_pose.PoseLandmark.LEFT_WRIST.value],
        "pushup": [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_ELBOW.value,
                   mp_pose.PoseLandmark.LEFT_WRIST.value],
        "squat": [mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_KNEE.value,
                  mp_pose.PoseLandmark.LEFT_ANKLE.value],
        "lunge": [mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_KNEE.value,
                  mp_pose.PoseLandmark.LEFT_ANKLE.value],
        "situp": [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_HIP.value,
                  mp_pose.PoseLandmark.LEFT_KNEE.value],
        "jumpingjack": [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_HIP.value,
                        mp_pose.PoseLandmark.LEFT_KNEE.value],
        "pullup": [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_ELBOW.value,
                   mp_pose.PoseLandmark.LEFT_WRIST.value],
    }

    while True:
        success, frame = camera.read()  # read the camera frame

        if not success:
            break
        else:
            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make detection
                results = pose.process(image)

                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark

                    # Get coordinates according to the exercise
                    first = [landmarks[exercise_dict[exercise][0]].x, landmarks[exercise_dict[exercise][0]].y]
                    mid = [landmarks[exercise_dict[exercise][1]].x, landmarks[exercise_dict[exercise][1]].y]
                    end = [landmarks[exercise_dict[exercise][2]].x, landmarks[exercise_dict[exercise][2]].y]

                    # Calculate angle
                    angle = calculate_angle(first, mid, end)

                    # Visualize angle

                    # Update counter
                    UpdateCounter(angle)

                    # Render detections


                except:
                    pass
            # display_counter('curl',frame=frame)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames('curl'), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def homepage() :
    return render_template('index.html')

@app.route('/level_page')
def level_page() :
    return render_template('level_page.html')


@app.route('/workout_page')
def workout_page():
    """Video streaming home page."""
    return render_template('workout_page.html', curl_count = counter)





if __name__ == '__main__':
    app.run(debug=True)




