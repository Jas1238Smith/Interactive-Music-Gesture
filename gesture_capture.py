import cv2 as cv
import numpy as np
import mediapipe as mp

from pythonosc import osc_message_builder
from pythonosc import udp_client

import mediapipe as mp

from threading import Thread
import time

# Connect to Pure Data
client = udp_client.UDPClient("127.0.0.1", 8002)
start_client = udp_client.UDPClient("127.0.0.1", 8003)
end_client = udp_client.UDPClient("127.0.0.1", 8004)

empty_msg = osc_message_builder.OscMessageBuilder(address="dummy").build()

INPUT_FLAG = False
CONNECTED_FINGERS = [4, 8]


def update_labels():
    global INPUT_FLAG
    global CONNECTED_FINGERS

    while True:
        if INPUT_FLAG:
            print("Which keypoints should be connected?")
            idx = int(input("Which finger is first (0-20)?"))
            if -1 < idx < 21:
                CONNECTED_FINGERS[0] = idx
            idx = int(input("Which finger is first (0-20)?"))
            if -1 < idx < 21:
                CONNECTED_FINGERS[1] = idx
            INPUT_FLAG = False

        time.sleep(0.2)


def main():
    global INPUT_FLAG
    global CONNECTED_FINGERS

    # Load the camera
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 540)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # Mediapipe Hands
    hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.25,
    )

    # Online Training/Relabeling
    inputThread = Thread(
        target=update_labels,
    )
    inputThread.daemon = True
    inputThread.start()

    # Stored Data
    pose_data = np.zeros((1, 2))
    pose_data_block = np.zeros((10, 2))

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        key = cv.waitKey(1)

        if key == 27:  # ESC
            end_client.send(empty_msg)
            cap.release()
            break
        if key == 114 or key == 108:  # r: retrain. l: label
            INPUT_FLAG = True

        image = cv.flip(frame, 1)  # Mirror Display
        image_width, image_height = image.shape[1], image.shape[0]

        # Detect Hands
        hands_results = hands.process(image)

        if hands_results.multi_hand_landmarks is not None:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                landmark_array = np.empty((0, 2), int)

                for position, landmark in enumerate(hand_landmarks.landmark):
                    landmark_x = min(int(landmark.x * image_width), image_width - 1)
                    landmark_y = min(int(landmark.y * image_height), image_height - 1)

                    cv.circle(
                        image,
                        (landmark_x, landmark_y),
                        5 if (position % 4) else 20,
                        (0, 0, 0 if (position % 4) else 255),
                    )

                    landmark_point = [np.array((landmark_x, landmark_y))]
                    landmark_array = np.append(landmark_array, landmark_point, axis=0)

                x, y, w, h = cv.boundingRect(landmark_array)
                cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), 1)

                pose_data = np.array(
                    [
                        landmark_array[4][0] / image_width,
                        landmark_array[4][1] / image_height,
                    ]
                )

                pose_data_block = np.roll(pose_data_block, -1, axis=0)
                pose_data_block[-1] = np.copy(pose_data)

                horizontal_motion = pose_data_block[-1][0] - pose_data_block[0][0]
                vertical_motion = pose_data_block[-1][1] - pose_data_block[0][1]

                cv.line(
                    image,
                    (
                        landmark_array[CONNECTED_FINGERS[0]][0],
                        landmark_array[CONNECTED_FINGERS[0]][1],
                    ),
                    (
                        landmark_array[CONNECTED_FINGERS[1]][0],
                        landmark_array[CONNECTED_FINGERS[1]][1],
                    ),
                    (0, 255, 0),
                    10,
                )

                line_length = np.abs(
                    np.linalg.norm(
                        landmark_array[CONNECTED_FINGERS[0]]
                        - landmark_array[CONNECTED_FINGERS[1]]
                    )
                )

                # Send OSC Output
                if pose_data[0] > 0 or pose_data[1] > 0:
                    msg = osc_message_builder.OscMessageBuilder(address="/hand")
                    msg.add_arg(str(pose_data[0]))
                    msg.add_arg(str(pose_data[1]))
                    msg.add_arg(str((w) / image_width))
                    msg.add_arg(str((h) / image_height))
                    msg.add_arg(str(horizontal_motion))
                    msg.add_arg(str(vertical_motion))
                    msg.add_arg(str(line_length))
                    msg = msg.build()
                    client.send(msg)

        cv.imshow("frame", image)

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
