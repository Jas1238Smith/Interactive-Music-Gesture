import cv2 as cv
import numpy as np
import mediapipe as mp


def main():
    # Load the camera
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # Mediapipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.25,
        min_tracking_confidence=0.15,
    )

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

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
                        -1,
                    )

                    landmark_point = [np.array((landmark_x, landmark_y))]
                    landmark_array = np.append(landmark_array, landmark_point, axis=0)

                x, y, w, h = cv.boundingRect(landmark_array)
                cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)

        cv.imshow("frame", image)
        if cv.waitKey(1) == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
