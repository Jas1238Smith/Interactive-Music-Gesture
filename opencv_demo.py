import cv2 as cv


def main():
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break
        # Display the resulting frame
        cv.imshow("frame", frame)
        if cv.waitKey(1) == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
