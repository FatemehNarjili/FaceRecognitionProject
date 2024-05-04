import cv2
from face_recognition import FaceRecognition


class VideoCapture:
    def __init__(self):
        self.video_capture = cv2.VideoCapture(0)
        self.face_recognition = FaceRecognition()

    def run(self):
        while True:
            ret, frame = self.video_capture.read()
            if not ret:
                continue

            processed_frame = self.face_recognition.process_frame(frame)

            cv2.imshow("Video", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    video_capture = VideoCapture()
    video_capture.run()
