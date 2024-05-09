import cv2
from face_id_manager import FaceIDManager


class FaceProcessor:
    def __init__(self, face_detection_model, face_recognition_model):
        self.face_detection = face_detection_model
        self.face_recognition_model = face_recognition_model
        self.face_id_manager = FaceIDManager()

    def process_frame(self, frame):
        boxes, landmarks = self.face_detection.detect_faces(frame)

        if boxes is not None and landmarks is not None:
            for box, landmark in zip(boxes, landmarks):
                self.process_face(frame, box, landmark)

        self.face_id_manager.update_current_ids()

        return frame

    def process_face(self, frame, box, landmark):
        embeddings = self.face_recognition_model.get_embeddings(
            frame, box, landmark)

        left, top, right, bottom = [int(coord) for coord in box]

        relative_face_area = self.calculate_relative_face_area(
            frame, left, top, right, bottom)

        face_id = self.face_id_manager.find_face_id(
            embeddings, relative_face_area)

        cv2.rectangle(frame, (int(left), int(top)),
                      (int(right), int(bottom)), (0, 0, 255), 2)

        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, f"ID: {face_id}, Count: {self.face_id_manager.known_faces[face_id]['count']}", (int(
            left) + 6, int(bottom) - 6), font, 0.7, (255, 255, 255), 2)

    def calculate_relative_face_area(self, frame, left, top, right, bottom):
        face_area = (right - left) * (bottom - top)
        frame_width, frame_height, _ = frame.shape
        frame_area = frame_width * frame_height
        relative_face_area = face_area / frame_area
        return relative_face_area
