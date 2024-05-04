from collections import defaultdict
import numpy as np
from PIL import Image
import cv2
import torch
from facenet_pytorch import MTCNN
from adaface.align_trans import get_reference_facial_points, warp_and_crop_face
from adaface import net
from distances import find_distance, find_threshold


class FaceRecognition:
    def __init__(self, model_architecture='ir_50'):
        self.adaface_models = {
            'ir_50': "pretrained/adaface_ir50_ms1mv2.ckpt",
            'ir_101': "pretrained/adaface_ir101_ms1mv2.ckpt"
        }
        self.crop_size = (112, 112)
        self.refrence = get_reference_facial_points(
            default_square=self.crop_size[0] == self.crop_size[1])

        self.face_detection_model = MTCNN()
        self.face_recognition_model = self.load_pretrained_model(
            model_architecture)

        self.known_faces = defaultdict(
            lambda: {"embedding": None, "last_seen": 0, "present": False, "count": 0})
        self.current_ids = set()

    def load_pretrained_model(self, architecture='ir_50'):
        assert architecture in self.adaface_models.keys()
        model = net.build_model(architecture)
        statedict = torch.load(
            self.adaface_models[architecture], map_location=torch.device('cpu'))['state_dict']
        model_statedict = {key[6:]: val for key,
                           val in statedict.items() if key.startswith('model.')}
        model.load_state_dict(model_statedict)
        model.eval()
        return model

    def verify_face(self, embedding, saved_embedding, distance_metric, relative_face_area):
        distance = find_distance(
            embedding, saved_embedding, distance_metric)
        threshold = find_threshold(distance_metric, relative_face_area)
        if distance <= threshold:
            return True
        return False

    def recognize_face(self, face_encoding, relative_face_area):
        for face_id, data in self.known_faces.items():
            known_embedding = data["embedding"]
            if known_embedding is not None:
                match = self.verify_face(
                    face_encoding, known_embedding, distance_metric="euclidean", relative_face_area=relative_face_area)
                if match:
                    if not data["present"]:
                        data["count"] += 1
                        data["present"] = True
                    return face_id
        return None

    def update_known_faces(self, face_encoding):
        face_id = len(self.known_faces)
        self.known_faces[face_id]["embedding"] = face_encoding
        self.known_faces[face_id]["present"] = True
        self.known_faces[face_id]["count"] += 1
        return face_id

    def to_input(self, pil_rgb_image):
        np_img = np.array(pil_rgb_image)
        brg_img = ((np_img[:, :, ::-1] / 255.) - 0.5) / 0.5
        tensor = torch.tensor([brg_img.transpose(2, 0, 1)]).float()
        return tensor

    def align_multi(self, img, limit=None):
        boxes, _, landmarks = self.face_detection_model.detect(
            img, landmarks=True)
        if limit:
            boxes = boxes[:limit]
            landmarks = landmarks[:limit]
        faces = []
        for landmark in landmarks:
            facial5points = [tuple(landmark[j]) for j in range(5)]
            warped_face = warp_and_crop_face(
                np.array(img), facial5points, self.refrence, crop_size=self.crop_size)
            faces.append(Image.fromarray(warped_face))
        return boxes, faces

    def detect_faces(self, img):
        img = Image.fromarray(img).convert("RGB")
        try:
            bboxes, faces = self.align_multi(img)
        except:
            bboxes = None
            faces = None
        return bboxes, faces

    def process_frame(self, frame):
        boxes, faces = self.detect_faces(frame)
        if boxes is not None and faces is not None:
            for box, face in zip(boxes, faces):
                self.process_face(frame, box, face)
        absent_ids = set(list(self.known_faces.keys())) - self.current_ids
        for id in absent_ids:
            self.known_faces[id]["present"] = False
        self.current_ids = set()
        return frame

    def process_face(self, frame, box, face):
        bgr_input = self.to_input(face)
        embeddings, _ = self.face_recognition_model(bgr_input)
        left, top, right, bottom = [int(coord) for coord in box]
        face_area = (right-left) * (bottom-top)
        frame_width, frame_height, _ = frame.shape
        relative_face_area = face_area / (frame_width * frame_height)
        face_id = self.recognize_face(embeddings, relative_face_area)
        if face_id is None:
            face_id = self.update_known_faces(embeddings)
        self.current_ids.add(face_id)
        cv2.rectangle(frame, (int(left), int(top)),
                      (int(right), int(bottom)), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, f"ID: {face_id}, Count: {self.known_faces[face_id]['count']}", (int(
            left) + 6, int(bottom) - 6), font, 0.7, (255, 255, 255), 2)
