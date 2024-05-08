from abc import ABC, abstractmethod
from facenet_pytorch import MTCNN


class FaceDetection(ABC):
    @abstractmethod
    def detect_faces(self):
        pass


class MTCNNModel(FaceDetection):
    def __init__(self):
        self.model = MTCNN()

    def detect_faces(self, img):
        try:
            boxes, _, landmarks = self.model.detect(img, landmarks=True)
        except:
            boxes = None
            landmarks = None
        return boxes, landmarks
