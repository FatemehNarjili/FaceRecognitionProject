import torch
import numpy as np
from PIL import Image
from abc import ABC, abstractmethod
from torchvision import transforms
from adaface import net
from facenet_pytorch import InceptionResnetV1
from adaface.align_trans import get_reference_facial_points, warp_and_crop_face


class FaceRecognition(ABC):
    @abstractmethod
    def get_embeddings(self, frame, box, landmark):
        pass

    @abstractmethod
    def extract_face(self, frame, box, landmark):
        pass


class AdaFaceModel(FaceRecognition):
    def __init__(self, model_architecture='ir_50'):
        self.adaface_models = {
            'ir_50': "pretrained/adaface_ir50_ms1mv2.ckpt",
            'ir_101': "pretrained/adaface_ir101_ms1mv2.ckpt"
        }
        self.model = self.load_pretrained_model(
            model_architecture)

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

    def to_input(self, pil_rgb_image):
        np_img = np.array(pil_rgb_image)
        brg_img = ((np_img[:, :, ::-1] / 255.) - 0.5) / 0.5
        tensor = torch.tensor([brg_img.transpose(2, 0, 1)]).float()
        return tensor

    def extract_face(self, frame, box, landmark):
        crop_size = (112, 112)
        refrence = get_reference_facial_points(
            default_square=crop_size[0] == crop_size[1])

        frame = Image.fromarray(frame).convert("RGB")

        facial5points = [tuple(landmark[j]) for j in range(5)]
        warped_face = warp_and_crop_face(
            np.array(frame), facial5points, refrence, crop_size=crop_size)
        warped_face = Image.fromarray(warped_face)

        return warped_face

    def get_embeddings(self, frame, box, landmark):
        face = self.extract_face(frame, box, landmark)
        bgr_input = self.to_input(face)
        embeddings, _ = self.model(bgr_input)
        return embeddings


class InceptionResnetModel(FaceRecognition):
    def __init__(self):
        self.model = InceptionResnetV1(
            pretrained="vggface2").eval()
        self.preprocess = transforms.Compose([
            transforms.Lambda(lambda x: Image.fromarray(x)),
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    def extract_face(self, frame, box, landmark):
        left, top, right, bottom = [int(coord) for coord in box]
        face = frame[top:top+bottom, left:left+right]
        return face

    def get_embeddings(self, frame, box, landmark):
        face = self.extract_face(frame, box, landmark)
        face_tensor = self.preprocess(face).unsqueeze(0)

        with torch.no_grad():
            embeddings = self.model(face_tensor)

        return embeddings
