from collections import defaultdict
from typing import Union
import torch
from distances import find_distance, find_threshold


class FaceIDManager:
    def __init__(self):
        self.known_faces = defaultdict(
            lambda: {"embedding": None, "last_seen": 0,
                     "present": False, "count": 0}
        )
        self.current_ids = set()

    def is_match(
        self, embedding: torch.Tensor, saved_embedding: torch.Tensor,
        distance_metric: str, relative_face_area: float
    ) -> bool:
        distance = find_distance(embedding, saved_embedding, distance_metric)
        threshold = find_threshold(distance_metric, relative_face_area)

        if distance <= threshold:
            return True
        return False

    def find_face_id(self, face_encoding: torch.Tensor, relative_face_area: float) -> Union[int, None]:
        for face_id, data in self.known_faces.items():
            known_embedding = data["embedding"]
            if known_embedding is not None:
                match = self.is_match(face_encoding, known_embedding,
                                      distance_metric="euclidean_l2", relative_face_area=relative_face_area)
                if match:
                    new_embedding = (known_embedding + face_encoding) / 2
                    self.known_faces[face_id]["embedding"] = new_embedding
                    if not self.known_faces[face_id]["present"]:
                        self.known_faces[face_id]["count"] += 1
                        self.known_faces[face_id]["present"] = True
                    break
        else:
            face_id = self.assign_face_id(face_encoding)

        self.current_ids.add(face_id)

        return face_id

    def assign_face_id(self, face_encoding: torch.Tensor) -> int:
        face_id = len(self.known_faces)
        self.known_faces[face_id] = {
            "embedding": face_encoding,
            "present": True,
            "count": 1
        }
        return face_id

    def update_current_ids(self):
        absent_ids = set(list(self.known_faces.keys())) - self.current_ids

        for id in absent_ids:
            self.known_faces[id]["present"] = False
        self.current_ids = set()
