from typing import Union
import numpy as np
import torch


def find_cosine_distance(
    source_representation: torch.Tensor, test_representation: torch.Tensor
) -> np.float64:

    source_representation = source_representation.detach().numpy()
    test_representation = test_representation.detach().numpy()

    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


def find_euclidean_distance(
    source_representation: Union[np.ndarray, torch.Tensor],
    test_representation: Union[np.ndarray, torch.Tensor]
) -> np.float64:

    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(
        euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


def l2_normalize(x: torch.Tensor) -> np.ndarray:
    x = x.detach().numpy()
    return x / np.sqrt(np.sum(np.multiply(x, x)))


def find_distance(
    alpha_embedding: torch.Tensor,
    beta_embedding: torch.Tensor,
    distance_metric: str,
) -> np.float64:

    if distance_metric == "cosine":
        distance = find_cosine_distance(alpha_embedding, beta_embedding)
    elif distance_metric == "euclidean":
        distance = find_euclidean_distance(alpha_embedding, beta_embedding)
    elif distance_metric == "euclidean_l2":
        distance = find_euclidean_distance(
            l2_normalize(alpha_embedding), l2_normalize(beta_embedding)
        )
    else:
        raise ValueError("Invalid distance_metric passed - ", distance_metric)
    return distance


def find_threshold(distance_metric: str, reletive_face_area: float) -> float:
    thresholds = {"cosine": 0.68, "euclidean": 1.17, "euclidean_l2": 1.17} if reletive_face_area > 0.02 else\
                 {"cosine": 0.5, "euclidean": 1.0, "euclidean_l2": 1.0}

    threshold = thresholds.get(distance_metric, 0.4)
    return threshold
