import time
from typing import Any, Dict, List, Union
import numpy as np

import time
from typing import Any, Dict, List, Union
import numpy as np

# Assuming find_cosine_distance, find_euclidean_distance, l2_normalize, and find_threshold functions are defined as before

def verify_embeddings_with_list(
    emb1: np.ndarray,
    embeddings_dict: Dict[str, List[np.ndarray]],
    model_name: str = "VGG-Face",
    distance_metric: str = "cosine",
) -> Dict[str, Any]:
    tic = time.time()

    # Initialize minimum distance and the name of the best match
    min_distance = float('inf')
    best_match_name = None

    # Iterate through each person and their list of embeddings in the dictionary
    for person_name, emb_list in embeddings_dict.items():
        for emb2 in emb_list:
            if distance_metric == "cosine":
                distance = find_cosine_distance(emb1, emb2)
            elif distance_metric == "euclidean":
                distance = find_euclidean_distance(emb1, emb2)
            elif distance_metric == "euclidean_l2":
                distance = find_euclidean_distance(l2_normalize(emb1), l2_normalize(emb2))
            else:
                raise ValueError("Invalid distance_metric passed - ", distance_metric)

            if distance < min_distance:
                min_distance = distance
                best_match_name = person_name

    threshold = find_threshold(model_name, distance_metric)
    verified = min_distance <= threshold #true or false
    toc = time.time()

    return {
        "verified": verified,
        "best_match_name": best_match_name,
        "distance": min_distance,
        "threshold": threshold,
        "model": model_name,
        "similarity_metric": distance_metric,
        "time": round(toc - tic, 2),
    }


def find_cosine_distance(
    source_representation: Union[np.ndarray, list], test_representation: Union[np.ndarray, list]
) -> np.float64:
    """
    Find cosine distance between two given vectors
    Args:
        source_representation (np.ndarray or list): 1st vector
        test_representation (np.ndarray or list): 2nd vector
    Returns
        distance (np.float64): calculated cosine distance
    """
    if isinstance(source_representation, list):
        source_representation = np.array(source_representation)

    if isinstance(test_representation, list):
        test_representation = np.array(test_representation)

    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


def find_euclidean_distance(
    source_representation: Union[np.ndarray, list], test_representation: Union[np.ndarray, list]
) -> np.float64:
    """
    Find euclidean distance between two given vectors
    Args:
        source_representation (np.ndarray or list): 1st vector
        test_representation (np.ndarray or list): 2nd vector
    Returns
        distance (np.float64): calculated euclidean distance
    """
    if isinstance(source_representation, list):
        source_representation = np.array(source_representation)

    if isinstance(test_representation, list):
        test_representation = np.array(test_representation)

    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


def l2_normalize(x: Union[np.ndarray, list]) -> np.ndarray:
    """
    Normalize input vector with l2
    Args:
        x (np.ndarray or list): given vector
    Returns:
        y (np.ndarray): l2 normalized vector
    """
    if isinstance(x, list):
        x = np.array(x)
    return x / np.sqrt(np.sum(np.multiply(x, x)))


def find_threshold(model_name: str, distance_metric: str) -> float:
    """
    Retrieve pre-tuned threshold values for a model and distance metric pair
    Args:
        model_name (str): facial recognition model name
        distance_metric (str): distance metric name. Options are cosine, euclidean
            and euclidean_l2.
    Returns:
        threshold (float): threshold value for that model name and distance metric
            pair. Distances less than this threshold will be classified same person.
    """

    base_threshold = {"cosine": 0.40, "euclidean": 0.55, "euclidean_l2": 0.75}

    thresholds = {
        # "VGG-Face": {"cosine": 0.40, "euclidean": 0.60, "euclidean_l2": 0.86}, # 2622d
        "VGG-Face": {
            "cosine": 0.68,
            "euclidean": 1.17,
            "euclidean_l2": 1.17,
        },  # 4096d - tuned with LFW
        "Facenet": {"cosine": 0.40, "euclidean": 10, "euclidean_l2": 0.80},
        "Facenet512": {"cosine": 0.30, "euclidean": 23.56, "euclidean_l2": 1.04},
        "ArcFace": {"cosine": 0.68, "euclidean": 4.15, "euclidean_l2": 1.13},
        "Dlib": {"cosine": 0.07, "euclidean": 0.6, "euclidean_l2": 0.4},
        "SFace": {"cosine": 0.593, "euclidean": 10.734, "euclidean_l2": 1.055},
        "OpenFace": {"cosine": 0.10, "euclidean": 0.55, "euclidean_l2": 0.55},
        "DeepFace": {"cosine": 0.23, "euclidean": 64, "euclidean_l2": 0.64},
        "DeepID": {"cosine": 0.015, "euclidean": 45, "euclidean_l2": 0.17},
    }

    threshold = thresholds.get(model_name, base_threshold).get(distance_metric, 0.4)

    return threshold