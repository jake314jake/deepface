from typing import Any, Dict, List, Union
import numpy as np
import cv2
from deepface.modules import detection, preprocessing
from deepface.models.FacialRecognition import FacialRecognition

def represent(
    img_path: Union[str, np.ndarray],
    model: FacialRecognition,
    enforce_detection: bool = True,
    detector_backend: str = "yunet",
    align: bool = True,
    expand_percentage: int = 0,
    normalization: str = "base",
) -> List[Dict[str, Any]]:
    """
    Represent facial images as multi-dimensional vector embeddings using an already instantiated model.
    This function filters out faces at the origin (x=0, y=0) before embedding extraction.

    Args:
        img_path (str or np.ndarray): The exact path to the image or a numpy array in BGR format.
        model (FacialRecognition): An instantiated model for face recognition.
        enforce_detection (bool): If True, raises an exception if no face is detected.
        detector_backend (str): Specifies the face detector backend to use.
        align (bool): If True, performs alignment based on the eye positions.
        expand_percentage (int): Optionally expands the detected facial area by a percentage.
        normalization (str): Normalization method for the image before processing.

    Returns:
        List[Dict[str, Any]]: Each dictionary contains:
            - 'embedding' (np.array): The facial feature vector.
            - 'facial_area' (dict): Coordinates and dimensions of the detected face.
            - 'face_confidence' (float): Confidence score of the face detection.
    """
    resp_objs = []
    target_size = model.input_shape

    img_objs = detection.extract_faces(
        img_path=img_path,
        target_size=(target_size[1], target_size[0]),
        detector_backend=detector_backend,
        grayscale=False,
        enforce_detection=enforce_detection,
        align=align,
        expand_percentage=expand_percentage,
    )

    for img_obj in img_objs:
        region = img_obj["facial_area"]
        confidence = img_obj["confidence"]

        # Skip embedding extraction if the face is at the origin (x=0 and y=0)
        if confidence < 0.90:
            continue

        img = img_obj["face"]
        confidence = img_obj["confidence"]
        img = preprocessing.normalize_input(img=img, normalization=normalization)

        embedding = model.find_embeddings(img)

        resp_obj = {
            "embedding": embedding,
            "facial_area": region,
            "face_confidence": confidence
        }
        resp_objs.append(resp_obj)

    return resp_objs
