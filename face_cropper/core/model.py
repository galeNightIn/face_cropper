import logging
import os
from pathlib import Path

import cv2

logger = logging.getLogger(__file__)

PATH_DIR = Path(__file__)
logger.info(f'Path dir {PATH_DIR!r}')

PATH_DIR_PARENT = str(PATH_DIR.parent.parent)
logger.info(f'Path dir parent {PATH_DIR_PARENT!r}')


class NetModel:
    def __init__(self, dnn_source="RES10"):
        self._model_file = None
        self._config_file = None
        self._net = None

        if dnn_source == "RES10":
            self._model_file = os.path.join(PATH_DIR_PARENT, "models/res10_300x300_ssd_iter_140000_fp16.caffemodel")
            self._config_file = os.path.join(PATH_DIR_PARENT, "models/deploy.prototxt")
            self._net = cv2.dnn.readNetFromCaffe(self._config_file, self._model_file)
        else:
            self._model_file = os.path.join(PATH_DIR_PARENT, "models/opencv_face_detector_uint8.pb")
            self._config_file = os.path.join(PATH_DIR_PARENT, "models/opencv_face_detector.pbtxt")
            self._net = cv2.dnn.readNetFromTensorflow(self._model_file, self._config_file)

    @property
    def net(self):
        return self._net
