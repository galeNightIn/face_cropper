import cv2

__all__ = ['NetModel']


class NetModel:
    def __init__(self, dnn_source="RES10"):
        self._model_file = None
        self._config_file = None
        self._net = None

        if dnn_source == "RES10":
            self._model_file = "../models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
            self._config_file = "../models/deploy.prototxt"
            self._net = cv2.dnn.readNetFromCaffe(self._config_file, self._model_file)
        else:
            self._model_file = "../models/opencv_face_detector_uint8.pb"
            self._config_file = "../models/opencv_face_detector.pbtxt"
            self._net = cv2.dnn.readNetFromTensorflow(self._model_file, self._config_file)

    @property
    def net(self):
        return self._net
