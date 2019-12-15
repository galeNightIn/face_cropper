import logging
import os
import uuid
from pathlib import PurePath
from typing import Optional

import cv2

from core.model import NetModel

logger = logging.getLogger(__name__)

__all__ = ['FaceExtractor']


class FaceExtractor:
    def __init__(
        self,
        net_model: 'NetModel',
        source_dir: Optional[str] = None,
        target_dir: Optional[str] = None,
        conf_threshold: float = 0.7
    ):
        self._net = net_model.net
        self._conf_threshold = conf_threshold

        current_dir = os.path.dirname(os.path.realpath(__file__))
        logger.info(f'Current directory {current_dir!r}')
        logger.info(f'Threshold {self._conf_threshold!r}')

        self._source_dir = source_dir or current_dir
        self._target_dir = target_dir or current_dir

    def detect_face_opencv_dnn(self, frame, rectangle=False) -> Optional[object]:
        frame_dnn = frame.copy()
        frame_height = frame_dnn.shape[0]
        frame_width = frame_dnn.shape[1]

        frame_size = (300, 300)
        mean_substraction_values = (104, 117, 123)  # (R, G, B)
        scalefactor = 1.0

        blob = cv2.dnn.blobFromImage(frame_dnn, scalefactor, frame_size, mean_substraction_values, False, False)

        self._net.setInput(blob)
        detections = self._net.forward()
        cropped_img = None

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self._conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frame_width)
                y1 = int(detections[0, 0, i, 4] * frame_height)
                x2 = int(detections[0, 0, i, 5] * frame_width)
                y2 = int(detections[0, 0, i, 6] * frame_height)

                rec_size = 0
                if rectangle:
                    rec_color = (0, 255, 0)
                    rec_size = int(round(frame_height / 300))
                    cv2.rectangle(frame_dnn, (x1, y1), (x2, y2), rec_color, rec_size)

                cropped_img = frame_dnn[y1 + rec_size:y2 - rec_size, x1 + rec_size:x2 - rec_size]

        return frame_dnn, cropped_img

    @staticmethod
    def save_frame(frame, file_name: str):
        """ Saves to target_dir/file_name.jpg"""
        cv2.imwrite(f'{file_name}-{uuid.uuid4()}.jpg', frame)

    def save_faces_from_video(self, file_path: str):
        cap = cv2.VideoCapture(file_path)
        while True:
            hasFrame, frame = cap.read()
            if not hasFrame:
                break

            rec_frame, cropped_frame = self.detect_face_opencv_dnn(frame)

            if cropped_frame:
                file_name = PurePath(file_path).name
                self.save_frame(cropped_frame, file_name)

    def save_faces_from_picture(self, file_path: str):
        frame = cv2.imread(file_path)

        rec_frame, cropped_frame = self.detect_face_opencv_dnn(frame)
        if cropped_frame:
            file_name = PurePath(file_path).name
            self.save_frame(cropped_frame, file_name)
