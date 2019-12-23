import logging
import os
import uuid
from pathlib import PurePath
from typing import Optional
import cv2

from face_cropper.utils.utils import walk_through_files
from face_cropper.core.model import NetModel

logger = logging.getLogger(__name__)

__all__ = ['FaceExtractor']


class FaceExtractor:

    VIDEO_EXTENTIONS = ('.mp4', '.MP4', '.mpg')
    IMAGE_EXTENTIONS = ('.PNG', '.png', '.JPG', '.jpg', '.jpeg')

    def __init__(
        self,
        net_model: 'NetModel' = NetModel(),
        source_dir: Optional[str] = None,
        target_dir: Optional[str] = None,
        conf_threshold: float = 0.7,
        from_video=True,
        from_pictures=True,
        n_frame=None,
        skip_every_n_frame=None
    ):
        self._skip_every_n_frame = skip_every_n_frame
        self._from_video = from_video
        self._from_pictures = from_pictures
        self._net = net_model.net
        self._conf_threshold = conf_threshold
        self._n_frame=n_frame

        current_dir = os.path.dirname(os.path.realpath(__file__))
        logger.info(f'Current directory {current_dir!r}')
        logger.info(f'Threshold {self._conf_threshold!r}')

        self._source_dir = source_dir or current_dir
        self._target_dir = target_dir or current_dir

    def detect_face_opencv_dnn(self, frame) -> Optional[object]:
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

                rec_color = (0, 255, 0)
                rec_size = int(round(frame_height / 300))
                cv2.rectangle(frame_dnn, (x1, y1), (x2, y2), rec_color, rec_size)

                cropped_img = frame_dnn[y1 + rec_size:y2 - rec_size, x1 + rec_size:x2 - rec_size]

        return frame_dnn, cropped_img

    def save_frame(self, frame, file_name: str):
        """ Saves to target_dir/file_name.jpg"""
        try:
            cv2.imwrite(f'{self._target_dir}/{file_name}-{uuid.uuid4()}.jpg', frame)
        except Exception as e:
            logger.warning(f'{file_name}-error: {e!r}')

    def save_faces_from_video(self, file_path: str):
        cap = cv2.VideoCapture(file_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self._n_frame:
            self._skip_every_n_frame = frame_count // self._n_frame

        logger.warning(f'Frame count: {frame_count} File: {file_path}')

        count = 0
        while True:
            has_frame, frame = cap.read()
            if not has_frame:
                break

            rec_frame, cropped_frame = self.detect_face_opencv_dnn(frame)

            if cropped_frame is not None:
                file_name = PurePath(file_path).name
                self.save_frame(cropped_frame, file_name)

            if self._skip_every_n_frame:
                count += self._skip_every_n_frame
                if count >= frame_count:
                    logger.warning(f'Out of frames {frame_count} < {count}')
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, count)

        cap.release()
        cv2.destroyAllWindows()

    def save_faces_from_picture(self, file_path: str):
        frame = cv2.imread(file_path)

        rec_frame, cropped_frame = self.detect_face_opencv_dnn(frame)
        if cropped_frame is not None:
            file_name = PurePath(file_path).name
            self.save_frame(cropped_frame, file_name)

    def run(self):
        for file_path in walk_through_files(self._source_dir):
            if file_path.endswith(self.VIDEO_EXTENTIONS) and self._from_video:
                self.save_faces_from_video(file_path)
            elif file_path.endswith(self.IMAGE_EXTENTIONS) and self._from_pictures:
                self.save_faces_from_picture(file_path)
            else:
                logger.warning(f'Unknown format file {file_path}')
