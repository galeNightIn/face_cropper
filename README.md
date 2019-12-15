# face_cropper
Detect faces from source directory. Works with both image or video. Saves cropped faces in target directory.

## Install
```bash
git clone https://github.com/galeNightIn/face_cropper.git
```

```bash
pip install /path/to/face_cropper/
```

## Usage 

```python
from face_cropper import FaceExtractor

face_extractor = FaceExtractor(
    source_dir='data/1',
    target_dir='prepared_data/1',
    from_pictures=True,
    from_video=True,
    conf_threshold=0.8
)
face_extractor.run()

```

