import logging
import numpy as np
import torch
from typing import List, Optional
from config import AppConfig
from tracker import RawDetection
from utils import load_vocabulary

logger = logging.getLogger(__name__)


class ObjectDetector:
    def __init__(self, config: AppConfig):
        self._config = config
        self._model = None
        self._vocabulary: List[str] = []
        self._backend: str = config.detector_backend
        self._device = config.device
        self._loaded = False

    def load(self):
        vocab = load_vocabulary(self._config.vocabulary_file)
        if not vocab:
            logger.warning("Vocabulary file empty or missing — using fallback navigation vocabulary")
            vocab = _FALLBACK_VOCAB
        self._vocabulary = vocab

        success = False

        if self._backend == "yolov8-oiv7":
            success = self._load_yolov8_oiv7()

        if not success and self._backend not in ("yolo-world",):
            logger.warning("OIV7 load failed — trying YOLO-World fallback")

        if not success:
            success = self._load_yolo_world()

        if not success:
            logger.warning("YOLO-World load failed — trying YOLOE fallback")
            success = self._load_yoloe()

        if not success:
            logger.warning("YOLOE failed — falling back to YOLOv8 fixed-class detector")
            success = self._load_yolov8_fixed()

        if not success:
            logger.error("All detector backends failed")
            self._loaded = False
        else:
            self._loaded = True
            logger.info(f"Detector ready: {self._backend} | {len(self._vocabulary)} classes | device={self._device}")

    def _load_yolov8_oiv7(self) -> bool:
        try:
            from ultralytics import YOLO
            self._model = YOLO("yolov8s-oiv7.pt")
            if self._device != "cpu":
                self._model.to(self._device)
            self._backend = "yolov8-oiv7"
            self._vocabulary = list(self._model.names.values())
            logger.info(f"YOLOv8-OIV7 loaded: {len(self._vocabulary)} classes")
            return True
        except Exception as e:
            logger.warning(f"YOLOv8-OIV7 load error: {e}")
            return False

    def _load_yolo_world(self) -> bool:
        try:
            from ultralytics import YOLOWorld
            self._model = YOLOWorld(self._config.detector_model)
            self._model.set_classes(self._vocabulary)
            if self._device == "cpu":
                pass
            else:
                self._model.to(self._device)
            self._backend = "yolo-world"
            return True
        except Exception as e:
            logger.warning(f"YOLOWorld load error: {e}")
            return False

    def _load_yoloe(self) -> bool:
        try:
            from ultralytics import YOLOE
            self._model = YOLOE("yoloe-11s-seg.pt")
            self._model.set_classes(self._vocabulary, self._model.get_text_pe(self._vocabulary))
            if self._device != "cpu":
                self._model.to(self._device)
            self._backend = "yoloe"
            return True
        except Exception as e:
            logger.warning(f"YOLOE load error: {e}")
            return False

    def _load_yolov8_fixed(self) -> bool:
        try:
            from ultralytics import YOLO
            self._model = YOLO("yolov8n.pt")
            if self._device != "cpu":
                self._model.to(self._device)
            self._backend = "yolov8-fixed"
            self._vocabulary = list(self._model.names.values())
            logger.warning("Running with fixed 80-class COCO vocabulary — open-vocabulary unavailable")
            return True
        except Exception as e:
            logger.warning(f"YOLOv8 fallback load error: {e}")
            return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def vocabulary(self) -> List[str]:
        return self._vocabulary

    def set_vocabulary(self, vocab: List[str]):
        if not vocab:
            return
        if self._backend == "yolov8-oiv7":
            logger.info("OIV7 uses fixed 600-class vocabulary — custom vocab ignored")
            return
        self._vocabulary = vocab
        if self._backend == "yolo-world":
            try:
                self._model.set_classes(vocab)
                logger.info(f"Vocabulary updated: {len(vocab)} classes")
            except Exception as e:
                logger.warning(f"Vocabulary update failed: {e}")
        elif self._backend == "yoloe":
            try:
                self._model.set_classes(vocab, self._model.get_text_pe(vocab))
                logger.info(f"Vocabulary updated: {len(vocab)} classes")
            except Exception as e:
                logger.warning(f"Vocabulary update failed: {e}")

    def detect(self, frame_bgr: np.ndarray) -> List[RawDetection]:
        if not self._loaded or self._model is None:
            return []
        try:
            results = self._model.predict(
                source=frame_bgr,
                conf=self._config.detector_conf_threshold,
                iou=self._config.detector_iou_threshold,
                imgsz=self._config.detector_imgsz,
                verbose=False,
                stream=False,
            )
            return _parse_results(results, self._vocabulary, self._backend)
        except Exception as e:
            logger.warning(f"Detection error: {e}")
            return []


def _parse_results(results, vocabulary: List[str], backend: str) -> List[RawDetection]:
    detections: List[RawDetection] = []
    if not results:
        return detections

    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return detections

    boxes_xyxy = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    cls_ids = r.boxes.cls.cpu().numpy().astype(int)

    names = r.names if hasattr(r, "names") and r.names else {}

    for i in range(len(boxes_xyxy)):
        cls_id = int(cls_ids[i])
        if cls_id < len(vocabulary):
            label = vocabulary[cls_id]
        elif cls_id in names:
            label = str(names[cls_id])
        else:
            label = f"obj_{cls_id}"

        detections.append(
            RawDetection(
                box=boxes_xyxy[i].astype(np.float32),
                label=label,
                det_conf=float(confs[i]),
            )
        )
    return detections


_FALLBACK_VOCAB = [
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "traffic light", "stop sign", "chair", "table", "door",
    "stairs", "curb", "pole", "tree", "trash can", "bench",
    "wheelchair", "dog", "cat", "bottle", "backpack", "suitcase",
]
