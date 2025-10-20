# Copyright (c) 2025 Awakening AI

# au_detector.py
import os
import sys
import platform
from dataclasses import dataclass
from typing import Union # for Python <3.10 compatibility
import numpy as np
import cv2
import onnxruntime as ort


def _prefer_providers(force_cpu: bool) -> list:
    """
    Provider priority:
      - macOS: CoreML -> CPU
      - elsewhere: CUDA -> CPU (if CUDA build is present)
    Falls back gracefully to whatever this wheel actually supports.
    """
    if force_cpu:
        return ["CPUExecutionProvider"]

    available = set(ort.get_available_providers())
    wanted: list = []

    if sys.platform == "darwin" and "CoreMLExecutionProvider" in available:
        # CoreML uses ANE/GPU on Apple Silicon; CPU fallback keeps things robust.
        wanted.append(("CoreMLExecutionProvider", {}))
    if "CUDAExecutionProvider" in available:
        wanted.append(("CUDAExecutionProvider", {}))

    wanted.append("CPUExecutionProvider")
    # Filter to only those that really exist on this wheel
    filtered = []
    for p in wanted:
        name = p if isinstance(p, str) else p[0]
        if name in available:
            filtered.append(p)
    return filtered or ["CPUExecutionProvider"]


def _session_options(num_threads: Union[int, None] = None) -> ort.SessionOptions:
    so = ort.SessionOptions()
    # Full graph fusion/constant folding
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Reduce Python↔C logging overhead
    so.log_severity_level = 3  # 0=verbose … 4=fatal
    # Threading: macOS benefits from modest intra-op threads
    if num_threads is None:
        # Heuristic: don’t oversubscribe tiny laptops
        num_threads = min(os.cpu_count() or 4, 6)
    so.intra_op_num_threads = num_threads
    so.inter_op_num_threads = 1  # keep it simple; models here are small
    # Memory arenas help CPU EP perf
    so.enable_mem_pattern = True
    so.enable_cpu_mem_arena = True
    return so


@dataclass
class ORTModel:
    session: ort.InferenceSession
    input_name: str
    layout: str   # "NCHW" or "NHWC"
    H: int
    W: int

    @staticmethod
    def create(path: str, force_cpu: bool = False, num_threads: Union[int, None] = None) -> "ORTModel":
        # Tip: If you ever want heavy validation, enable the checker once offline.
        # from onnx import load, checker; checker.check_model(load(path))

        # Keep OpenCV from competing for threads
        try:
            cv2.setNumThreads(1)
        except Exception:
            pass

        providers = _prefer_providers(force_cpu)
        sess = ort.InferenceSession(
            path,
            sess_options=_session_options(num_threads=num_threads),
            providers=providers,
        )

        inp = sess.get_inputs()[0]
        in_shape = list(inp.shape)

        if len(in_shape) != 4:
            raise RuntimeError(f"Unsupported input rank: {in_shape}")

        # Resolve dynamic dims (None) conservatively
        # Expect either NCHW or NHWC with 3 channels
        if in_shape[1] == 3 or (in_shape[1] is not None and int(in_shape[1]) == 3):
            layout, H, W = "NCHW", int(in_shape[2] or 224), int(in_shape[3] or 224)
        elif in_shape[3] == 3 or (in_shape[3] is not None and int(in_shape[3]) == 3):
            layout, H, W = "NHWC", int(in_shape[1] or 224), int(in_shape[2] or 224)
        else:
            # Fallback guess
            layout, H, W = "NCHW", 224, 224

        print(
            f"[INFO] ORT input='{inp.name}' expects {layout} {H}x{W}; "
            f"Providers={sess.get_providers()}"
        )
        return ORTModel(sess, inp.name, layout, H, W)



class AUDetector:
    def __init__(self, model_path: str, force_cpu: bool=False):
        self.model = ORTModel.create(model_path, force_cpu=force_cpu)

    def _preprocess(self, frame_bgr):
        H, W = self.model.H, self.model.W
        h, w = frame_bgr.shape[:2]
        scale = max(H / h, W / w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        y0 = (nh - H) // 2
        x0 = (nw - W) // 2
        crop = resized[y0:y0+H, x0:x0+W]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb = (rgb - mean) / std
        if self.model.layout == "NCHW":
            return np.transpose(rgb, (2,0,1))[None,...].astype(np.float32)
        else:
            return rgb[None,...].astype(np.float32)

    def infer(self, frame_bgr):
        inp = self._preprocess(frame_bgr)
        outputs = self.model.session.run(None, {self.model.input_name: inp})
        au = outputs[0][0].astype(np.float32)
        return np.clip(au, 0.0, 1.0)
