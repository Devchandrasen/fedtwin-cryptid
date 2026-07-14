"""Piracy-channel transformation catalog."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Transformation:
    name: str
    modality: str
    video_strength: float
    audio_strength: float
    video_reliability: float
    audio_reliability: float


TRANSFORMATIONS: tuple[Transformation, ...] = (
    Transformation("clean_copy", "multimodal", 0.08, 0.08, 0.98, 0.98),
    Transformation("reencode", "video", 0.20, 0.05, 0.92, 0.98),
    Transformation("crop_overlay", "video", 0.34, 0.04, 0.78, 0.98),
    Transformation("speed_change", "multimodal", 0.28, 0.28, 0.80, 0.78),
    Transformation("cam_recording", "video", 0.45, 0.22, 0.62, 0.72),
    Transformation("pitch_shift", "audio", 0.05, 0.32, 0.97, 0.74),
    Transformation("audio_noise", "audio", 0.04, 0.38, 0.98, 0.66),
    Transformation("audio_replaced", "multimodal", 0.18, 0.95, 0.88, 0.08),
    Transformation("partial_segment", "multimodal", 0.42, 0.42, 0.70, 0.70),
    Transformation("hard_negative_same_topic", "multimodal", 0.90, 0.90, 0.95, 0.95),
)


def transformation_names() -> list[str]:
    return [t.name for t in TRANSFORMATIONS]


def get_transformation(name: str) -> Transformation:
    for item in TRANSFORMATIONS:
        if item.name == name:
            return item
    raise KeyError(name)
