"""Offline-only Street View panorama store for VIRL.

The data format is the one published with SFTvsRL: a pickle mapping GPS tuples
to panorama IDs, ``<pano_id>.jpg`` equirectangular images, and adjacent
``<pano_id>.metadata.json`` files containing the north ``rotation``.
"""

from __future__ import annotations

import json
import math
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from .geospatial import distance_meters


class OfflinePanoramaStore:
    def __init__(
        self,
        panorama_dir: str,
        mapping_path: str,
        *,
        mapping_radius: float = 50.0,
        view_size: tuple[int, int] = (640, 640),
        fov: float = 90.0,
        pitch: float = 0.0,
    ):
        self.panorama_dir = Path(panorama_dir).expanduser().resolve()
        self.mapping_path = Path(mapping_path).expanduser().resolve()
        if not self.panorama_dir.is_dir():
            raise FileNotFoundError(f"VIRL panorama directory not found: {self.panorama_dir}")
        if not self.mapping_path.is_file():
            raise FileNotFoundError(f"VIRL GPS-to-panorama mapping not found: {self.mapping_path}")
        with self.mapping_path.open("rb") as handle:
            mapping = pickle.load(handle)
        self.gps_to_pano_mapping = {
            tuple(float(component) for component in geocode): str(pano_id)
            for geocode, pano_id in mapping.items()
        }
        if not self.gps_to_pano_mapping:
            raise ValueError(f"Empty VIRL panorama mapping: {self.mapping_path}")
        self._geocodes = tuple(self.gps_to_pano_mapping)
        self.mapping_radius = float(mapping_radius)
        self.view_size = tuple(int(component) for component in view_size)
        self.fov = float(fov)
        self.pitch = float(pitch)

    @classmethod
    def from_env_config(cls, config):
        platform = config.platform_cfg or {}
        offline = platform.get("OFFLINE", {})
        street_view = platform.get("STREET_VIEW", {})
        panorama_dir = config.panorama_dir or offline.get("PANORAMA_DIR")
        mapping_path = config.gps_to_pano_path or offline.get("GPS_TO_PANO_PATH")
        if not panorama_dir or not mapping_path:
            raise ValueError(
                "VIRL requires panorama_dir and gps_to_pano_path (directly or under "
                "platform_cfg.OFFLINE); online fallback is intentionally unsupported"
            )
        return cls(
            panorama_dir,
            mapping_path,
            mapping_radius=config.mapping_radius or offline.get("MAPPING_RADIUS", 50.0),
            view_size=tuple(street_view.get("SIZE", (640, 640))),
            fov=street_view.get("FOV", 90.0),
            pitch=street_view.get("PITCH", 0.0),
        )

    def relocate(self, geocode) -> tuple[tuple[float, float], str]:
        geocode = tuple(float(component) for component in geocode)
        return self._relocate_cached(geocode)

    @lru_cache(maxsize=8192)
    def _relocate_cached(self, geocode) -> tuple[tuple[float, float], str]:
        pano_id = self.gps_to_pano_mapping.get(geocode)
        if pano_id is not None:
            return geocode, pano_id
        nearest = min(self._geocodes, key=lambda candidate: distance_meters(geocode, candidate))
        distance = distance_meters(geocode, nearest)
        if distance > self.mapping_radius:
            raise LookupError(
                f"No offline VIRL panorama within {self.mapping_radius:g} m of {geocode}; "
                f"nearest is {distance:.2f} m away"
            )
        return nearest, self.gps_to_pano_mapping[nearest]

    def pano_id(self, geocode) -> str:
        return self.relocate(tuple(geocode))[1]

    @lru_cache(maxsize=256)
    def _load_panorama(self, pano_id: str):
        image_path = self.panorama_dir / f"{pano_id}.jpg"
        metadata_path = self.panorama_dir / f"{pano_id}.metadata.json"
        if not image_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(
                f"Incomplete VIRL panorama {pano_id}: expected {image_path.name} and "
                f"{metadata_path.name} in {self.panorama_dir}"
            )
        with Image.open(image_path) as image:
            panorama = np.asarray(image.convert("RGB"))
        with metadata_path.open() as handle:
            north_rotation = float(json.load(handle)["rotation"])
        return panorama, north_rotation

    def render(self, geocode, heading: float) -> Image.Image:
        relocated, pano_id = self.relocate(tuple(geocode))
        panorama, north_rotation = self._load_panorama(pano_id)
        return _perspective_view(
            panorama,
            fov=self.fov,
            heading=float(heading),
            pitch=self.pitch,
            width=self.view_size[0],
            height=self.view_size[1],
            north_rotation=north_rotation,
        )

    def render_views(self, geocode, headings) -> list[Image.Image]:
        return [self.render(geocode, heading) for heading in headings]


def _perspective_view(
    panorama: np.ndarray,
    *,
    fov: float,
    heading: float,
    pitch: float,
    width: int,
    height: int,
    north_rotation: float,
) -> Image.Image:
    """Project an equirectangular panorama into a perspective view."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - covered by install preflight
        raise RuntimeError("VIRL visual mode requires opencv-python-headless") from exc

    adjusted_heading = ((heading - 180.0) % 360.0 + north_rotation) % 360.0
    focal = 0.5 * width / math.tan(0.5 * math.radians(fov))
    intrinsic = np.array(
        [[focal, 0, (width - 1) / 2], [0, focal, (height - 1) / 2], [0, 0, 1]],
        dtype=np.float32,
    )
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    xyz = np.stack((x, y, np.ones_like(x)), axis=-1) @ np.linalg.inv(intrinsic).T
    yaw, _ = cv2.Rodrigues(np.array([0.0, math.radians(adjusted_heading), 0.0], dtype=np.float32))
    pitch_axis = yaw @ np.array([1.0, 0.0, 0.0], dtype=np.float32)
    pitch_rotation, _ = cv2.Rodrigues(pitch_axis * math.radians(pitch))
    xyz = xyz @ (pitch_rotation @ yaw).T
    xyz /= np.linalg.norm(xyz, axis=-1, keepdims=True)
    longitude = np.arctan2(xyz[..., 0], xyz[..., 2])
    latitude = np.arcsin(xyz[..., 1])
    map_x = ((longitude / (2 * np.pi) + 0.5) * (panorama.shape[1] - 1)).astype(np.float32)
    map_y = ((latitude / np.pi + 0.5) * (panorama.shape[0] - 1)).astype(np.float32)
    view = cv2.remap(
        panorama,
        map_x,
        map_y,
        cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_WRAP,
    )
    return Image.fromarray(view)
