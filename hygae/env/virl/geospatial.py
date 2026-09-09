"""Small dependency-free geospatial helpers used by the VIRL environment."""

import math


EARTH_RADIUS_METERS = 6_371_008.8


def distance_meters(point_a, point_b) -> float:
    """Return great-circle distance between two ``(latitude, longitude)`` pairs."""
    lat_a, lon_a = map(math.radians, point_a)
    lat_b, lon_b = map(math.radians, point_b)
    delta_lat = lat_b - lat_a
    delta_lon = lon_b - lon_a
    haversine = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat_a) * math.cos(lat_b) * math.sin(delta_lon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_METERS * math.asin(min(1.0, math.sqrt(haversine)))
