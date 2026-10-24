from __future__ import annotations

from copy import deepcopy
from typing import Any

AMTOWN_CALIBRATION = {
    "camera_intrinsic": [1453.72, 0.0, 1172.18, 0.0, 1453.28, 1041.78, 0.0, 0.0, 1.0],
    "camera_dist_coeffs": [-0.121, 0.1113, 0.0016, 0.00013, -0.062353],
    "camera_ext_R": [
        0.00298088,
        -0.999728,
        -0.0231416,
        -0.00504636,
        0.0231263,
        -0.99972,
        0.999983,
        0.00309683,
        -0.00497605,
    ],
    "camera_ext_t": [0.0025563, 0.0567484, -0.0512149],
}

AMVALLEY_CALIBRATION = {
    "camera_intrinsic": [1453.88, 0.0, 1182.53, 0.0, 1452.85, 1045.82, 0.0, 0.0, 1.0],
    "camera_dist_coeffs": [-0.052, 0.1168, 0.0015, 0.00013, -0.068564],
    "camera_ext_R": [
        0.00298068,
        -0.999735,
        -0.0231428,
        -0.00504595,
        0.023132,
        -0.99974,
        0.999985,
        0.00309701,
        -0.00497598,
    ],
    "camera_ext_t": [-0.0025563, 0.0567484, -0.0512149],
}

HK_GNSS_CALIBRATION = {
    "camera_intrinsic": [1444.43, 0.0, 1179.50, 0.0, 1444.34, 1044.90, 0.0, 0.0, 1.0],
    "camera_dist_coeffs": [-0.0560, 0.1180, 0.00122, 0.00064, -0.0627],
    "camera_ext_R": [
        0.00363212,
        -0.999819,
        -0.0213618,
        -0.0111679,
        0.0214512,
        -0.999591,
        0.999879,
        0.00375613,
        -0.0111134,
    ],
    "camera_ext_t": [-0.0021928, 0.0470312, -0.0513126],
}

HKISLAND_CALIBRATION = {
    "camera_intrinsic": [1444.43, 0.0, 1177.8, 0.0, 1444.34, 1043.6, 0.0, 0.0, 1.0],
    "camera_dist_coeffs": [-0.053, 0.121, 0.00127, 0.00043, -0.06495],
    "camera_ext_R": [
        0.00352762,
        -0.999765,
        -0.0213775,
        -0.0111803,
        0.0213369,
        -0.99971,
        0.999931,
        0.00376561,
        -0.0111025,
    ],
    "camera_ext_t": [-0.0025563, 0.0470454, -0.0513375],
}

FEATURELESS_GNSS_CALIBRATION = {
    "camera_intrinsic": [1450.09, 0.0, 1177.5, 0.0, 1450.09, 1044.5, 0.0, 0.0, 1.0],
    "camera_dist_coeffs": [-0.0558, 0.1247, 0.00126, 0.0007, -0.06762],
    "camera_ext_R": [
        0.00941312,
        -0.999586,
        -0.0271775,
        -0.00737429,
        0.0271086,
        -0.999605,
        0.999929,
        0.00960988,
        -0.00711613,
    ],
    "camera_ext_t": [-0.0025563, 0.0567484, -0.0512149],
}

SCENE_NAME_TO_CALIBRATION: dict[str, dict[str, Any]] = {
    "AMtown01": AMTOWN_CALIBRATION,
    "AMtown02": AMTOWN_CALIBRATION,
    "AMtown03": AMTOWN_CALIBRATION,
    "AMvalley01": AMVALLEY_CALIBRATION,
    "AMvalley02": AMVALLEY_CALIBRATION,
    "AMvalley03": AMVALLEY_CALIBRATION,
    "HKairport01": HK_GNSS_CALIBRATION,
    "HKairport02": HK_GNSS_CALIBRATION,
    "HKairport03": HK_GNSS_CALIBRATION,
    "HKisland01": HKISLAND_CALIBRATION,
    "HKisland02": HKISLAND_CALIBRATION,
    "HKisland03": HKISLAND_CALIBRATION,
    "HKairport_GNSS_Evening": HK_GNSS_CALIBRATION,
    "HKisland_GNSS_Evening": FEATURELESS_GNSS_CALIBRATION,
    "HKairport_GNSS01": HK_GNSS_CALIBRATION,
    "HKairport_GNSS02": HK_GNSS_CALIBRATION,
    "HKairport_GNSS03": HK_GNSS_CALIBRATION,
    "HKisland_GNSS01": HK_GNSS_CALIBRATION,
    "HKisland_GNSS02": HK_GNSS_CALIBRATION,
    "HKisland_GNSS03": HK_GNSS_CALIBRATION,
    "Featureless_GNSS01": FEATURELESS_GNSS_CALIBRATION,
    "Featureless_GNSS02": FEATURELESS_GNSS_CALIBRATION,
    "Featureless_GNSS03": FEATURELESS_GNSS_CALIBRATION,
}


def has_calibration(scene_name: str) -> bool:
    return scene_name in SCENE_NAME_TO_CALIBRATION


def get_calibration(scene_name: str) -> dict[str, Any]:
    if scene_name not in SCENE_NAME_TO_CALIBRATION:
        raise KeyError(f"No UAVScenes calibration registered for scene: {scene_name}")
    return deepcopy(SCENE_NAME_TO_CALIBRATION[scene_name])


def list_calibrated_scenes() -> list[str]:
    return sorted(SCENE_NAME_TO_CALIBRATION)
