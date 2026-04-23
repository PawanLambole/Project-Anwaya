"""
camera_utils.py
───────────────
Smart camera enumeration for Windows.
Uses DirectShow (via pygrabber / win32com / ctypes fallback) to get the REAL
device names shown in Device Manager, then filters out virtual-camera software
(OBS, ManyCam, DroidCam, etc.) so the app always defaults to the integrated
webcam.
"""

import cv2
import re
import sys

# ---------- Keywords that identify virtual / software cameras ----------
VIRTUAL_CAM_KEYWORDS = [
    "obs", "virtual", "droidcam", "ivcam", "manycam", "epoccam",
    "snap camera", "xsplit", "streamlabs", "mmhmm", "ndii",
    "loopback", "iriun", "vcam", "screen capture",
]

# Names that strongly indicate the *real* built-in camera
BUILTIN_CAM_KEYWORDS = [
    "integrated", "internal", "built-in", "builtin",
    "ir camera",                     # Windows Hello IR is still a real sensor
    "hd webcam", "hd camera",        # Dell / Lenovo naming
    "realtek", "bison", "chicony",   # Common OEM webcam manufacturers
    "syntek", "azurewave", "omni",
]


def _get_device_names_via_pygrabber() -> list[str]:
    """Try to get names via pygrabber (pip install pygrabber)."""
    try:
        from pygrabber.dshow_graph import FilterGraph
        graph = FilterGraph()
        return graph.get_input_devices()   # list of str, index-aligned with OpenCV
    except Exception:
        return []


def _get_device_names_via_win32com() -> list[str]:
    """Try to get names via win32com.client (pywin32)."""
    try:
        import win32com.client
        wmi = win32com.client.GetObject("winmgmts:")
        devices = wmi.InstancesOf("Win32_PnPEntity")
        # Win32_PnPEntity doesn't map perfectly to OpenCV indices, but gives names
        cam_names = [d.Name for d in devices
                     if d.Name and "camera" in d.Name.lower()]
        return cam_names
    except Exception:
        return []


def _is_virtual(name: str) -> bool:
    name_lower = name.lower()
    return any(kw in name_lower for kw in VIRTUAL_CAM_KEYWORDS)


def _is_builtin(name: str) -> bool:
    name_lower = name.lower()
    return any(kw in name_lower for kw in BUILTIN_CAM_KEYWORDS)


def enumerate_cameras(max_test: int = 8) -> list[dict]:
    """
    Returns a list of dicts, one per working camera:
        {"index": int, "name": str, "is_virtual": bool, "is_builtin": bool}
    Index-0 in this list is the recommended default camera.
    """
    # 1. Try to get real device names
    device_names = _get_device_names_via_pygrabber()
    if not device_names:
        device_names = _get_device_names_via_win32com()

    cameras = []
    # If we successfully got device names, only test that many indices to save massive startup time
    test_range = len(device_names) if device_names else max_test
    
    for idx in range(test_range):
        dshow = getattr(cv2, 'CAP_DSHOW', 0) if sys.platform == 'win32' else 0
        cap = cv2.VideoCapture(idx, dshow) if dshow else cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            continue

        # Quick read to confirm it's not a ghost device
        ret, _ = cap.read()
        cap.release()
        if not ret:
            continue

        # Assign a name
        if idx < len(device_names) and device_names[idx]:
            name = device_names[idx]
        else:
            name = f"Camera {idx}"

        virtual = _is_virtual(name)
        builtin = _is_builtin(name)
        cameras.append({
            "index": idx,
            "name": name,
            "is_virtual": virtual,
            "is_builtin": builtin,
        })

    # 2. Sort: real built-ins first, then other real cameras, then virtual cams
    def sort_key(cam):
        if cam["is_builtin"]:
            return 0
        if not cam["is_virtual"]:
            return 1
        return 2

    cameras.sort(key=sort_key)
    return cameras


def get_default_camera_index(cameras: list[dict] | None = None) -> int:
    """
    Returns the OS camera index that should be used by default.
    Prefers the integrated webcam; falls back to the first real camera;
    falls back to index 0.
    """
    if cameras is None:
        cameras = enumerate_cameras()

    if not cameras:
        return 0

    # Prefer a built-in
    for cam in cameras:
        if cam["is_builtin"] and not cam["is_virtual"]:
            return cam["index"]

    # Then any non-virtual
    for cam in cameras:
        if not cam["is_virtual"]:
            return cam["index"]

    # Last resort
    return cameras[0]["index"]
