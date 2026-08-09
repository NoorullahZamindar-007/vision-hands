"""Built-in BGR image effects used by the portal renderer."""

import cv2
import numpy as np

_K2 = np.ones((2, 2), np.uint8)


def thermal(image):
    return cv2.applyColorMap(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)


def posterize_neon(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.8, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * 1.1, 0, 255)
    boosted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return ((boosted // 64) * 64 + 32).astype(np.uint8)


def edge_mesh(image):
    edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 60, 150)
    edges = cv2.dilate(edges, _K2, iterations=1)
    base = np.full_like(image, (20, 20, 20))
    base[edges > 0] = (255, 255, 255)
    return base


def invert_glitch(image):
    blue, green, red = cv2.split(255 - image)
    return cv2.merge([np.roll(blue, -6, axis=1), green, np.roll(red, 6, axis=1)])


def grayscale_sketch(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(255 - gray, (21, 21), 0)
    return cv2.cvtColor(cv2.divide(gray, 255 - blur, scale=256), cv2.COLOR_GRAY2BGR)


def chromatic(image):
    blue, green, red = cv2.split(image)
    shifted = cv2.merge([np.roll(blue, -8, axis=1), green, np.roll(red, 8, axis=1)])
    hsv = cv2.cvtColor(shifted, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.25, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def pixel_mosaic(image):
    height, width = image.shape[:2]
    block = 14
    small = cv2.resize(image, (max(1, width // block), max(1, height // block)), interpolation=cv2.INTER_AREA)
    output = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
    seam = np.array([184], np.uint8)
    output[::block, :] = cv2.multiply(output[::block, :], seam, scale=1 / 255)
    output[:, ::block] = cv2.multiply(output[:, ::block], seam, scale=1 / 255)
    return output


def _duotone_lut():
    dark = np.array([32, 24, 90], np.float32)
    light = np.array([255, 196, 64], np.float32)
    amount = np.linspace(0, 1, 256, dtype=np.float32)[:, None]
    return (amount * light + (1 - amount) * dark).astype(np.uint8)


_DUOTONE_LUT = _duotone_lut()


def duotone(image):
    return _DUOTONE_LUT[cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)]


EFFECTS = [thermal, edge_mesh, posterize_neon, invert_glitch, grayscale_sketch, chromatic, pixel_mosaic, duotone]
EFFECT_NAMES = ["Thermal", "Edge Mesh", "Posterize Neon", "Invert Glitch", "Sketch", "Chromatic", "Pixel Mosaic", "Duotone"]
