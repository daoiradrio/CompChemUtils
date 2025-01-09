import numpy as np



def rotate(from_coords: np.array, to_coords: np.array, coords: np.array, deg: float) -> np.array:
    axis = (to_coords - from_coords) / np.linalg.norm(to_coords - from_coords)
    angle = np.deg2rad(deg)
    new_coords = coords - from_coords
    new_coords = np.dot(axis, np.dot(axis, new_coords)) \
               + np.cos(angle) * np.cross(np.cross(axis, new_coords), axis) \
               + np.sin(angle) * np.cross(axis, new_coords)
    new_coords = new_coords + from_coords
    return new_coords
