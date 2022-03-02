import numpy as np
from matplotlib import cm, pyplot as plt

from src.utils.constants import IMG_DIM
from src.utils.vector_transformations import normalize


class Cut3dViewVisualization:
    @staticmethod
    def __explode(data: np.ndarray) -> np.ndarray:
        shape_arr = np.array(data.shape)
        size = shape_arr[:3] * 2 - 1
        exploded = np.zeros(np.concatenate([size, shape_arr[3:]]), dtype=data.dtype)
        exploded[::2, ::2, ::2] = data
        return exploded

    @staticmethod
    def __expand_coordinates(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, y, z = indices

        x[1::2, :, :] += 1
        y[:, 1::2, :] += 1
        z[:, :, 1::2] += 1

        return x, y, z

    @classmethod
    def plot_cube(cls, cube: np.ndarray, angle=320) -> None:
        cube = normalize(cube)

        face_colors = cm.viridis(cube)
        face_colors[:, :, :, -1] = cube
        face_colors = cls.__explode(face_colors)

        filled = face_colors[:, :, :, -1] != 0
        x, y, z = cls.__expand_coordinates(np.indices(np.array(filled.shape) + 1))

        fig = plt.figure(figsize=(30 / 2.54, 30 / 2.54))
        ax = fig.gca(projection='3d')
        ax.view_init(30, angle)
        ax.set_xlim(right=IMG_DIM * 2)
        ax.set_ylim(top=IMG_DIM * 2)
        ax.set_zlim(top=IMG_DIM * 2)

        ax.voxels(x, y, z, filled, facecolors=face_colors, shade=False)
        plt.show()
