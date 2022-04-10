import os
from typing import Iterator
from PIL import Image
from skimage import io


PixelValue = tuple[int, int, int]


class InputImagePreprocessor:
    def __init__(self) -> None:
        self.colors_to_exclude = self.__colors_to_exclude()

    def __colors_to_exclude(self) -> Iterator[PixelValue]:
        abspath_black_image = 'resources\\first_party_data\\black.png'
        image = io.imread(abspath_black_image)
        black_pixel = tuple([0, 0, 0])

        unique_pixel_values = {tuple(pixel_value) for row in image for pixel_value in row}
        unique_pixel_values.discard(black_pixel)

        return unique_pixel_values

    def delete_axis(self, abspath_subfile: os.path.abspath) -> None:
        image = Image.open(abspath_subfile)
        pixel_map = image.load()
        width, height = image.size

        for i in range(width):
            for j in range(height):
                pixel_color_value = image.getpixel((i, j))

                if pixel_color_value in self.colors_to_exclude:
                    pixel_map[i, j] = self.__get_mean_pixel_value(image, i, j)

        image.save(abspath_subfile)

    def __get_mean_pixel_value(self, image: Image.Image, i: int, j: int) -> PixelValue:
        sum_r, sum_g, sum_b = 0, 0, 0
        n_valid_neighbors = 0

        for (r, g, b) in self.__get_valid_neighbours_pixel_values(image, i, j):
            sum_r += r
            sum_g += g
            sum_b += b
            n_valid_neighbors += 1

        sum_r //= n_valid_neighbors
        sum_g //= n_valid_neighbors
        sum_b //= n_valid_neighbors

        return sum_r, sum_g, sum_b
    
    def __get_valid_neighbours_pixel_values(self, image: Image.Image, i, j) -> Iterator[PixelValue]:
        neighbour_pixel_coordinates = [
            (i-1, j-1), (i-1, j), (i-1, j-1), 
            (i, j-1), (i, j+1), 
            (i+1, j-1), (i+1, j), (i+1, j+1)]

        neighbour_pixel_values = map(image.getpixel, neighbour_pixel_coordinates)

        return filter(self.__is_valid_pixel_value, neighbour_pixel_values)

    def __is_valid_pixel_value(self, pixel_value: PixelValue) -> bool:
        return pixel_value not in self.colors_to_exclude
