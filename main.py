from src.data_processing.data_resizer import DataResizer
from src.data_visualization.cut_3d_view_visualization import Cut3dViewVisualization
from src.gata_gathering.image_loader import ImageLoader


arr = ImageLoader.get_data('20050317215037-41818.MRC15540.nii.gz')
resized_data = DataResizer.custom_resize(arr)
Cut3dViewVisualization.plot_cube(resized_data[:35, ::-1, :25])
