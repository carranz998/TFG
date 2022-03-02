from src.data_gathering.image_loader import ImageLoader
from src.data_processing.processing_pipeline import ProcessingPipeline
from src.data_visualization.cut_3d_view_visualization import Cut3dViewVisualization


raw_image_data = ImageLoader.get_image_data('20050317215037-41818.MRC15540.nii.gz')
processed_image_data = ProcessingPipeline.process(raw_image_data)
Cut3dViewVisualization.plot_cube(processed_image_data[:35, ::-1, :25])
