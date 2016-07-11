import numpy as np
from skimage import io

class ThresholdModel:

  # set a reference matrix
  def set_ref_feature(self, ref_img):
    self.ref_mat = self.extract_features(ref_img)

  # set thresholds
  def set_pixel_change_threshold(self, threshold):
    self.pixel_change_threshold = threshold
    
  def set_image_score_threshold(self, threshold):
    self.image_score_threshold = threshold

  def set_crop(self, xstart, xend, ystart, yend):
    self.cropx_start = xstart
    self.cropx_end = xend
    self.cropy_start = ystart
    self.cropy_end = yend 

  def set_n_cells(self, n_cells_y, n_cells_x):
    self.n_cells_x = n_cells_x
    self.n_cells_y = n_cells_y

  def set_feature_extractor(self, new_feature_extractor):
    self.extract_features = new_feature_extractor

  def get_crop_feature(self, img):
    pass

  def get_grid_feature(self, img):
    pass

  def get_full_image_feature(self, img):
    return img

  def predict(self, image_file):
    """
    Predict the label for an image
    Inputs:
    - image_file: filename of the image to predict
    Outputs:
    - return 1 if the image is positive, 0 if negative
    """
    pass