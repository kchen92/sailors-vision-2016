import numpy as np
from skimage import io

class LinearClassifier:
     
  def compute_loss(self, features, labels, W):
    return 0
    
  def train(self, features, labels, feature_scale, step_size, num_iterations):
    feat_size, num_examples = features.shape
    W = np.random.randn(1, feat_size) * feature_scale
    self.W = W       
    
  def test(self, features):
    feat_size, num_examples = features.shape
    scores = self.W.dot(features)
    probs = 1.0 / (1.0 + np.exp (-1 * scores))
    predictions = probs > 0.5
    return predictions