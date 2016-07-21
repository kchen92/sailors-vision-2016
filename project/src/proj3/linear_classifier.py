import numpy as np
from skimage import io

class LinearClassifier:
     
  def compute_loss(self, features, labels, W):
    num_examples, feat_size = features.shape
    scores = W.dot(features)
    probs = 1.0 / (1.0 + np.exp (-1 * scores))
    log_likelihood = ?
    total_loss = -1.0*log_likelihood / num_examples
    d_W = ?
    return total_loss, log_likelihood, d_W
    
  def train(self, features, labels, feature_scale, step_size, num_iterations):
    feat_size, num_examples = features.shape
    self.W = np.random.randn(1, feat_size) * feature_scale
    bestloss = float("inf")
    d_W = 0
    for i in xrange(num_iterations):
      Wtry = self.W + d_W*step_size
      loss, log_likelihood, d_W = self.compute_loss(features, labels, Wtry)
      self.W = Wtry
      if loss < bestloss:
        bestloss = loss
      
      print 'iter %d, cur loss: %f, best loss: %f, log likelihood: %f' %(i, loss, bestloss, log_likelihood)
      
      # evaluate fscore
      predictions = self.test(features)
      num_correct = 0
      num_true_pos = 0
      num_pos_preds = 0
      num_pos_labels = 0
      num_pos_labels = 0
      for img_idx in xrange(num_examples):
        image_label = labels[img_idx]
        image_prediction = predictions[0,img_idx]
        correct = (image_label == image_prediction)
        if correct:
          num_correct += 1
        if (correct and image_label == 1):
          num_true_pos += 1
        if (image_prediction == 1):
          num_pos_preds += 1
        if (image_label == 1):
          num_pos_labels += 1

      accuracy = float(num_correct) / num_examples
      precision = float(num_true_pos) / (num_pos_preds + 1e-12)
      recall = float(num_true_pos) / (num_pos_labels + 1e-12)
      fscore = (2*precision*recall) / (precision + recall + 1e-12)
      print 'iter %d, acc: %.3f, prec: %.3f, rec: %.3f, fscore: %.3f' %(i, accuracy, precision, recall, fscore)
    
  def test(self, features):
    feat_size, num_examples = features.shape
    scores = self.W.dot(features)
    probs = 1.0 / (1.0 + np.exp (-1 * scores))
    predictions = probs > 0.5
    return predictions

