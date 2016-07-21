def evaluate_accuracy(features, labels, model):
  """
  Evaluate accuracy on a list of image files
  Inputs:
  - filelist: a list of image filenames
  - labels: a list of ground truth labels corresponding to the filelist
  - model: the methold will call model.predict on each image in the filelist
  Outputs:
  - predictions: a list of predicted labels corresponding to the filelist
  - accuracy: computed accuracy of the predictions based on the ground truth labels

  """
  feat_size, num_images = features.shape
  num_correct = 0
  num_true_pos = 0
  num_pos_preds = 0
  num_pos_labels = 0
  num_pos_labels = 0

  scores = []
  predictions = model.test(features)
  for img_idx in xrange(num_images):

    image_label = labels[img_idx]
    image_prediction = predictions[0,img_idx]
    #print image_label
    #print image_prediction

    correct = (image_label == image_prediction)
    if correct:
        num_correct += 1
    if (correct and image_label == 1):
        num_true_pos += 1
    if (image_prediction == 1):
        num_pos_preds += 1
    if (image_label == 1):
        num_pos_labels += 1

  accuracy = float(num_correct) / num_images
  precision = float(num_true_pos) / (num_pos_preds + 1e-12)
  recall = float(num_true_pos) / (num_pos_labels + 1e-12)
  fscore = (2*precision*recall) / (precision + recall + 1e-12)
  return predictions, scores, accuracy, precision, recall, fscore



