labels_csv = '../../data/train/labels.csv'
labels_txt = '../../data/train/labels.txt'

with open(labels_csv) as f:
  data = f.read().strip().split('\n')
data = data[1::]

labels = [x.split(',')[1] for x in data]

lf = open(labels_txt, 'w')
for label in labels:
  lf.write(label + '\n')
lf.close()

'''
To read labels txt file, do:

with open(labels_txt) as f:
  labels = f.read().strip().split('\n')
labels = [int(x) for x in labels]
'''
