import sys

target_file = sys.argv[1]
pred_file = sys.argv[2]

fin = open(target_file, 'r')
lines = fin.readlines()
fin.close()
ground_fns = []
ground_labels = []
for line in lines[1:]:
	fn, label = line.split(',')
	ground_fns.append(fn)
	ground_labels.append(int(label))

fin = open(pred_file, 'r')
lines = fin.readlines()
fin.close()
pred_fns = []
pred_labels = []
for line in lines[1:]:
	fn, label = line.split(',')
	pred_fns.append(fn)
	pred_labels.append(int(label))

assert len(pred_labels) == len(ground_labels)

correct = 0
for i in range(len(pred_labels)):
	assert pred_fns[i] == ground_fns[i]
	if pred_labels[i] == ground_labels[i]:
		correct += 1
print('AVG ACCURACY: {}'.format(correct/len(pred_labels)))