predict_filename = 'output_p2.csv'
ground_filename = 'hw4_data/office/val.csv'

fin = open(predict_filename, 'r')
lines_pred = fin.readlines()
fin.close()

fin = open(ground_filename, 'r')
lines_ground = fin.readlines()
fin.close()

correct = 0
cnt = 0
for line1, line2 in zip(lines_pred[1:], lines_ground[1:]):
	id1, fn1, label1 = line1.split(',')
	id2, fn2, label2 = line2.split(',')
	assert id1 == id2
	assert fn1 == fn2
	label1, label2 = label1.strip(), label2.strip()
	if label1 == label2:
		correct += 1
	cnt += 1
print('VAL acc: {}'.format(correct/cnt))

