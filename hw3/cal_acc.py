fin = open('output.csv', 'r')
lines = fin.readlines()
fin.close()
correct = 0
total = 0
for line in lines[1:]:
	fname, predict = line.split(',')
	label = fname.split('_')[0]
	if int(predict) == int(label):
		correct += 1
	total += 1
print('Avg acc: {}'.format(correct/total))