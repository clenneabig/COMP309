import pickle

with open('vgg116_shit.txt', 'r') as f:
	lines = f.readlines()

acc = []
val_acc = []
loss = []
val_loss = []

for line in lines:
	if 'Epoch' not in line:
		split = line.split()
		for i, s in enumerate(split):
			if s == 'loss:':
				loss.append(float(split[i+1]))
			elif s == 'accuracy:':
				acc.append(float(split[i+1]))
			elif s == 'val_loss:':
				val_loss.append(float(split[i+1]))
			elif s == 'val_accuracy:':
				val_acc.append(float(split[i+1]))

hist = {'accuracy':acc, 'loss':loss, 'val_accuracy':val_acc, 'val_loss':val_loss}

with open('vgg16history.pkl', 'wb') as f:
	pickle.dump(hist, f) 