import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Get Classification Plots')
parser.add_argument('--file', type=str, default='file.out.txt', metavar='F', help='Path to output file')
args = parser.parse_args()

fp = open(args.file)

train_loss = []
val_loss = []
curr_epoch = []
for line in fp.readlines():
	if 'Train Epoch' in line:
		loss = line.split(' ')[-1]
		curr_epoch.append(float(loss))
	elif 'Validation set' in line:
		train_loss.append(sum(curr_epoch) / len(curr_epoch))
		curr_epoch = []
		loss = line.split(',')[0].split(' ')[-1]
		val_loss.append(float(loss))

# plt.ylim((0, 1))
plt.plot(train_loss)
plt.title('Training Loss per Epoch (Cross Entropy)')
plt.show()

# plt.ylim((0, 1))
plt.plot(val_loss)
plt.title('Validation Loss per Epoch (Cross Entropy)')
plt.show()
