import pickle
import matplotlib.pyplot as plt

with open('mlpGraphData.pkl', 'rb') as f:
	hist = pickle.load(f)
x = list(range(99))

acc = hist['accuracy'][1:]
val_acc = hist['val_accuracy'][1:]

loss = hist['loss'][1:]
val_loss = hist['val_loss'][1:]

fig, ax1 = plt.subplots()

color = 'black'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color=color)
ax1.plot(x, loss, color='red')
ax1.plot(x, val_loss, color='green')
# ax1.tick_params(axis='y', labelcolor=color)

print(len(loss))

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'black'
ax2.set_ylabel('Accuracy', color=color)  # we already handled the x-label with ax1
ax2.plot(x, acc, color='blue')
ax2.plot(x, val_acc, color='purple')
# ax2.tick_params(axis='y', labelcolor=color)

fig.legend(['Training loss', 'Validation loss', 'Training accuracy', 'Validation accuracy'], loc ="upper left")


fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
