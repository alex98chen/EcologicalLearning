#%matplotlib inline 
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np

fig = plt.figure()
ax = plt.axes()

epochTimeScores = np.load('epoch_loss_scores.npy')
loss_scores = np.load('loss_scores.npy')

print(epochTimeScores.shape)
print(loss_scores.shape)

x = np.linspace(0, 100, 11)
print(x)
print(loss_scores)
for i in range(1,10):
    plt.plot(x, loss_scores[i], label ='{}'.format(i))
plt.title('loss scores for each number by percent of each number')
plt.legend(loc='best')
plt.savefig('lossByPercent.png')
