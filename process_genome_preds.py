import numpy as np

chrom = 21
arr = np.load("/datadrive/project_data/genomeIndelPredictions{}.npy".format(chrom))

bsize = 10000
sumprobs = True
arr2 = np.array_split(arr[:(len(arr)//bsize)*bsize], len(arr)//bsize) # split the first part that is divisible by bsize
arr2.append(arr[(len(arr)//bsize)*bsize:]) # tack on the last piece

thresh = 0.8
avg_sum = 0
avg_pred = 0
sumcount = 0
results = [[], []]
probs = []
for window in arr2:
  num_indels_true = np.sum(window[:, 1])
  avg_sum += num_indels_true
  sumcount += 1
  num_indels_pred = np.sum(window[:, 2] > thresh)
  sumprobs = np.sum(window[:, 2])
  avg_pred += num_indels_pred
  results[0].append(num_indels_true)
  results[1].append(num_indels_pred)
  probs.append(sumprobs)

results[1] = np.array(results[1]) * avg_sum / avg_pred

print(results[0][:50])
print(results[1][:50])

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#from sklearn import metrics
from scipy import stats
from sklearn import linear_model

#r2 = metrics.r2_score(results[0], results[1]) #this givse bizarre values?

# we should really do this trained on only some of the datapoints? FIXED
pred = results[1]
print(pred[:50])
r, p = stats.pearsonr(results[0], pred)
print(r)
print(p)

regr = linear_model.LinearRegression()
regr.fit(np.expand_dims(results[0], axis=1), pred)
reg_pred = regr.predict(np.expand_dims(results[0], axis=1))

plt.scatter(results[0], pred)
plt.xlabel('True number of indels')
plt.ylabel('Predicted number of indels')
plt.title('Predicted vs. actual indel mutation rates ($r = {:.2f}'.format(r) + ', p < 10^{-12}$)')
plt.plot(results[0], reg_pred, color='m', linewidth=2.5)
plt.savefig('indel_rate_pred.png')
