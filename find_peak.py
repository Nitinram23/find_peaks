import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

def maximas(x,threshold):
	maxs = []
	yhat = x
	grad = np.diff(yhat) # finding the first derivative
	
	if np.all(grad>=0): # checking if array is monotonically increasing
		return len(yhat)-1,x[len(yhat)-1]

	if np.all(grad<=0): # checking if array is monotonically decreasing
		return 0,x[0]

	for i in range(len(grad)-1):
		if grad[i]> 0 and grad[i+1]<=0 and yhat[i+1]>=threshold: # check for [+ -] pairs in grad and if the peak is greater than threshold
			maxs.append(i+1)
	if yhat[0]>=yhat[1] and yhat[0]>=threshold:	# condition to check if first element is also a peak
		maxs.insert(0, 0)
	if yhat[len(x)-1]>=yhat[len(x)-2] and yhat[len(x)-1]>=threshold:	# condition to check if last element is also a peak
		maxs.append(len(yhat)-1)
	return maxs,x[maxs]

# test cases
x1 = np.array([9,2,3,2,3,4,5,6,5,7,8,7,6,5,6,4,3,2,2,4,5,9,9,9,9,8,7,6,6,6,6,6,5,4,3,2,1,3,2,4,5,8,10,12,11,9,8,7,6,4,2,10])
threshold1 = 8	# setting minimum threshold to be considered as a peak
	
x2 = np.array([1,2,3,4,5,6,7,8,8,9,10,11]) 
threshold2 = 3	# setting minimum threshold to be considered as a peak

x3 = np.array([10,9,8,7,4,3,2,1,0,-1,-3])
threshold3 = 3	# setting minimum threshold to be considered as a peak

Fs = 1000
f = 5
sample = 1000
x = np.arange(sample)
x4 = np.sin(2 * np.pi * f * x / Fs)
noise = np.random.normal(0,0.3,1000)
x4 = x4+(noise/2) # adding noise to the sine wave generated 
threshold4 = 1	# setting minimum threshold to be considered as a peak
denoise = savgol_filter(x4, 13, 3)	# smoothening the input to account for noise

maxs1,values1 = maximas(x1,threshold1)
plt.plot(x1)
plt.scatter(maxs1,x1[maxs1],facecolor='red')
print(maxs1)

maxs2,values2 = maximas(x2,threshold2)
plt.figure()
plt.plot(x2)
plt.scatter(maxs2,x2[maxs2],facecolor='red')
print(maxs2)

maxs3,values3 = maximas(x3,threshold3)
plt.figure()
plt.plot(x3)
plt.scatter(maxs3,x3[maxs3],facecolor='red')
print(maxs3)

maxs4,values4 = maximas(denoise,threshold4)
plt.figure()
plt.plot(x4)
plt.plot(denoise)
plt.gca().legend(('original signal','filtered signal'))
plt.scatter(maxs4,x4[maxs4],facecolor='red')
print(maxs4)

plt.show()