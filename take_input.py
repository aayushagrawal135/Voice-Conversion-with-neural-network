import numpy as np
import os
import glob
#from scipy.spatial.distance import euclidean
#from scipy.io import loadmat
#from scipy.io import savemat
#import matplotlib.pyplot as plt
#import tensorflow as tf

def get_training(filename):
	data = loadmat(filename)
	Z = data["Z"]
	X = Z[0:25, :]
	Y = Z[25:, :]
	X = np.transpose(X)
	Y = np.transpose(Y)

	return X, Y

x, y = get_training("/home/aayush/speech/VC_CMU_ARCTIC/Z.mat")
print(np.shape(x))
print(np.shape(y))
