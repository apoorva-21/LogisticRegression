import numpy as np
import pandas as pd

def getRegularized(X):
	nEntries = X.shape[0]
	Xsum = sum(X)
	Xaverage = Xsum/nEntries
	XMax = np.amax(X, axis = 0)
	XMin= np.amin(X,axis = 0)
	XReg = (X - Xaverage)/(XMax - XMin)
	return XReg

def getDataset(DIR):
	df = pd.read_csv(DIR)
	df = np.array(df, np.float32)
	X = df[:,:-1]
	y = df[:, -1]
	return X, y

def appendOnes(X):
	o = np.ones((X.shape[0],1))
	o = np.array(o, np.float32)
	X = np.hstack([o,X])
	return X

def sigmoid(X):
	sig = 1.0/(1.0 + np.exp(-1.0*X))
	return sig