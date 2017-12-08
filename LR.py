import numpy as np 
import pandas as pd
import utilityML as ut

TRAIN_DIR = 'train.csv'
TEST_DIR = 'test.csv'
ALPHA = 3e-3
NUM_EPOCHS = 1000

# def getRegularized(X):
# 	nEntries = X.shape[0]
# 	Xsum = sum(X)
# 	Xaverage = Xsum/nEntries
# 	XMax = np.amax(X, axis = 0)
# 	XMin= np.amin(X,axis = 0)
# 	XReg = (X - Xaverage)/(XMax - XMin)
# 	return XReg

# def getDataset(DIR):
# 	df = pd.read_csv(DIR)
# 	df = np.array(df, np.float32)
# 	X = df[:,:-1]
# 	y = df[:, -1]
# 	return X, y

# def appendOnes(X):
# 	o = np.ones((X.shape[0],1))
# 	o = np.array(o, np.float32)
# 	X = np.hstack([o,X])
# 	return X

def getHypothesis(theta,X):
	hTheta = np.matmul(X,theta)
	print X
	return ut.sigmoid(hTheta)

# def sigmoid(X):
# 	sig = 1.0/(1.0 + np.exp(-1.0*X))
# 	return sig

def getCost(y_predict, y_real):
	error = 0.5*np.average(-y_real*(np.log(y_predict))-(1-y_real)*np.log(1-y_predict))
	return error

def optimizeTheta(X, theta, y_predict, y_real):
	for i in range(X.shape[0]):
		theta += ALPHA*(y_real[i] - y_predict[i])*X[i] #or theta += alpha*(y-prediction)*(1-prediction)*prediction*x
	return theta

train_X, train_y = ut.getDataset(TRAIN_DIR)
train_X = np.array(train_X, np.float32)
train_X = ut.getRegularized(train_X)
train_X = ut.appendOnes(train_X)
theta = np.zeros(train_X.shape[1])

for i in range(NUM_EPOCHS):
	hypothesis = getHypothesis(theta, train_X)
	error = getCost(hypothesis, train_y)
	theta = optimizeTheta(train_X, theta, hypothesis, train_y)
	print "Epoch = {} Error = {}".format(i,error)

print "Training Completed"

test_X, test_y = ut.getDataset(TEST_DIR)
test_X = np.array(test_X, np.float32)
test_X = ut.getRegularized(test_X)
test_X = ut.appendOnes(test_X)
y_predict = getHypothesis(theta, test_X)
correct = 0.
for i in range(test_y.shape[0]):
	if test_y[i] == 1 and y_predict[i] > 0.5:
		correct += 1
	if test_y[i] == 0 and y_predict[i] <= 0.5:
		correct += 1
accuracy = 100.0*correct/test_y.shape[0]
print "Test Accuracy : ", accuracy