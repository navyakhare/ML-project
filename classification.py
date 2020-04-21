import numpy as np
import matplotlib.pyplot as plt
import simtk.openmm as mm

import pandas as pd 
from pandas import DataFrame 
from sklearn import datasets 

from sklearn.covariance import EllipticEnvelope

import pickle

inputData = np.loadtxt('input_ANC.dat',dtype='str')

colvar = np.loadtxt(inputData[0])
version = inputData[1]

sumabs = colvar[:,7]

train = pd.DataFrame(sumabs) 

estimator_EE = EllipticEnvelope()

estimator_EE.fit(train)

filename = 'EE_model_'+str(version)+'.sav'
pickle.dump(estimator_EE, open(filename, 'wb'))

