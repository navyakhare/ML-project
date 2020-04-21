import numpy as np

import pandas as pd 
from pandas import DataFrame 
from sklearn import datasets 

from sklearn.covariance import EllipticEnvelope

import pickle

inputData = np.loadtxt('EE_model_list.dat',dtype='str')

colvar = np.loadtxt('COLVAR')
sumabs = colvar[:,7]

test = pd.DataFrame(sumabs) 

version="-1"
new_colvar=np.array([])
new_colvar_out=np.array([])
for name in inputData:	
	estimator_EE = pickle.load(open(name, 'rb'))
	results = estimator_EE.predict(test)

	inCount=0
	outCount=0
	totCount= len(results)
	print('total count - ',len(results))
	inData=[]
	outData=[]
	for i in range(totCount):
		if(results[i]==1): 
			inCount+=1
			inData.append(colvar[i,:])
		else:
			outCount+=1
			outData.append(colvar[i,:])

	print('inCount, outCount - ',inCount,outCount)
	percentage = float((float(inCount)/float(totCount))*100)
	print('percentage - ',percentage)
	new_colvar_out=np.array(outData)
	if(percentage>float(60)):
		version=str(name[9])
		c=10
		while(str(name[c])!="."):
			version += str(name[c])
			c+=1		
		print(version)
		new_colvar=np.array(inData)
		break


if(version!="-1"):
	a = np.loadtxt('COLVAR_anc_'+str(version))
	c=[]
	for i in a:
		c.append(i)
	for i in new_colvar:
		c.append(i)
	c=np.array(c)
	print('length of new COLVAR - ', len(c))
	np.savetxt('COLVAR_anc_'+str(version),c)
	ANC_list=open('input_ANC.dat','w')
	ANC_list.write('COLVAR_anc_'+str(version)+'\n')
	ANC_list.write(str(version))

elif(version=="-1"):
	
	version=len(inputData)-1
	
	np.savetxt('COLVAR_anc_'+str(version),colvar)
	ANC_list=open('input_ANC.dat','w')
	ANC_list.write('COLVAR_anc_'+str(version)+'\n')
	ANC_list.write(str(version))
	
	EE_model_list=open('EE_model_list.dat','a')
	EE_model_list.write('EE_model_'+str(version)+'.sav\n')



