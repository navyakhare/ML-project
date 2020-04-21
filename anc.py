import numpy as np
import matplotlib.pyplot as plt
import simtk.openmm as mm

import pandas as pd 
from pandas import DataFrame 
from sklearn import datasets 
import pickle


inputData = np.loadtxt('input_ANC.dat',dtype='str')

colvar = np.loadtxt(inputData[0])
version = inputData[1]

data = colvar[:,1:7]
sumabs = colvar[:,7]

sincos=[]

for i in range(0,len(data)):
    temp=[]
    for j in range(0,6):
        temp.append(np.sin(data[i,j]))
        temp.append(np.cos(data[i,j]))
    sincos.append(temp)

diheds_sincos=np.array(sincos)

X = sumabs
X_dihed = diheds_sincos

xnew_sincos=[]
xnew=[]

for i in range(0,len(X)):
    t1=[]
    t1.append(X[i])
    xnew.append(t1)
    t3=[]
    for j in range(0,12):
        t3.append(X_dihed[i][j])
    xnew_sincos.append(t3)

xnew=np.array(xnew)
xnew_sincos=np.array(xnew_sincos)

x_sincos_centered=xnew_sincos
recenterted_weights=[]
for i in range(0,len(xnew_sincos[0])):
    avg=sum(xnew_sincos[:,i])/len(xnew_sincos)
    recenterted_weights.append(avg)
    x_sincos_centered[:,i]=x_sincos_centered[:,i]-avg

recenterted_weights=np.array(recenterted_weights)

import sklearn
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_sincos_centered, sumabs, test_size=0.2, random_state=42)

import tensorflow as tf
from keras.layers import Lambda, Input, Dense, Concatenate
from keras.models import Model
from keras import backend as K
from keras.losses import mse

def circular(args):
    
    z_circular = args
    return z_circular/K.sqrt(K.sum(K.square(z_circular),axis=-1,keepdims=True))


original_dim=12
inputs = Input(shape=(original_dim,),name='encoder_input')
latent_dim = 2
batch_size=100
epochs=500

    
x = Dense(8)(inputs)
x = Dense(4, activation='tanh')(x)
 
z_mean = Dense(latent_dim,activation='tanh',name='z_mean')(x)
z_circular = Lambda(circular, output_shape=(latent_dim,), name='z_circular')(z_mean)

encoder = Model(inputs,  z_circular, name='encoder')
encoder.summary()

latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(4, activation='tanh')(latent_inputs)
x = Dense(8, activation='tanh')(x)
outputs = Dense(original_dim, activation='tanh')(x)    
    
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()


outputs = decoder(encoder(inputs))
vae = Model(inputs, outputs, name='anc-model-round_'+str(version))

reconstruction_loss = mse(inputs, outputs)
reconstruction_loss *= original_dim

vae.add_loss(reconstruction_loss)

vae.compile(optimizer='rmsprop')
vae.summary()

autoencoder_train=vae.fit(x_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, None))
vae.save_weights('anc_explicit_round_'+str(version)+'.h5')

filename = 'anc_trained_model_'+str(version)+'.sav'
pickle.dump(autoencoder_train, open(filename, 'wb'))

pred_test = vae.predict(x_test)
pred_train = vae.predict(x_train)


z_test = encoder.predict(x_test, batch_size=batch_size)
decoder_test=decoder.predict(z_test,batch_size=batch_size)
z_train = encoder.predict(x_train, batch_size=batch_size)
z_final = encoder.predict(x_sincos_centered, batch_size=batch_size)

weights = vae.get_weights()

plumed=open('plumed.dat','w')
#plumed.write( '')

plumed.write("RESTART\n")
plumed.write("\n")

dihed_count=6


plumed.write("phi1: TORSION ATOMS=5,7,9,15\n"+
"psi1: TORSION ATOMS=7,9,15,17\n"+
"phi2: TORSION ATOMS=15,17,19,25\n"+
"psi2: TORSION ATOMS=17,19,25,27\n"+
"phi3: TORSION ATOMS=25,27,29,35\n"+
"psi3: TORSION ATOMS=27,29,35,37\n")


plumed.write("\n")

plumed.write("ALPHABETA ATOMS1={5,7,9,15} REFERENCE=1.25 LABEL=c1\n"+
"ALPHABETA ATOMS1={15,17,19,25} REFERENCE=1.25 LABEL=c3\n"+
"ALPHABETA ATOMS1={25,27,29,3} REFERENCE=1.25 LABEL=c5\n"+
"ALPHABETA ATOMS1={7,9,15,17} REFERENCE=1.25 LABEL=c2\n"+
"ALPHABETA ATOMS1={17,19,25,27} REFERENCE=1.25 LABEL=c4\n"+
"ALPHABETA ATOMS1={27,29,35,37} REFERENCE=1.25 LABEL=c6\n")

plumed.write("\n")

plumed.write("COMBINE LABEL=sum_abs ARG=c1,c2,c3,c4,c5,c6 POWERS=1,1,1,1,1,1 COEFFICIENTS=0.6228,0.1201,0.5643,0.1102,0.5153,0.0403 PERIODIC=NO")

plumed.write("\n")

phi_count=0
psi_count=0
for i in range(0,dihed_count):
    if(i%2==0):
        phi_count+=1
        plumed.write("phi"+str(phi_count)+"_sin: MATHEVAL ARG="+"phi"+str(phi_count)+" VAR=t FUNC=sin(t) PERIODIC=NO"+"\n")
        plumed.write("phi"+str(phi_count)+"_cos: MATHEVAL ARG="+"phi"+str(phi_count)+" VAR=t FUNC=cos(t) PERIODIC=NO"+"\n")    
    if(i%2==1):
        plumed.write("psi"+str(phi_count)+"_sin: MATHEVAL ARG="+"psi"+str(phi_count)+" VAR=t FUNC=sin(t) PERIODIC=NO"+"\n")
        plumed.write("psi"+str(phi_count)+"_cos: MATHEVAL ARG="+"psi"+str(phi_count)+" VAR=t FUNC=cos(t) PERIODIC=NO"+"\n")
        
plumed.write("\n")

phi_count=0
psi_count=0
count=0
for i in range(0,dihed_count):
    if(i%2==0):
        phi_count+=1
        plumed.write("r"+str(count+1)+": MATHEVAL ARG=phi"+str(phi_count)+"_sin VAR=t FUNC=t-("+str(recenterted_weights[count])+") PERIODIC=NO"+"\n")
        plumed.write("r"+str(count+2)+": MATHEVAL ARG=phi"+str(phi_count)+"_cos VAR=t FUNC=t-("+str(recenterted_weights[count+1])+") PERIODIC=NO"+"\n")
        count+=2
    if(i%2==1):
        psi_count+=1
        plumed.write("r"+str(count+1)+": MATHEVAL ARG=psi"+str(psi_count)+"_sin VAR=t FUNC=t-("+str(recenterted_weights[count])+") PERIODIC=NO"+"\n")
        plumed.write("r"+str(count+2)+": MATHEVAL ARG=psi"+str(psi_count)+"_cos VAR=t FUNC=t-("+str(recenterted_weights[count+1])+") PERIODIC=NO"+"\n")
        count+=2

plumed.write("\n")

layer_count=3

plumed.write("ANN ...\n"+
"LABEL=ann\n"+
"ARG=r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12\n"+
"NUM_LAYERS=4\n"+
"NUM_NODES=12,8,4,2\n"+
"ACTIVATIONS=Linear,Tanh,Tanh\n")

plumed.write("\n")

for i in range(0,layer_count):
    weight_temp=""

    for j in range(len(weights[i*2].T)):
        for k in range(len(weights[i*2].T[0])):
            weight_temp+=str(weights[i*2].T[j][k])+","
    weight_temp=weight_temp[:-1]
    plumed.write("WEIGHTS"+str(i)+"="+str(weight_temp)+"\n")
    plumed.write("\n")

for i in range(0,layer_count):
    bias_temp=""

    for j in range(len(weights[i*2+1])):
        bias_temp+=str(weights[i*2+1][j])+","
    bias_temp=bias_temp[:-1]
    plumed.write("BIASES"+str(i)+"="+str(bias_temp)+"\n")
    plumed.write("\n")
plumed.write("... ANN\n")
plumed.write("\n")
plumed.write("METAD ...\n"

 "ARG=ann.node-0,ann.node-1\n"
 "HEIGHT=0.4\n"
 "BIASFACTOR=4\n"
 "TEMP=300.0\n"
 "SIGMA=0.04,0.04\n"
 "GRID_MIN=-4,-4 GRID_MAX=4,4 GRID_BIN=200,200\n"
 "LABEL=WTMetaD\n"
 "FILE=HILLS\n"

 "PACE=1000\n"

 "ACCELERATION\n"

"... METAD\n")
plumed.write("\n")
plumed.write("PRINT STRIDE=100 ARG=phi1,psi1,phi2,psi2,phi3,psi3,sum_abs,ann.node-0,ann.node-1,WTMetaD.bias,WTMetaD.acc FILE=COLVAR\n")
        










