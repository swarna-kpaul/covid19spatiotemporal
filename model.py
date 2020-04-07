from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import ConvLSTM2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import losses
import numpy as np
import pylab as plt
from PIL import Image
import pandas as pd
from matplotlib import image
from sklearn import preprocessing

from tensorflow import keras
def create_model(pixel,filters,channel,hiddenlayers = 5):
	seq = Sequential()
	seq.add(ConvLSTM2D(filters=filters, kernel_size=(3, 3),
				   input_shape=(None, pixel, pixel, channel),
				   padding='same', return_sequences=True))
	for layer in range(hiddenlayers):
		seq.add(ConvLSTM2D(filters=filters, kernel_size=(3, 3),
				   padding='same', return_sequences=True))#, kernel_regularizer=keras.regularizers.l2(l=0.1)))
	#seq.add(BatchNormalization())

	seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
			   activation='elu',
			   padding='same', data_format='channels_last'))#,kernel_regularizer=keras.regularizers.l2(l=0.1)))
	seq.compile(loss='mean_squared_error', optimizer='adam',metrics=['mae']) #losses.KLDivergence()
	return seq
	
import pandasql as ps
import pickle
def convert_image_to_data(image,frames_grid,grid,margin):
	#image = image
	image[image<0.01] = 0
	frame = np.flip(image,0)
	#print(frame) 
	pix = frame.shape[0]
	#margin = np.int(pix/4)
	pop = frames_grid[frames_grid['grid']==grid]
	pop = ps.sqldf("select pop from (select pixno,max(pop) pop from pop group by pixno order by 1)",locals())
	pop = np.array(pop).reshape(pix,pix)
	pop = np.log(pop[margin:pix-margin,margin:pix-margin]+2)
	frame = frame[margin:pix-margin,margin:pix-margin]
	frame = np.multiply(frame,pop)
	frame = np.exp(frame) -1
	frame = np.round(frame,0)
	return frame
def validate(model,test,testout,test_gridday,frames_grid,margin):
	errorsum = 0
	cnt = 1
	griderror={}
	for k,(grid,span) in test_gridday.items():
		######## for each test grid
		track = test[k]#,::,::,::,::]
		totpop = track[0,::,::,1:] #+ track[0,::,::,0]
		popexists = pickle.loads(pickle.dumps(totpop[::,::,0],-1))
		popexists[popexists>0] = 1
		out = testout[k]
		######## for each prediction day
		for i in range(span):
			new_pos = model.predict(track[np.newaxis, ::, ::, ::, ::])
			new = new_pos[::, -1, ::, ::, ::]
			new = np.multiply(new[0,::,::,0],popexists)[np.newaxis,::,::,np.newaxis]
			#sus_pop = np.multiply((totpop - new[0,::,::,0]),popexists)
			newtrack = np.concatenate((new,totpop[np.newaxis,::,::,::]),axis = 3)
			track = np.concatenate((track, newtrack), axis=0)
			predictframe = np.round(convert_image_to_data(np.squeeze(new,0)[::,::,0],frames_grid,grid,margin),0)
			actualframe = convert_image_to_data(out[i,::,::,0],frames_grid,grid,margin)
			notzeroframe = pickle.loads(pickle.dumps(actualframe, -1))
			notzeroframe[notzeroframe == 0] =1	 
			error = np.sqrt(sum(sum(np.square((predictframe - actualframe)/notzeroframe))))/((actualframe.shape[0])**2)
			#print(error,k,grid,i)	 
			errorsum +=error
			cnt +=1
		griderror[grid] = {'actual': sum(sum(actualframe)), 'predicted': sum(sum(predictframe)),'error': np.absolute(sum(sum(predictframe)) - sum(sum(actualframe)))}
	averageerror = errorsum/cnt
	return (averageerror,griderror)