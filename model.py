from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import ConvLSTM2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import losses
import numpy as np
import pandas as pd
import random
import pandasql as ps
import pickle
from scipy.stats import entropy
########## Create ConvLSTM network ##############
 
def create_model(pixel,filters,channel,hiddenlayers = 4):
	seq = Sequential()
	seq.add(ConvLSTM2D(filters=filters, kernel_size=(3, 3),
				   input_shape=(None, pixel, pixel, channel),
				   padding='same', return_sequences=True, activation = 'elu'))
	for layer in range(hiddenlayers):
		seq.add(ConvLSTM2D(filters=filters, kernel_size=(3, 3),
				   padding='same', return_sequences=True))

	seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
			   activation='elu',
			   padding='same', data_format='channels_last'))
	seq.compile(loss='mean_squared_error', optimizer='adam',metrics=['mae'])
	return seq

############## Save ensemble model to disk ###################
def save_ensemble(ensemble,tgt_dir,name='default'):
	_ensemble_models = ensemble[0]
	ensemble_weights = ensemble[1]
	ensemble_model_def = []
	for i,model in enumerate(_ensemble_models):
		ensemble_model_def.append(model.to_json())
		model.save_weights(tgt_dir+name+'-model-'+str(i)+'.h5')
	with open(tgt_dir+name+'-modeldef.pkl','wb') as file:
		pickle.dump(ensemble_model_def,file)
	with open(tgt_dir+name+'-ensembleweights.pkl','wb') as file:
		pickle.dump(ensemble_weights,file)
	return

################ Load ensemble from disk ##########################
def load_ensemble(name,tgt_dir):
	####### load model_def
	ensemble_models = []
	with open(tgt_dir+name+'-modeldef.pkl','rb') as file:
		ensemble_model_def = pickle.load(file)
	#### load model weightages
	with open(tgt_dir+name+'-ensembleweights.pkl','rb') as file:
		ensemble_weights = pickle.load(file)
	
	for i,model in enumerate(ensemble_model_def):
		loaded_model = model_from_json(model)
		loaded_model.load_weights(tgt_dir+name+'-model-'+str(i)+'.h5')
		loaded_model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mae'])
		ensemble_models.append(loaded_model)
	return((ensemble_models,ensemble_weights))
		

# """Compute softmax values for each sets of scores in x."""
def softmax(x):
	if np.max(x) > 1:
		e_x = np.exp(x/np.max(x))
	else:
		e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum()


########## Train ensemble model ##########################
def train_ensemble(train,output, hiddenlayers = 3,epochs = 20,ensembles=5,gamma = 0.6, channel = 2 , pixel = 16, filters = 32):
	ensemble_models = []
	ensemble_weights = []
	ensemble_size = np.int(train.shape[0]*gamma)
	for i in range(ensembles):
		ensemble_train_idx = random.sample(range(train.shape[0]),ensemble_size)
		ensemble_train_X = train[ensemble_train_idx,::,::,::,::]
		ensemble_train_Y = output[ensemble_train_idx,::,::,::,::]
		ensemble_test_idx = list(set(range(train.shape[0])) - set(ensemble_train_idx))
		ensemble_test_X = train[ensemble_test_idx,::,::,::,::]
		ensemble_test_Y = output[ensemble_test_idx,::,::,::,::]
		model = create_model(pixel=pixel,filters=filters,channel=channel,hiddenlayers=hiddenlayers) 
		hist = model.fit(ensemble_train_X, ensemble_train_Y, batch_size=10, epochs=epochs, validation_data=(ensemble_test_X,ensemble_test_Y))
		ensemble_test_sum = np.sum(ensemble_train_Y)
		model_weight = np.log((1/hist.history['val_mean_absolute_error'][-1])*ensemble_test_sum+1)
		ensemble_models.append(model)
		ensemble_weights.append(model_weight)
	ensemble_weights = softmax(np.array(ensemble_weights))
	print(ensemble_weights)
	return ((ensemble_models,ensemble_weights))

################# Train ensemble with USA data ###########################
def train_usa_ensemble(indata,epochs=30):
	group_ensembles = []
	for (train,output,test,testoutput,test_gridday,frames_grid) in indata:
		ensemble_us = train_ensemble(train,output,hiddenlayers=2,channel =2,epochs = epochs, gamma = 0.6, ensembles = 5)
		group_ensembles.append(ensemble_us)
	return group_ensembles

############ Predict next from by passing a sequence of input frames to the ensemble #####
def ensemble_predict(ensemble,inp_data):
	ensemble_models = ensemble[0]
	ensemble_weights = ensemble[1]
	prediction = np.zeros(inp_data.shape)
	for i,model in enumerate(ensemble_models):
		prediction += model.predict(inp_data) * ensemble_weights[i]
	return prediction

########### Merge multiple ensembles to 1 #########################
def merge_ensemble(ensemble_list):
	ensemble_models = []
	ensemble_weights = []
	for ensemble in ensemble_list:
		ensemble_models += ensemble[0]
		ensemble_weights += list(ensemble[1])
	ensemble_weights = softmax(np.array(ensemble_weights))
	return ((ensemble_models,ensemble_weights))
	
########## Convert image pixel values to number of infection cases ########
def convert_image_to_data(image,margin,sus_pop):  
	frame = image
	frame[frame<0.001] = 0
	pix = frame.shape[0]
	frame = frame[margin:pix-margin,margin:pix-margin]
	_sus_pop = np.log(sus_pop +2)
	frame = np.multiply(frame,_sus_pop)
	popexists_size = len(sus_pop[sus_pop>0])
	frame = np.exp(frame) -1
	frame = np.round(frame,0)
	return (frame,popexists_size)
	
def forecast(ensemble,input_sequence,frames_grid,span):	
	pix = np.int(np.sqrt(max(frames_grid['pixno'])))
	gridpix = np.flip(np.array(range(1,max(frames_grid['pixno'])+1)).reshape(pix,pix),0)
	gridpix = gridpix[margin:pix-margin,margin:pix-margin]
	forecastframe = pd.DataFrame()
	for k,(grid,_filler) in test_gridday.items():
		track = input_sequence[k]
		totpop = track[0,::,::,1:] 
		pix = totpop.shape[0]
		pred_sus_pop = frames_grid[(frames_grid['grid'] ==grid) & (frames_grid['day'] <=(max(frames_grid['day'])-span))]
		pred_sus_pop = pred_sus_pop.groupby(['pixno'])[['no_pat','pop']].max()
		pred_sus_pop = 	np.array(pred_sus_pop['pop']) -np.array(pred_sus_pop['no_pat'])
		pred_sus_pop = np.flip(pred_sus_pop.reshape(pix,pix),0)[margin:pix-margin,margin:pix-margin]
		popexists = pickle.loads(pickle.dumps(totpop[::,::,0],-1))
		popexists[popexists>0] = 1
		######## for each prediction day
		for i in range(span):
			new_pos = ensemble_predict(ensemble,track[np.newaxis, ::, ::, ::, ::])
			new = new_pos[::, -1, ::, ::, ::]
			new = np.multiply(new[0,::,::,0],popexists)[np.newaxis,::,::,np.newaxis]
			newtrack = np.concatenate((new,totpop[np.newaxis,::,::,::]),axis = 3)
			track = np.concatenate((track, newtrack), axis=0)
			predictframe,popexists_size = convert_image_to_data(np.squeeze(new,0)[::,::,0],margin,pred_sus_pop)
			predictframe = np.round(predictframe,0)
			pred_sus_pop = pred_sus_pop - predictframe
			pred_sus_pop[pred_sus_pop<0] = 0
			_forecastframe = pd.DataFrame({'pixno':gridpix[pred_sus_pop>0].flatten(), 'predict':predictframe[pred_sus_pop>0].flatten()}) 
			_forecastframe['day'] = i
			_forecastframe['grid'] = grid   
			forecastframe = forecastframe.append(_forecastframe)   		
	return forecastframe
		

################## Test an ensemble model ###########################
def validate(ensemble,test,testout,test_gridday,frames_grid,margin):
	errorsum = 0
	cnt = 1
	predicttotal = pd.DataFrame()
	pix = np.int(np.sqrt(max(frames_grid['pixno'])))
	gridpix = np.flip(np.array(range(1,max(frames_grid['pixno'])+1)).reshape(pix,pix),0)
	gridpix = gridpix[margin:pix-margin,margin:pix-margin]
	errorframe = pd.DataFrame()  
	for k,(grid,span) in test_gridday.items():
		######## for each test grid
		track = test[k]
		totpop = track[0,::,::,1:] 
		pix = totpop.shape[0]
		print(grid)
		pred_sus_pop = frames_grid[(frames_grid['grid'] ==grid) & (frames_grid['day'] <=(max(frames_grid['day'])-span))]
		pred_sus_pop = pred_sus_pop.groupby(['pixno'])[['no_pat','pop']].max()
		pred_sus_pop = 	np.array(pred_sus_pop['pop']) -np.array(pred_sus_pop['no_pat'])
		pred_sus_pop = np.flip(pred_sus_pop.reshape(pix,pix),0)[margin:pix-margin,margin:pix-margin]
		act_sus_pop = frames_grid[(frames_grid['grid'] ==grid) & (frames_grid['day'] > (max(frames_grid['day'])-span))][['day','no_pat','pop']]
		act_sus_pop['day'] = act_sus_pop['day'] - min(act_sus_pop['day']) 
		
		popexists = pickle.loads(pickle.dumps(totpop[::,::,0],-1))
		popexists[popexists>0] = 1
		out = testout[k]
		######## for each prediction day
		for i in range(span):
			#new_pos = ensemble.predict(track[np.newaxis, ::, ::, ::, ::])
			new_pos = ensemble_predict(ensemble,track[np.newaxis, ::, ::, ::, ::])
			new = new_pos[::, -1, ::, ::, ::]
			new = np.multiply(new[0,::,::,0],popexists)[np.newaxis,::,::,np.newaxis]
			newtrack = np.concatenate((new,totpop[np.newaxis,::,::,::]),axis = 3)
			track = np.concatenate((track, newtrack), axis=0)
			predictframe,popexists_size = convert_image_to_data(np.squeeze(new,0)[::,::,0],margin,pred_sus_pop)
			predictframe = np.round(predictframe,0)
			pred_sus_pop = pred_sus_pop - predictframe
			pred_sus_pop[pred_sus_pop<0] = 0
			_act_sus_pop = act_sus_pop[act_sus_pop['day'] == i]
			_act_sus_pop = np.array(_act_sus_pop['pop']) - np.array(_act_sus_pop['no_pat'])
			_act_sus_pop = np.flip(_act_sus_pop.reshape(pix,pix),0)[margin:pix-margin,margin:pix-margin]
			actualframe,popexists_size = convert_image_to_data(out[i,::,::,0],margin,_act_sus_pop)
			notzeroframe = pickle.loads(pickle.dumps(actualframe, -1))
			notzeroframe[notzeroframe == 0] =1	 
			_errorframe = pd.DataFrame({'pixno':gridpix[_act_sus_pop>0].flatten(), 'predict':predictframe[_act_sus_pop>0].flatten(), 'actual':actualframe[_act_sus_pop>0].flatten()}) 
			_errorframe['day'] = i
			_errorframe['grid'] = grid   
			errorframe = errorframe.append(_errorframe)   			
			error = np.sum(np.absolute((predictframe - actualframe)/notzeroframe))/(popexists_size+1)
			errorsum +=error
			cnt +=1
			predicttotal = predicttotal.append(pd.DataFrame([[grid,i,np.sum(predictframe),np.sum(actualframe)]],columns=['grid','day', 'predict','actual']))
		####### predict for another span
		for i in range(span):
			#new_pos = ensemble.predict(track[np.newaxis, ::, ::, ::, ::])
			new_pos = ensemble_predict(ensemble,track[np.newaxis, ::, ::, ::, ::])
			new = new_pos[::, -1, ::, ::, ::]
			new = np.multiply(new[0,::,::,0],popexists)[np.newaxis,::,::,np.newaxis]
			newtrack = np.concatenate((new,totpop[np.newaxis,::,::,::]),axis = 3)
			track = np.concatenate((track, newtrack), axis=0)
			predictframe,popexists_size = convert_image_to_data(np.squeeze(new,0)[::,::,0],margin,pred_sus_pop)
			pred_sus_pop = pred_sus_pop - predictframe
			pred_sus_pop[pred_sus_pop<0] = 0
			predictframe = np.round(predictframe,0)
			predicttotal = predicttotal.append(pd.DataFrame([[grid,i+span,np.sum(predictframe),0]],columns=['grid','day', 'predict','actual']))

	griderror = predicttotal[predicttotal['day']<span]
	griderror = np.sum(np.absolute(np.array(griderror['predict']) - np.array(griderror['actual']))/(np.array(griderror['actual'])+1))/len(griderror)
	errortotal = predicttotal[predicttotal['day']<span]
	errortotal = ps.sqldf("select day, sum(predict) predict, sum(actual) actual from errortotal group by day", locals())
	averagetotalerror = np.sum(np.absolute(np.array(errortotal['predict']) - np.array(errortotal['actual']))/(np.array(errortotal['actual'])+1))/span
	averageerror = errorsum/cnt
	return (averageerror,predicttotal,averagetotalerror,errorframe)
	
############ Test ensemble model foor Italy ####################
def test_ensemble(ensemble,test,testoutput,test_gridday,frames_grid,span=5,margin=4):
	test_gridday_span = {}
	for i,v in test_gridday.items():
		test_gridday_span[i] = (v[0],span)
	averageerror,predicttotal,averagetotalerror,errorframe = validate(ensemble,test,testoutput,test_gridday_span,frames_grid,margin)
	_errorframe=errorframe.groupby(['grid','pixno']).sum().reset_index()
	_errorframe=pd.merge(_errorframe,frames_grid[frames_grid['day'] == max(frames_grid['day']) -testoutput.shape[1]][['grid','pixno','no_pat']],on = ['grid','pixno'])
	_errorframe['actual'] =  _errorframe['actual']+_errorframe['no_pat']
	_errorframe['predict'] =  _errorframe['predict']+_errorframe['no_pat']
	_errorframe['actual_denom'] = _errorframe['actual']
	_errorframe[_errorframe['actual_denom']<1]['actual_denom'] = 1
	KL_div = entropy( softmax(_errorframe['predict']), softmax(_errorframe['actual']) )
	MAPE = (np.sum(np.absolute((_errorframe['actual']-_errorframe['predict'])/np.array(_errorframe['actual_denom'])))/len(_errorframe))
	cumulative_predicttotal_day,MAPE_countrytotal = predict_countrytotal(frames_grid,span,predicttotal,margin)
	return(KL_div,MAPE,_errorframe,MAPE_countrytotal,cumulative_predicttotal_day,predicttotal)

################# Calculate total predicted infection cases across country ##############
def predict_countrytotal(frames_grid,span,predicttotal,margin):
	pix = np.int(np.sqrt(max(frames_grid['pixno'])))
	gridpix = np.flip(np.array(range(1,max(frames_grid['pixno'])+1)).reshape(pix,pix),0)
	gridpix = gridpix[margin:pix-margin,margin:pix-margin].flatten()
	total_cases_W = sum(frames_grid[(frames_grid.pixno.isin(gridpix)) & (frames_grid['day'] <= max(frames_grid['day'])-span)]['new_pat'])
	predicttotal_day = predicttotal.groupby(['day']).sum().reset_index()
	predicttotal_day[['predict','actual']] = predicttotal_day[['predict','actual']].cumsum()
	predicttotal_day['predict'] += total_cases_W
	predicttotal_day['actual'] += total_cases_W
	_predicttotal_day = predicttotal_day[predicttotal_day['day']<span]
	averagetotalerror_countrytotal  = sum(np.absolute(np.array(_predicttotal_day['predict'] - _predicttotal_day['actual']))/np.array(_predicttotal_day['actual']))/span
	return (predicttotal_day,averagetotalerror_countrytotal)
 
############ Test ensemble model of USA #####################
def test_usa_ensemble(group_ensembles,indata,span,margin=4):
	predicttotal_country =pd.DataFrame()
	averageerror_country =0
	country_errorframe = pd.DataFrame()
	frames_grid_country = pd.DataFrame()
	for group, (train,output,test,testoutput,test_gridday,frames_grid) in enumerate(indata):
		test_gridday_span = {}
		for i,v in test_gridday.items():
			test_gridday_span[i] = (v[0],span)
		averageerror,predicttotal,averagetotalerror,errorframe = validate(group_ensembles[group],test,testoutput,test_gridday_span,frames_grid,margin)
		predicttotal_country = predicttotal_country.append(predicttotal)
		country_errorframe = country_errorframe.append(errorframe)
		frames_grid_country = frames_grid_country.append(frames_grid)
	_errorframe=country_errorframe.groupby(['grid','pixno']).sum().reset_index()
	_errorframe=pd.merge(_errorframe,frames_grid_country[frames_grid_country['day'] == max(frames_grid_country['day']) -testoutput.shape[1]][['grid','pixno','no_pat']],on = ['grid','pixno'])
	_errorframe['actual'] =  _errorframe['actual']+_errorframe['no_pat']
	_errorframe['predict'] =  _errorframe['predict']+_errorframe['no_pat']
	_errorframe['actual_denom'] = _errorframe['actual']
	_errorframe.loc[_errorframe['actual_denom']<1,['actual_denom']] = 1
	KL_div = entropy( softmax(_errorframe['predict']), softmax(_errorframe['actual']))
	MAPE = (np.sum(np.absolute((_errorframe['actual']-_errorframe['predict'])/np.array(_errorframe['actual_denom'])))/len(_errorframe))
	cumulative_predicttotal_day,MAPE_countrytotal = predict_countrytotal(frames_grid_country,span,predicttotal_country,margin)
	return (KL_div,MAPE,_errorframe,MAPE_countrytotal,cumulative_predicttotal_day,predicttotal_country)

def train_country_ensemble(src_dir,country,epochs = 1,hiddenlayers=2,ensembles=5,gamma = 0.6, channel = 2 , pixel = 16, filters = 32):
	with open(src_dir+country+'prepdata.pkl', 'rb') as filehandler:
		indata = pickle.load(filehandler)
	if country = 'USA':
		ensemble = train_usa_ensemble(indata)
		for group,ensemble_us in enumerate(ensemble):
			save_ensemble(ensemble,src_dir,name='USA_group_'+str(group))
	else:
		(train,output,test,testoutput,test_gridday,frames_grid) = indata
		ensemble = train_ensemble(train,output, hiddenlayers = hiddenlayers,epochs = epochs,ensembles=ensembles,gamma=gamma,channel=channel,pixel=pixel,filters=filters)
		save_ensemble(ensemble,src_dir,name=country)
		
def test_country_ensemble(src_dir,country,span,margin=4):
	with open(src_dir+country+'prepdata.pkl', 'rb') as filehandler:
		indata = pickle.load(filehandler)
	if country = 'USA':
		(train,output,test,testoutput,test_gridday,frames_grid) = indata[0]
		if span > test_gridday[0][1]:
			print("span should be less than ",test_gridday[0][1]+1)
			raise
		ensemble = []
		for i in range(len(indata)):
			ensemble.append(load_ensemble('USA_group_'+str(i),src_dir))
		KL_div,MAPE,_errorframe,MAPE_countrytotal,cumulative_predicttotal_day,predicttotal_country = test_usa_ensemble(ensemble,indata,span,margin)
	else:
		(train,output,test,testoutput,test_gridday,frames_grid) = indata
		ensemble = load_ensemble(country,src_dir)
		if span > test_gridday[0][1]:
			print("span should be less than ",test_gridday[0][1]+1)
			raise
		KL_div,MAPE,_errorframe,MAPE_countrytotal,cumulative_predicttotal_day,predicttotal_country = test_ensemble(ensemble,test,testoutput,test_gridday,frames_grid,span=span,margin)
	return (KL_div,MAPE,_errorframe,MAPE_countrytotal,cumulative_predicttotal_day,predicttotal_country)

def forecast_country_cases(src_dir,country,span=5):
	with open(src_dir+country+'prepdata.pkl', 'rb') as filehandler:
		indata = pickle.load(filehandler)
	forecast_frame = pd.DataFrame()
	if country == 'USA':
		for group, (train,output,test,testoutput,test_gridday,frames_grid) in enumerate(indata):
			ensemble = load_ensemble('USA_group_'+str(group),src_dir)
			forecast_frame.append(forecast(ensemble,test,frames_grid,span))
	else:
		(train,output,test,testoutput,test_gridday,frames_grid) = indata
		ensemble = load_ensemble(country,src_dir)
		forecast_frame = forecast(ensemble,test,frames_grid,span)
	return forecast_frame
