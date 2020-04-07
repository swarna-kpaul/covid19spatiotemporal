import numpy as np
import math
import pandas as pd
import pandasql as ps
def truncate(number, digits) -> float:
	stepper = 10.0 ** digits
	return math.trunc(stepper * number) / stepper

def GetPixelDF(pixlatmin,pixlatmax,pixlonmin,pixlonmax,pixelsize,grid_no):
	fact=100000000
	latmin = np.int(pixlatmin*fact)
	latmax = np.int(pixlatmax*fact)
	longmin = np.int(pixlonmin*fact)
	longmax = np.int(pixlonmax*fact)
	pixelLatRangeStep = np.int((latmax-latmin)/(pixelsize))
	pixelLonRangeStep = np.int((longmax-longmin)/(pixelsize))
	pixlatvals = list(np.round(np.arange(latmin,latmax,pixelLatRangeStep)/fact,5))
	if len(pixlatvals) == pixelsize:
		pixlatvals.append(pixlatmax)
	pixlonvals = list(np.round(np.arange(longmin,longmax,pixelLonRangeStep)/fact,5))
	if len(pixlonvals) == pixelsize:
		pixlonvals.append(pixlonmax)
	ret_df = []#pd.DataFrame()
	pixno = 1
	for i in range(len(pixlatvals)-1):
		minlat = pixlatvals[i]
		maxlat = pixlatvals[i+1]
		for j in range(len(pixlonvals)-1):
			minlong = pixlonvals[j]
			maxlong = pixlonvals[j+1]
			ret_df.append([grid_no,pixno,minlat,maxlat,minlong,maxlong])
			pixno +=1 
	ret_df = pd.DataFrame(ret_df,columns =['grid','pixno','minlat','maxlat','minlong','maxlong'])
	return ret_df

def get_The_Area_Grid(latlongrange,latstep,longstep,margin,pixelsize, counties):
	fact=100000000
	(min_lat,max_lat,min_long,max_long) = latlongrange#(23, 49,-124.5, -66.31)
	min_lat = np.int(min_lat*fact)
	max_lat = np.int(max_lat*fact)
	min_long = np.int(min_long*fact)
	max_long = np.int(max_long*fact)
	range_of_longitude = max_long - min_long
	range_of_latitude = max_lat - min_lat
	block_longitude = np.int(range_of_longitude/(longstep))
	block_latitude = np.int(range_of_latitude/(latstep))
	lattitudes = list(np.round(np.arange(min_lat,max_lat,block_latitude)/fact,5))
	if len(lattitudes) == latstep:
		lattitudes.append(max_lat/fact)
	longitudes = list(np.round(np.arange(min_long,max_long,block_longitude)/fact,5))
	if len(longitudes) == longstep:
		longitudes.append(max_long/fact)
	print(len(lattitudes),len(longitudes))
	#print(longitudes)
	Area_Grid =  {}
	Area_pixel_Grid = pd.DataFrame()
	
	Area_Grid['lattitudes'] = lattitudes
	Area_Grid['longitudes'] = longitudes
	grid_no = 1
	for a in range(len(Area_Grid['lattitudes'])-1):
		pixlatmin = Area_Grid['lattitudes'][a] -(block_latitude*margin)/(fact*pixelsize)
		pixlatmax = Area_Grid['lattitudes'][a+1] + (block_latitude*margin)/(fact*pixelsize)
		for b in range(len(Area_Grid['longitudes'])-1):
			pixlonmin = Area_Grid['longitudes'][b] - (block_longitude*margin)/(fact*pixelsize)
			pixlonmax = Area_Grid['longitudes'][b+1] + (block_longitude*margin)/(fact*pixelsize)
			Area_pixel_Grid = Area_pixel_Grid.append(GetPixelDF(pixlatmin,pixlatmax,pixlonmin,pixlonmax,pixelsize+2*margin,grid_no))
			grid_no +=1
	Area_pixel_Grid = ps.sqldf("""select a.*, sum(ifnull(b.pop,0)) pop from Area_pixel_Grid a left outer join counties b
		on b.lat between a.minlat and a.maxlat and b.long between a.minlong and a.maxlong 
		group by a.grid,a.pixno,a.minlong,a.maxlong,a.minlat,a.maxlat""", locals())
	return Area_pixel_Grid

def frames_df(df_pop_pat,Area_df):
	
	days = ps.sqldf("select distinct data_date from df_pop_pat order by 1",locals())
	days['day'] = np.arange(len(days))
	Area_df['key'] = 1
	days['key'] = 1
	Area_day_df =Area_df.merge(days, on='key')
	frames_grid = pd.DataFrame()
	for grid in set(Area_df['grid']):
		Area_day_df_grid = Area_day_df[Area_day_df['grid']==grid]
		df_pop_pat_grid = df_pop_pat[(df_pop_pat['lat'] >= np.min(Area_day_df_grid['minlat'])) & (df_pop_pat['lat'] < np.max(Area_day_df_grid['maxlat']))
									 & (df_pop_pat['long'] >= np.min(Area_day_df_grid['minlong'])) & (df_pop_pat['long'] < np.max(Area_day_df_grid['maxlong']))]
		if len(df_pop_pat_grid) == 0:
			continue
		frames = ps.sqldf("""select a.grid,a.day,a.pixno,a.data_date,sum(ifnull(b.no_pat,0)) no_pat,sum(ifnull(a.pop,0))*0.6 pop, 
					sum(ifnull(a.pop,0))*0.6 - sum(ifnull(b.no_pat,0)) sus_pop, 
					max(ifnull(b.lat,0)) lat, min(ifnull(b.long,0)) long
					from Area_day_df_grid a left outer join df_pop_pat_grid b on a.data_date = b.data_date and
					b.lat between a.minlat and a.maxlat and b.long between a.minlong and a.maxlong
					group by a.grid,a.day,a.pixno""",locals())
		#frames['no_pat_sum'] = frames.groupby(['grid','pixno'])['no_pat'].apply(lambda x: x.cumsum())
		frames['pop'] = frames.groupby(['grid','pixno'])['pop'].transform('max')
		frames['pixel'] = np.array((np.log(frames[['no_pat']].values.astype(float)+1)/np.log(frames[['pop']].values.astype(float)+2)))
		if np.sum(df_pop_pat_grid['no_pat']) != np.sum(frames['no_pat']):
			print("failure", grid)
			break
		frames_grid = frames_grid.append(frames)
	return frames_grid


def validate_frames(frames_grid,df_pop_pat,margin):
	days = np.max(frames_grid['day'])+1
	pixno = np.int(np.max(frames_grid['pixno']))
	pix = np.int(math.sqrt(pixno))
	print(pix)
	print(np.sum(frames_grid['no_pat']),np.sum(df_pop_pat['no_pat']),len(frames_grid),
		 len(set(frames_grid['grid'])),len(frames_grid)/(len(set(frames_grid['grid']))*days) )
	a = np.reshape(range(1,pixno+1),(pix,pix))
	a=np.flip(a,0)
	start = np.int(margin)
	end = np.int(pix-margin)
	a=a[start:end,start:end].flatten()
	print(np.sum(frames_grid[frames_grid['pixno'].isin(a)]['no_pat']))

#Area_df = get_The_Area_Grid((23, 49,-124.5, -66.31),24,40)
import matplotlib.pyplot as plt
def write_images(frames_grid,tgtdir,pixelsize):
	days = set(frames_grid['day'])
	for grid in set(frames_grid['grid']):
		for day in days:
			frame = frames_grid[(frames_grid['grid']==grid) & (frames_grid['day']==day)].sort_values(['pixno'])
			fp = np.array(frame['pixel']).reshape(2*pixelsize,2*pixelsize)
			fp = np.flip(fp,0)
			if sum(sum(fp)) == 0:
				continue
			plt.imsave(tgtdir+'img-'+str(grid)+'-'+str(day)+'.png', 1-fp,cmap='gray')
			
def prep_pop_image(Area_df):
	pix=np.int(np.sqrt(max(Area_df['pixno'])))
	maxpop = max(Area_df['pop'])
	popframes = []
	for i in sorted(set(Area_df['grid'])):
		frame = (Area_df[Area_df['grid']==i].sort_values(['pixno']))
		frame = np.log(np.array(frame['pop'])+1)/np.log(maxpop)
		frame = np.flip(frame.reshape(pix,pix),0)
		popframes.append(frame)
	popframes = np.array(popframes)
	return popframes
			
from scipy import ndimage
def prep_image(frames_grid,minframe,testspan=4,channel = 1, extframes = []):
	days = np.max(frames_grid['day'])
	pixno = np.int(np.max(frames_grid['pixno']))
	pix = np.int(math.sqrt(pixno))
	train = []
	output = []
	test =[]
	testoutput = []
	test_gridday = dict()
	testseq = 0
	maxpop = max(frames_grid['pop'])
	for grid in sorted(set(frames_grid['grid'])):
		train_samp = []
		output_samp = []
		for day in range(days,0,-1):
			frames = frames_grid[(frames_grid['grid']==grid) & (frames_grid['day']==day-1)].sort_values(['pixno'])
			frame = np.array(frames['pixel']).reshape(pix,pix)
			#popframes = np.log((np.array(frames['sus_pop'])+1)*(np.array(frames['no_pat'])+1))/np.log((maxpop**2)/4)
			popframes = np.log((np.array(frames['pop'])+1))/np.log(maxpop)
			popframes = popframes.reshape(pix,pix)
			if sum(sum(frame)) == 0:
				continue
			frame = np.flip(frame,0)
			popframes = np.flip(popframes,0)
			frame[frame<0] = 0			
			#frame += 0.1
			if np.max(frame) > 1:
				frame /= np.max(frame)
			frame = frame[::,::,np.newaxis]
			if channel > 1:
				frame = np.concatenate((frame,popframes[::,::,np.newaxis]),axis = 2)
			train_samp.append(frame)
			frame = frames_grid[(frames_grid['grid']==grid) & (frames_grid['day']==day)].sort_values(['pixno'])
			frame = np.array(frame['pixel']).reshape(pix,pix)
			frame = np.flip(frame,0)
			frame[frame<0] = 0	
			#frame += 0.1
			frame = frame[::,::,np.newaxis]
			output_samp.append(frame)
		train_samp = np.array(train_samp)
		output_samp = np.array(output_samp)
		if train_samp.shape[0]-testspan < minframe:
			continue  
		#testspan= 4 #np.int(minframe/2)
		test.append(np.flip(train_samp[testspan-1:minframe+testspan-1,::,::,::],0))
		testoutput.append(np.flip(output_samp[:testspan,::,::,::],0))
		test_gridday[testseq] = (grid,testspan)
		testseq += 1
		######################create training records
		for i in range(testspan-1,train_samp.shape[0]-minframe):
			train.append(np.flip(train_samp[i:i+minframe,::,::,::],0))
			output.append(np.flip(output_samp[i:i+minframe,::,::,::],0))
		#for i in range(np.int((train_samp.shape[0]-testspan+1)/minframe)):
		#	train.append(np.flip(train_samp[i*minframe:(i+1)*minframe,::,::,::],0))
		#	output.append(np.flip(output_samp[i*minframe:(i+1)*minframe,::,::,::],0))
	test = np.array(test)
	testoutput = np.array(testoutput) 

	train = np.array(train)
	output = np.array(output) 
	return(train,output,test,testoutput,test_gridday)