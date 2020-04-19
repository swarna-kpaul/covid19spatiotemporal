import numpy as np
import math
import pandas as pd
import pandasql as ps
import matplotlib.pyplot as plt
import pickle
## Function to divide the GRID Area into Pixels
## Parameter Needed - 1. pixlatmax - float - Maximum Value of Lattitude( GRID Boundary) 2. pixlatmin - float - Minimum value of the lattitudes( GRID Boundary)
##						3. pixlonmax - float - Maximum value of Longitude( GRID Boundary) 4. pixlonmin - float - Minimum value of longitude( GRID Boundary)
##						5. pixelsize - Number - Size of Earch Pixel in GRID(Number of Pixel in Grid)	6. Grid No - Number - The Id of Grid

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
	ret_df = []
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

## Function to divide the whole country into GRIDS and Pixels
## Parameter Needed - 1. latlongrange - Tuple -	Coordinate boundary of the country(south, north, west, east) 2. latstep - number -Number of division under lattitude range
##						3. longstep - Number - Number of division under longitude range	4. margin - Number - Overlapping adjustment for pixel boundaries
##						5. pixelsize - Number - Pixelsize of each subpixel	6. counties - Dataframe - The county Dataframe containing the lattitude longitude and population data

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
	Area_Grid =	{}
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

## Function to validate the frames based on the time series patient data
## Parameter Needed - 1. frames_grid - Dataframe - Will contain the dataframe population data for each and every frame 2. df_pop_pat - Dataframe - Population and patient data of country
## 					 3. margin - number - Overlapping adjustment for pixels
def validate_frames(frames_grid,df_pop_pat,margin):
	days = np.max(frames_grid['day'])+1
	pixno = np.int(np.max(frames_grid['pixno']))
	pix = np.int(math.sqrt(pixno))
	print(pix)
	print(np.sum(frames_grid['new_pat']),np.sum(df_pop_pat['new_pat']),len(frames_grid),
		 len(set(frames_grid['grid'])),len(frames_grid)/(len(set(frames_grid['grid']))*days) )
	a = np.reshape(range(1,pixno+1),(pix,pix))
	a=np.flip(a,0)
	start = np.int(margin)
	end = np.int(pix-margin)
	a=a[start:end,start:end].flatten()
	print(np.sum(frames_grid[frames_grid['pixno'].isin(a)]['new_pat']))

## Creates the Frame DF from Area DF and Patient Data
## Parameter Needed - 1. df_pop_pat - Dataframe - country specific pixel level patient and population data for country 2. Area_df - DataFrame-	Pixel level coridnate data with population for each pixel
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
		frames = ps.sqldf("""select a.grid,a.day,a.pixno,a.data_date,sum(ifnull(b.no_pat,0)) no_pat,sum(ifnull(a.pop,0))*0.6 pop, sum(ifnull(b.new_pat,0)) new_pat,
					sum(ifnull(a.pop,0))*0.6 - sum(ifnull(b.no_pat,0)) sus_pop, 
					max(ifnull(b.lat,0)) lat, min(ifnull(b.long,0)) long
					from Area_day_df_grid a left outer join df_pop_pat_grid b on a.data_date = b.data_date and
					b.lat between a.minlat and a.maxlat and b.long between a.minlong and a.maxlong
					group by a.grid,a.day,a.pixno""",locals())
		frames['pop'] = frames.groupby(['grid','pixno'])['pop'].transform('max')
		maxpop = max(frames['pop'])	
		frames['pixel'] = np.array((np.log(frames[['new_pat']].values.astype(float)+1)/np.log(frames[['sus_pop']].values.astype(float)+2)))
		if np.sum(df_pop_pat_grid['no_pat']) != np.sum(frames['no_pat']):
			print("failure", grid)
			break
		frames_grid = frames_grid.append(frames)
	return frames_grid

## Prepares the Training Images for the Nural network injestion after Test Train Validation if the results meets proper Threshold
## Parameter Needed - 1. frames_grid - Output of frames_df function 2. minframe - Minimum no of frames required 3. channel - no of feature variable (population,patients) 4. extframes - Array External parameters like (no of testing,VMT if needed, defult - None)
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
			popframes = np.log((np.array(frames['pop'])+1))/np.log(maxpop)
			popframes = popframes.reshape(pix,pix)
			#if sum(sum(frame)) == 0:
			#	continue
			frame = np.flip(frame,0)
			popframes = np.flip(popframes,0)
			frame[frame<0] = 0			
			if np.max(frame) > 1:
				frame /= np.max(frame)
			frame = frame[::,::,np.newaxis]
			if channel > 1:
				frame = np.concatenate((frame,popframes[::,::,np.newaxis]),axis = 2)
			################# add any external frames
			for newcol in extframes:
				newframe = np.array(frames[newcol])
				newframe = newframe.reshape(pix,pix)
				frame = np.concatenate((frame,newframe[::,::,np.newaxis]),axis = 2)
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
		if train_samp.shape[0]< minframe:
			continue	
		test.append(np.flip(train_samp[testspan:minframe+testspan,::,::,::],0))
		testoutput.append(np.flip(output_samp[:testspan,::,::,::],0))
		test_gridday[testseq] = (grid,testspan)
		testseq += 1
		######################create training records
		for i in range(testspan,train_samp.shape[0]-minframe):
			train.append(np.flip(train_samp[i:i+minframe,::,::,::],0))
			output.append(np.flip(output_samp[i:i+minframe,::,::,::],0))
	test = np.array(test)
	testoutput = np.array(testoutput) 

	train = np.array(train)
	output = np.array(output) 
	return(train,output,test,testoutput,test_gridday)

## Prepare Frames Grid Specially for USA
def prep_us_data(M,N,frames_grid,minframe = 10,channel = 2, testspan = 8):
	gridframe = np.flip(np.array(range(1,M*N+1)).reshape(M,N),0)
	g1 = gridframe[0:np.int(M/2),0:np.int(N/2)].flatten()
	g2 = gridframe[np.int(M/2):M,0:np.int(N/2)].flatten()
	g3 = gridframe[0:np.int(M/2),np.int(N/2):N].flatten()
	g4 = gridframe[np.int(M/2):M,np.int(N/2):N].flatten()
	frames_grid_group = [frames_grid[frames_grid.grid.isin(g1)],frames_grid[frames_grid.grid.isin(g2)],frames_grid[frames_grid.grid.isin(g3)],frames_grid[frames_grid.grid.isin(g4)]]
	outdata = []
	for fg in frames_grid_group:
		(train,output,test,testoutput,test_gridday) = prep_image(fg,minframe=minframe,channel =channel, testspan=testspan)
		outdata.append((train,output,test,testoutput,test_gridday,fg))
	return outdata

# """Compute softmax values for each sets of scores in x."""
def softmax(x):
	if np.max(x) > 1:
		e_x = np.exp(x/np.max(x))
	else:
		e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum()



def country_dataprep(src_dir,country='USA',testspan = 8,channel = 2,minframe=10,margin=4,pixelsize=8):
	pix = pixelsize+2*margin
	gridpix = np.flip(np.array(range(1,pix**2+1)).reshape(pix,pix),0)
	gridpix = gridpix[margin:pix-margin,margin:pix-margin].flatten()

	if country == 'USA':
		df_pop_pat = pd.read_csv(src_dir+"/USA_covid_data_final.csv")
		counties = pd.read_csv(src_dir+"/USA_counties.csv")
		df_pop_pat = df_pop_pat[df_pop_pat['data_date']>20200307]
		_df_pop_pat = df_pop_pat.groupby(['cfips'])['new_pat'].sum().reset_index()
		area=(23, 49,-124.5, -66.31)
		M=18
		N=30
		Area_df = get_The_Area_Grid(area,M,N,margin=margin,pixelsize=pixelsize,counties=counties)
		_df_area_county = ps.sqldf("""select b.cfips, b.county,b.lat,b.long,b.pop,ifnull(c.new_pat,0) no_pat from counties b 
								left outer join _df_pop_pat c on b.cfips = c.cfips""",locals())
		df_pixel_county = ps.sqldf("""select a.grid,a.pixno,d.cfips cfips,d.county county,d.no_pat, d.pop from Area_df a 
									join _df_area_county d
									on d.lat between a.minlat and a.maxlat and d.long between a.minlong and a.maxlong""",locals())
		df_pixel_county = df_pixel_county[df_pixel_county['pixno'].isin(gridpix)]
		df_pixel_county['ratio']=df_pixel_county.groupby(['grid','pixno','cfips','county'])['no_pat'].apply(lambda x: softmax(x))
	elif country == 'Italy':
		df_pop_pat = pd.read_csv(src_dir+"/Italy_Covid_Patient.csv")
		counties = pd.read_csv(src_dir+"/Italy_counties.csv")
		_df_pop_pat = df_pop_pat.groupby(['ProvinceName','RegionName'])['new_pat'].sum().reset_index()
		area=(36.5, 47,6.61, 18.66)
		M=7
		N=6
		Area_df = get_The_Area_Grid(area,M,N,margin=margin,pixelsize=pixelsize,counties=counties)
		_df_area_county = ps.sqldf("""select b.ProvinceName, c.RegionName,b.lat,b.long,b.pop,ifnull(c.new_pat,0) no_pat from counties b 
								left outer join _df_pop_pat c on b.ProvinceName = c.ProvinceName""",locals())
		df_pixel_county = ps.sqldf("""select a.grid,a.pixno,d.ProvinceName province,d.RegionName region,d.no_pat, d.pop from Area_df a 
									join _df_area_county d
									on d.lat between a.minlat and a.maxlat and d.long between a.minlong and a.maxlong""",locals())
		df_pixel_county = df_pixel_county[df_pixel_county['pixno'].isin(gridpix)]
		df_pixel_county['ratio']=df_pixel_county.groupby(['grid','pixno','province','region'])['no_pat'].apply(lambda x: softmax(x))
	elif country == 'India':
		df_pop_pat = pd.read_csv(src_dir+"/India_Covid_Patient.csv")
		counties = pd.read_csv(src_dir+"/India_counties.csv")
		_df_pop_pat = df_pop_pat.groupby(['District','State'])['new_pat'].sum().reset_index()
		area=(6.665, 36.91,68, 97.77)
		M=21
		N=18
		Area_df = get_The_Area_Grid(area,M,N,margin=margin,pixelsize=pixelsize,counties=counties)
		_df_area_county = ps.sqldf("""select b.District, b.State,b.lat,b.long,b.pop,ifnull(c.new_pat,0) no_pat from counties b 
								left outer join _df_pop_pat c on b.ProvinceName = c.ProvinceName""",locals())
		df_pixel_county = ps.sqldf("""select a.grid,a.pixno,d.District District,d.State State,d.no_pat, d.pop from Area_df a 
									join _df_area_county d
									on d.lat between a.minlat and a.maxlat and d.long between a.minlong and a.maxlong""",locals())
		df_pixel_county = df_pixel_county[df_pixel_county['pixno'].isin(gridpix)]
		df_pixel_county['ratio']=df_pixel_county.groupby(['grid','pixno','District','State'])['no_pat'].apply(lambda x: softmax(x))
	
	frames_grid = frames_df(df_pop_pat,Area_df)
	
	if country == 'USA':
		data = prep_us_data(M,N,frames_grid,minframe = minframe,channel = channel, testspan = testspan)
	else:
		(train,output,test,testoutput,test_gridday) = prep_image(frames_grid,minframe=minframe,channel =channel, testspan=testspan)
		data = (train,output,test,testoutput,test_gridday,frames_grid)
	with open(src_dir+country+"prepdata.pkl", 'wb') as filehandler:
		pickle.dump((data,df_pixel_county),filehandler)