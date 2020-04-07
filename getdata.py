import pandas as pd
import urllib.request
import numpy as np
import shapefile
from datetime import datetime
from zipfile import ZipFile

def fetch_us_patientdata(tgtdir):
	url='https://dataverse.harvard.edu/api/access/datafile/3787733?format=original&gbrecs=true'
	urllib.request.urlretrieve(url,'us_county_confirmed_cases.csv')
	latest_data = pd.read_csv('us_county_confirmed_cases.csv')
	allcols = list(latest_data.columns)
	datecols = allcols[allcols.index('HHD10')+1:]
	latest_data = latest_data[['COUNTY', 'NAME']+datecols]
	datecolsmod=[datetime.strptime(i,'%m/%d/%Y').strftime('%Y%m%d') for i in datecols]
	latest_data.columns = ['cfips', 'county']+datecolsmod
	latest_data = latest_data.melt(id_vars=['cfips', 'county'], var_name='data_date', value_name='no_pat')
	latest_data['county']=latest_data['county'].apply(lambda x : x.split(' County')[0])

	url='https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/HIDLTK/OFVFPY'
	urllib.request.urlretrieve(url,'COUNTY_MAP.zip')
	zip = ZipFile('COUNTY_MAP.zip')
	zip.extractall()
	#!unzip ./COUNTY_MAP.zip
	sf = shapefile.Reader("CO_CARTO")
	shape_df = pd.DataFrame()
	shapes = sf.shapes()
	records = sf.records()
	for eachrec in range(len(records)):
		eachRec = {}
		shapebbbox = shapes[eachrec].bbox
		shapelat = (shapebbbox[1]+shapebbbox[3])/2
		shapelong = (shapebbbox[0]+shapebbbox[2])/2
		eachRec['lat']=[shapelat]
		eachRec['long']=[shapelong]
		eachRec['county_fips']=[records[eachrec][0]]
		eachRec['county_name']=[records[eachrec][1]]
		eachRec['POP']=[records[eachrec][10]]
		eachRec['HHD']=[records[eachrec][11]]
		shape_df = shape_df.append(pd.DataFrame.from_dict(eachRec))

	us_counties = shape_df
	us_counties['county_name']=us_counties['county_name'].apply(lambda x : x.split(' County')[0])
	us_counties['county_fips'] = us_counties['county_fips'].apply(lambda x : int(x))
	
	us_counties.columns = ['lat', 'long','cfips','county', 'pop','HHD']
	full_data = pd.merge(latest_data,us_counties,on=['cfips','county'])
	if sum(full_data['no_pat']) != sum(latest_data['no_pat']):
		print("fetch failed")
		raise
	full_data['no_pat'] = full_data.groupby(['cfips'])['no_pat'].apply(lambda x: x.cummax())
	us_counties.to_csv(tgtdir+'us_counties.csv',index=False)	
	full_data.to_csv(tgtdir+'USA_covid_data_final.csv',index=False)