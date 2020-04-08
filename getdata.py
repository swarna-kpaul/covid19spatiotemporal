import pandas as pd
import urllib.request
import numpy as np
import shapefile
from datetime import datetime
from zipfile import ZipFile
import pandasql as ps
import requests
import json


def getProvinceBoundaryBox(provinceName):
	Place_Details = requests.get(
		'http://api.mapbox.com/geocoding/v5/mapbox.places/' + provinceName + '%20province%20Italy.json?access_token=pk.eyJ1Ijoic2Fpa2F0amFuYTIzMDkiLCJhIjoiY2s4OXMzbnM5MDByYjNsbXpqeDRncWptbyJ9.zGt4oEGDDYra_yRTGbmqcg').json()[
		'features']
	for eachPlace in Place_Details:
		try:
			if eachPlace['context'][0]['text'] == 'Italy' or eachPlace['context'][1]['text'] == 'Italy':
				getBbox = eachPlace['bbox']
		except:
			continue

	return getBbox


def fetch_us_patientdata(tgtdir):
	url='https://dataverse.harvard.edu/api/access/datafile/3792860?format=original&gbrecs=true'
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
		shapelat = (shapebbbox[1] + shapebbbox[3]) / 2
		shapelong = (shapebbbox[0] + shapebbbox[2]) / 2
		eachRec['lat'] = [shapelat]
		eachRec['long'] = [shapelong]
		eachRec['county_fips'] = [records[eachrec][0]]
		eachRec['county_name'] = [records[eachrec][1]]
		eachRec['POP'] = [records[eachrec][10]]
		eachRec['HHD'] = [records[eachrec][11]]
		shape_df = shape_df.append(pd.DataFrame.from_dict(eachRec))

	us_counties = shape_df
	us_counties['county_name'] = us_counties['county_name'].apply(lambda x: x.split(' County')[0])
	us_counties['county_fips'] = us_counties['county_fips'].apply(lambda x: int(x))
	us_counties.columns = ['lat','long', 'cfips', 'county', 'Population', 'HHD']
	full_data = pd.merge(latest_data, us_counties, on=['cfips', 'county'])
	if sum(full_data['no_pat']) != sum(latest_data['no_pat']):
		print("fetch failed")
		raise
	full_data['no_pat'] = full_data.groupby(['cfips'])['no_pat'].apply(lambda x: x.cummax())
	us_counties.to_csv(tgtdir+'us_counties.csv',index=False)	
	full_data.to_csv(tgtdir+'USA_covid_data_final.csv',index=False)
	print(' USA Patient Data Created under Directory :'+tgtdir)


def fetch_china_patientdata(tgtdir):
	url = 'https://dataverse.harvard.edu/api/access/datafile/3781338?format=original&gbrecs=true'
	urllib.request.urlretrieve(url, 'City_Confirmed_Map_China.csv')
	latest_data = pd.read_csv('City_Confirmed_Map_China.csv')
	latest_data = latest_data[
		['GbCity', 'GbProv', 'City_EN', 'Prov_EN', 'N_C_0115', 'N_C_0116', 'N_C_0117', 'N_C_0118', 'N_C_0119',
		 'N_C_0120', 'N_C_0121', 'N_C_0122', 'N_C_0123', 'N_C_0124', 'N_C_0125', 'N_C_0126', 'N_C_0127', 'N_C_0128',
		 'N_C_0129', 'N_C_0130', 'N_C_0131', 'N_C_0201', 'N_C_0202', 'N_C_0203', 'N_C_0204', 'N_C_0205', 'N_C_0206',
		 'N_C_0207', 'N_C_0208', 'N_C_0209', 'N_C_0210', 'N_C_0211', 'N_C_0212', 'N_C_0213', 'N_C_0214', 'N_C_0215',
		 'N_C_0216', 'N_C_0217', 'N_C_0218', 'N_C_0219', 'N_C_0220', 'N_C_0221', 'N_C_0222', 'N_C_0223', 'N_C_0224',
		 'N_C_0225', 'N_C_0226', 'N_C_0227', 'N_C_0228', 'N_C_0229', 'N_C_0301', 'N_C_0302', 'N_C_0303', 'N_C_0304',
		 'N_C_0305', 'N_C_0306', 'N_C_0307', 'N_C_0308', 'N_C_0309', 'N_C_0310', 'N_C_0311', 'N_C_0312', 'N_C_0313',
		 'N_C_0314', 'N_C_0315', 'N_C_0316', 'N_C_0317', 'N_C_0318', 'T_C_0115', 'T_C_0116', 'T_C_0117', 'T_C_0118',
		 'T_C_0119', 'T_C_0120', 'T_C_0121', 'T_C_0122', 'T_C_0123', 'T_C_0124', 'T_C_0125', 'T_C_0126', 'T_C_0127',
		 'T_C_0128', 'T_C_0129', 'T_C_0130', 'T_C_0131', 'T_C_0201', 'T_C_0202', 'T_C_0203', 'T_C_0204', 'T_C_0205',
		 'T_C_0206', 'T_C_0207', 'T_C_0208', 'T_C_0209', 'T_C_0210', 'T_C_0211', 'T_C_0212', 'T_C_0213', 'T_C_0214',
		 'T_C_0215', 'T_C_0216', 'T_C_0217', 'T_C_0218', 'T_C_0219', 'T_C_0220', 'T_C_0221', 'T_C_0222', 'T_C_0223',
		 'T_C_0224', 'T_C_0225', 'T_C_0226', 'T_C_0227', 'T_C_0228', 'T_C_0229', 'T_C_0301', 'T_C_0302', 'T_C_0303',
		 'T_C_0304', 'T_C_0305', 'T_C_0306', 'T_C_0307', 'T_C_0308', 'T_C_0309', 'T_C_0310', 'T_C_0311', 'T_C_0312',
		 'T_C_0313', 'T_C_0314', 'T_C_0315', 'T_C_0316', 'T_C_0317', 'T_C_0318']]
	latest_data['City_EN'] = latest_data['City_EN'].apply(lambda x: x.split('(')[0])
	latest_data.columns = ['GbCity', 'GbProv', 'city', 'Province', 'N_C_0115', 'N_C_0116', 'N_C_0117', 'N_C_0118',
						   'N_C_0119', 'N_C_0120', 'N_C_0121', 'N_C_0122', 'N_C_0123', 'N_C_0124', 'N_C_0125',
						   'N_C_0126', 'N_C_0127', 'N_C_0128', 'N_C_0129', 'N_C_0130', 'N_C_0131', 'N_C_0201',
						   'N_C_0202', 'N_C_0203', 'N_C_0204', 'N_C_0205', 'N_C_0206', 'N_C_0207', 'N_C_0208',
						   'N_C_0209', 'N_C_0210', 'N_C_0211', 'N_C_0212', 'N_C_0213', 'N_C_0214', 'N_C_0215',
						   'N_C_0216', 'N_C_0217', 'N_C_0218', 'N_C_0219', 'N_C_0220', 'N_C_0221', 'N_C_0222',
						   'N_C_0223', 'N_C_0224', 'N_C_0225', 'N_C_0226', 'N_C_0227', 'N_C_0228', 'N_C_0229',
						   'N_C_0301', 'N_C_0302', 'N_C_0303', 'N_C_0304', 'N_C_0305', 'N_C_0306', 'N_C_0307',
						   'N_C_0308', 'N_C_0309', 'N_C_0310', 'N_C_0311', 'N_C_0312', 'N_C_0313', 'N_C_0314',
						   'N_C_0315', 'N_C_0316', 'N_C_0317', 'N_C_0318', 'T_C_0115', 'T_C_0116', 'T_C_0117',
						   'T_C_0118', 'T_C_0119', 'T_C_0120', 'T_C_0121', 'T_C_0122', 'T_C_0123', 'T_C_0124',
						   'T_C_0125', 'T_C_0126', 'T_C_0127', 'T_C_0128', 'T_C_0129', 'T_C_0130', 'T_C_0131',
						   'T_C_0201', 'T_C_0202', 'T_C_0203', 'T_C_0204', 'T_C_0205', 'T_C_0206', 'T_C_0207',
						   'T_C_0208', 'T_C_0209', 'T_C_0210', 'T_C_0211', 'T_C_0212', 'T_C_0213', 'T_C_0214',
						   'T_C_0215', 'T_C_0216', 'T_C_0217', 'T_C_0218', 'T_C_0219', 'T_C_0220', 'T_C_0221',
						   'T_C_0222', 'T_C_0223', 'T_C_0224', 'T_C_0225', 'T_C_0226', 'T_C_0227', 'T_C_0228',
						   'T_C_0229', 'T_C_0301', 'T_C_0302', 'T_C_0303', 'T_C_0304', 'T_C_0305', 'T_C_0306',
						   'T_C_0307', 'T_C_0308', 'T_C_0309', 'T_C_0310', 'T_C_0311', 'T_C_0312', 'T_C_0313',
						   'T_C_0314', 'T_C_0315', 'T_C_0316', 'T_C_0317', 'T_C_0318']
	latest_data = latest_data.melt(id_vars=['GbCity', 'GbProv', 'city', 'Province'], var_name='Date',
								   value_name='No of Patient')

	New_Patients = ps.sqldf(
		''' select GbCity,GbProv,city,Province,Date,"No of Patient" from latest_data where Date like "N_C_%" ''',
		locals())
	New_Patients['Date'] = New_Patients['Date'].apply(lambda x: '2020' + x.split('N_C_')[1])
	New_Patients.columns = ['GbCity', 'GbProv', 'city', 'Province', 'Date', 'New Patient Count']
	Total_Patients = ps.sqldf(
		''' select GbCity,GbProv,city,Province,Date,"No of Patient" from latest_data where Date like "T_C_%" ''',
		locals())
	Total_Patients['Date'] = Total_Patients['Date'].apply(lambda x: '2020' + x.split('T_C_')[1])
	Total_Patients.columns = ['GbCity', 'GbProv', 'city', 'Province', 'Date', 'Total Patient Count']
	latest_data_Normalized = pd.merge(New_Patients, Total_Patients, on=['GbCity', 'GbProv', 'city', 'Province', 'Date'])
	latest_data_Normalized['GbCity'] = latest_data_Normalized['GbCity'].apply(lambda x: str(x))
	latest_data_Normalized['GbProv'] = latest_data_Normalized['GbProv'].apply(lambda x: str(x))
	url='https://dvn-cloud.s3.amazonaws.com/10.7910/DVN/MR5IJN/1710944b44b-ce6a2df0b32e?response-content-disposition=attachment%3B%20filename%2A%3DUTF-8%27%27china_city_basemap.zip&response-content-type=application%2Fzipped-shapefile&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20200408T040239Z&X-Amz-SignedHeaders=host&X-Amz-Expires=3600&X-Amz-Credential=AKIAIEJ3NV7UYCSRJC7A%2F20200408%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=ed0cbb34d3e1a129167cbd353afc469d13ddaf4dc14520366df279219b422957'
	urllib.request.urlretrieve(url,'china_city_basemap.zip')
	zip = ZipFile('china_city_basemap.zip')
	zip.extractall()
	sf = shapefile.Reader("china_city_basemap")
	shape_df = pd.DataFrame()
	shapes = sf.shapes()
	records = sf.records()
	for eachrec in range(len(records)):
		eachRec = {}
		shapebbbox = shapes[eachrec].bbox
		shapelat = (shapebbbox[1] + shapebbbox[3]) / 2
		shapelong = (shapebbbox[0] + shapebbbox[2]) / 2
		eachRec['lat'] = [shapelat]
		eachRec['long'] = [shapelong]
		eachRec['GbCity'] = [records[eachrec][0]]
		eachRec['city'] = [records[eachrec][2]]
		eachRec['GbProv'] = [records[eachrec][3]]
		eachRec['Province'] = [records[eachrec][5]]
		eachRec['Shape_Area'] = [records[eachrec][6]]
		shape_df = shape_df.append(pd.DataFrame.from_dict(eachRec))

	china_provinces = shape_df
	china_provinces['GbProv'] = china_provinces['GbProv'].apply(lambda x: str(x))
	full_data = pd.merge(latest_data_Normalized, china_provinces, on=['city', 'Province'])
	full_data = full_data[['city', 'Province', 'Date', 'Total Patient Count', 'lat', 'long']]
	full_data.columns = ['city', 'Province', 'Date', 'no_pat', 'lat', 'long']
	china_pop_data = pd.read_excel('data/China_Population_Data.xlsx')
	china_pop_data['Province'] = china_pop_data['Province'].apply(lambda x: x.split('[')[0])
	full_data = ps.sqldf(
		''' select a.*,b.Population from full_data a left join china_pop_data b on a.Province = b.Province ''',
		locals())
	china_provinces.to_csv(tgtdir+'China_provinces_data.csv', index=False)
	full_data.to_csv(tgtdir+'China_covid_data_final.csv', index=False)
	print(' China Patient Data Created under Directory :' + tgtdir)


def fetch_italy_patientdata(tgtdir):
	url = 'https://github.com/pcm-dpc/COVID-19/archive/master.zip'
	urllib.request.urlretrieve(url, 'IT_covid19.zip')
	zip = ZipFile('IT_covid19.zip')
	zip.extractall()
	latest_data = pd.read_csv('COVID-19-master/dati-province/dpc-covid19-ita-province.csv')
	latest_data = ps.sqldf(
		''' select  Date(data) as Date,denominazione_regione as "Region Name",denominazione_provincia as "Province Name", lat, long,totale_casi as "no_pat" from latest_data ''',
		locals())
	# latest_data_Area_Regions = latest_data[['Region Name', 'Province Name']].drop_duplicates()
	# Unique_Provinces = latest_data_Area_Regions['Province Name'].unique()
	# lat_long_df = pd.DataFrame()
	# for i in range(len(Unique_Provinces)):
	# 	if Unique_Provinces[i] != 'In fase di definizione/aggiornamento':
	# 		each_lat_long_df = {}
	# 		each_lat_long_df['Province Name'] = [Unique_Provinces[i]]
	# 		Cordinates = getProvinceBoundaryBox(Unique_Provinces[i])
	# 		shapelat = (Cordinates[1] + Cordinates[3]) / 2
	# 		shapelong = (Cordinates[0]+ Cordinates[2]) / 2
	# 		each_lat_long_df['lat'] = [shapelat]
	# 		each_lat_long_df['long'] = [shapelong]
	# 		each_lat_long_df = pd.DataFrame.from_dict(each_lat_long_df)
	# 		lat_long_df = lat_long_df.append(each_lat_long_df)
	#
	# full_data = ps.sqldf(
	# 	''' select a.*, b.* from latest_data a left join lat_long_df b on a."Province Name" = b."Province Name" ''',
	# 	locals())
	full_data = latest_data
	Dates_in_Data = full_data['Date'].unique()
	Regions_in_Data = full_data['Region Name'].unique()

	final_Data = pd.DataFrame()
	for eachDate in Dates_in_Data:
		for eachRegion in Regions_in_Data:
			full_region_data = full_data[(full_data['Date'] == eachDate) & (full_data['Region Name'] == eachRegion)]
			no_of_province = len(full_region_data['Province Name'].unique()) - 1
			try:
				UnIdentified = full_region_data[full_region_data['lat'] == 0.000000]['no_pat'].values[0]
			except:
				UnIdentified = 0

			Distribution = round(UnIdentified / no_of_province)
			full_region_data = full_region_data[full_region_data['lat'] != 0.000000]
			full_region_data['no_pat'] = full_region_data['no_pat'].apply(lambda x: int(int(x) + Distribution))

			final_Data = final_Data.append(full_region_data)

	# final_Data.columns = ['Date', 'Region Name', 'Province Name', 'lat', 'long', 'No_Pat', 'Province Name Extra','lat','long']
	# final_Data = final_Data[['Date', 'Region Name', 'Province Name', 'lat', 'long', 'No_Pat', 'lat', 'long']]

	Population_Data = pd.read_excel('data/Italy_population_and_Estimates.xlsx')
	Population_Data_prov = Population_Data[['Name', 'Status', 'Population 2019']]
	Population_Data_prov['Name'] = Population_Data_prov['Name'].apply(lambda x: x.strip())
	final_Data = ps.sqldf(
		''' select a.*, b."Population 2019" from final_Data a left join Population_Data_prov b on a."Province Name" = b.Name ''',
		locals())
	final_Data['no_pat'] = final_Data.groupby(['Province Name'])['no_pat'].apply(lambda x: x.cummax())
	final_Data.to_csv(tgtdir+'Italy_Covid_Patient.csv', index=False)
	print(' Italy Patient Data Created under Directory :' + tgtdir)

if __name__ == '__main__':
	getCountry = input('Please enter the country name for which you want to prepare the data : (USA, Italy, China)')
	getDirectory = input('Please mention the directory path you want save file:')
	directory = getDirectory+'/'

	if lower(getCountry) == 'italy':
		fetch_italy_patientdata(directory)

	elif lower(getCountry) == 'usa' or lower(getCountry) == 'united states' or lower(getCountry) == 'america':
		fetch_us_patientdata(directory)

	elif lower(getCountry) == 'china':
		fetch_china_patientdata(directory)

else:
	print('All Get Data Modules imported Successfully!!')

