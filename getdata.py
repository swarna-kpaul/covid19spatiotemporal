import pandas as pd
import urllib.request
import numpy as np
import shapefile
from datetime import datetime
from zipfile import ZipFile
import pandasql as ps
import requests
import json


## getProvinceBoundaryBox function is to get the cordinate details from Mapbox API for ITALY
## Parameter Needed - Province Name
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

# The below function used to get the USA Patient Data Automatically from HARVARD DATABASE COVID Patient Database and will create a timeseries patient file along with population of the Area at county along with a USA County file
## Parameter Needed - Target Directory to save the File
def fetch_us_patientdata(tgtdir):
	url='https://dataverse.harvard.edu/api/access/datafile/3792860?format=original&gbrecs=true'
	urllib.request.urlretrieve(url,tgtdir+'/us_county_confirmed_cases.csv')
	latest_data = pd.read_csv(tgtdir+'/us_county_confirmed_cases.csv')
	allcols = list(latest_data.columns)
	datecols = allcols[allcols.index('HHD10')+1:]
	latest_data = latest_data[['COUNTY', 'NAME']+datecols]
	datecolsmod=[datetime.strptime(i,'%m/%d/%Y').strftime('%Y%m%d') for i in datecols]
	latest_data.columns = ['cfips', 'county']+datecolsmod
	latest_data = latest_data.melt(id_vars=['cfips', 'county'], var_name='data_date', value_name='no_pat')
	latest_data['county']=latest_data['county'].apply(lambda x : x.split(' County')[0])

	url='https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/HIDLTK/OFVFPY'
	urllib.request.urlretrieve(url,tgtdir+'/COUNTY_MAP.zip')
	zip = ZipFile(tgtdir+'/COUNTY_MAP.zip')
	zip.extractall(tgtdir)
	sf = shapefile.Reader(tgtdir+"/CO_CARTO")
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
	us_counties.columns = ['lat','long', 'cfips', 'county', 'pop', 'HHD']
	full_data = pd.merge(latest_data, us_counties, on=['cfips', 'county'])
	if sum(full_data['no_pat']) != sum(latest_data['no_pat']):
		print("fetch failed")
		raise
	full_data['no_pat'] = full_data.groupby(['cfips'])['no_pat'].apply(lambda x: x.cummax())
	full_data['new_pat'] = full_data.groupby(['lat','long'])['no_pat'].diff()
	full_data = full_data.dropna()
	us_counties.to_csv(tgtdir+'us_counties.csv',index=False)
	full_data.to_csv(tgtdir+'USA_covid_data_final.csv',index=False)
	print(' USA Patient Data Created under Directory :'+tgtdir)


## Below function will create the China COVID19 time series Patient file by abosrving data from Harvard Database and it will create County file along with Population Data by county/province
## Parameter Needed - Target Directory to save the File

def fetch_china_patientdata(tgtdir):
	url = 'https://dataverse.harvard.edu/api/access/datafile/3781338?format=original&gbrecs=true'
	urllib.request.urlretrieve(url, tgtdir+'/City_Confirmed_Map_China.csv')
	latest_data = pd.read_csv(tgtdir+'/City_Confirmed_Map_China.csv')
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
	urllib.request.urlretrieve(url,tgtdir+'/china_city_basemap.zip')
	zip = ZipFile(tgtdir+'/china_city_basemap.zip')
	zip.extractall()
	sf = shapefile.Reader(tgtdir+"/china_city_basemap")
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
	full_data.columns = ['city', 'Province', 'data_date', 'no_pat', 'lat', 'long']
	china_pop_data = pd.read_excel('data/China_Population_Data.xlsx')
	china_pop_data['Province'] = china_pop_data['Province'].apply(lambda x: x.split('[')[0])
	full_data = ps.sqldf(
		''' select a.*,b.Population pop from full_data a left join china_pop_data b on a.Province = b.Province ''',
		locals())
	full_data['no_pat'] = full_data.groupby(['city'])['no_pat'].apply(lambda x: x.cummax())
	full_data['new_pat'] = full_data.groupby(['lat','long'])['no_pat'].diff()
	full_data = full_data.dropna()
	china_provinces.to_csv(tgtdir+'China_provinces_data.csv', index=False)
	full_data.to_csv(tgtdir+'China_covid_data_final.csv', index=False)
	print(' China Patient Data Created under Directory :' + tgtdir)


## The below function will give us the Patient count along with population in timeseries manner for ITALY provinces along with County file
## Parameter Needed - Target Directory to save the File
def fetch_italy_patientdata(tgtdir):
	url = 'https://github.com/pcm-dpc/COVID-19/archive/master.zip'
	urllib.request.urlretrieve(url, tgtdir+'/IT_covid19.zip')
	zip = ZipFile(tgtdir+'/IT_covid19.zip')
	zip.extractall(tgtdir)
	latest_data = pd.read_csv(tgtdir+'/COVID-19-master/dati-province/dpc-covid19-ita-province.csv')
	latest_data = ps.sqldf(
		''' select  Date(data) as data_date,denominazione_regione as "RegionName",denominazione_provincia as "ProvinceName", lat, long,totale_casi as "no_pat" from latest_data ''',
		locals())
	latest_data_Area_Regions = latest_data[['RegionName', 'ProvinceName']].drop_duplicates()
	Unique_Provinces = latest_data_Area_Regions['ProvinceName'].unique()
	lat_long_df = pd.DataFrame()
	for i in range(len(Unique_Provinces)):
		if Unique_Provinces[i] != 'In fase di definizione/aggiornamento':
			each_lat_long_df = {}
			each_lat_long_df['ProvinceName'] = [Unique_Provinces[i]]
			Cordinates = getProvinceBoundaryBox(Unique_Provinces[i])
			shapelat = (Cordinates[1] + Cordinates[3]) / 2
			shapelong = (Cordinates[0]+ Cordinates[2]) / 2
			each_lat_long_df['lat'] = [shapelat]
			each_lat_long_df['long'] = [shapelong]
			each_lat_long_df = pd.DataFrame.from_dict(each_lat_long_df)
			lat_long_df = lat_long_df.append(each_lat_long_df)

	full_data = ps.sqldf(
		''' select a.*, b.* from latest_data a left join lat_long_df b on a."ProvinceName" = b."ProvinceName" ''',
		locals())
	#full_data = latest_data
	Dates_in_Data = full_data['data_date'].unique()
	Regions_in_Data = full_data['RegionName'].unique()

	final_Data = pd.DataFrame()
	for eachDate in Dates_in_Data:
		for eachRegion in Regions_in_Data:
			full_region_data = full_data[(full_data['data_date'] == eachDate) & (full_data['RegionName'] == eachRegion)]
			no_of_province = len(full_region_data['ProvinceName'].iloc[:,0]..unique()) - 1
			try:
				UnIdentified = full_region_data[full_region_data['lat'] == 0.000000]['no_pat'].values[0]
			except:
				UnIdentified = 0

			Distribution = round(UnIdentified / no_of_province)
			full_region_data = full_region_data[full_region_data['lat'] != 0.000000]
			full_region_data['no_pat'] = full_region_data['no_pat'].apply(lambda x: int(int(x) + Distribution))

			final_Data = final_Data.append(full_region_data)

	Population_Data = pd.read_excel('data/Italy_population_and_Estimates.xlsx')
	Population_Data = Population_Data[Population_Data.Status.isin(['Province','Metropolitan City','Autonomous Province'])]
	Population_Data_prov = Population_Data[['Name', 'Status', 'Population 2019']]
	Population_Data_prov['Name'] = Population_Data_prov['Name'].apply(lambda x: x.strip())
	IT_counties = ps.sqldf("""select distinct a.ProvinceName,a.lat,a.long, b.'Population 2019' pop 
                      from final_Data a join Population_Data_prov b on a.ProvinceName = b.Name """,locals())
	final_Data = ps.sqldf(
		''' select a.*, b."Population 2019" as pop from final_Data a left join Population_Data_prov b on a."ProvinceName" = b.Name ''',
		locals())
	final_Data['no_pat'] = final_Data.groupby(['ProvinceName'])['no_pat'].apply(lambda x: x.cummax())
	final_Data['new_pat'] = final_Data.groupby(['lat','long'])['no_pat'].diff()
	final_Data = final_Data.dropna()
	IT_counties.to_csv(tgtdir+'/Italy_counties.csv',index=False)
	final_Data.to_csv(tgtdir+'/Italy_Covid_Patient.csv', index=False)
	print(' Italy Patient Data Created under Directory :' + tgtdir)


## The below function will get the Indian COVID-19 time series patient Data and District level details over India, Along with the population file
## Parameter Needed - Target Directory to save the File
def fetch_india_patientdata(tgtdir):
	India_Raw_Data = requests.get('https://api.covid19india.org/raw_data.json').json()['raw_data']
	India_full_Data = pd.DataFrame()
	for eachRecord in India_Raw_Data:
		each_record_df = pd.DataFrame(eachRecord, index=[0])
		India_full_Data = India_full_Data.append(each_record_df)

	India_full_Data = India_full_Data[India_full_Data['detectedstate'] != '']
	India_full_Data['no_pat'] = 1
	India_full_Data = ps.sqldf(
		''' select detecteddistrict,detectedstate,sum("no_pat") as "no_pat" from India_full_Data group by detecteddistrict,detectedstate''',
		locals())

	lat_long_df = pd.read_csv('data/India_District_Wise_Population_Data_with_Lat_Long.csv')
	lat_long_df['detecteddistrict'] = lat_long_df['detecteddistrict'].apply(lambda x: x.strip())
	lat_long_df['detectedstate'] = lat_long_df['detectedstate'].apply(lambda x: x.strip())

	India_Full_Merge_Data = ps.sqldf(
		''' select a.*, ifnull(b.population,0) as pop,ifnull(b.long,0) as long,ifnull(b.lat,0) as lat from India_full_Data a left join lat_long_df b on lower(a.detecteddistrict) = lower(b.detecteddistrict) and lower(a.detectedstate) = lower(b.detectedstate) order by a.no_pat desc''',
		locals())
	unique_States = India_Full_Merge_Data['detectedstate'].unique()
	India_Final_Merge_Data = pd.DataFrame()
	for eachState in unique_States:
		print(eachState)
		state_data_valid = India_Full_Merge_Data[
			(India_Full_Merge_Data['detectedstate'] == eachState) & (India_Full_Merge_Data['pop'] != 0)]
		valid_districts = list(state_data_valid['detecteddistrict'].unique())
		All_State_District = lat_long_df[lat_long_df['detectedstate'] == eachState]
		All_Districts = list(All_State_District['detecteddistrict'].unique())
		missing_districts = list(set(All_Districts) - set(valid_districts))
		number_of_Missing_district = len(missing_districts)
		state_invalid_data = India_Full_Merge_Data[
			(India_Full_Merge_Data['detectedstate'] == eachState) & (India_Full_Merge_Data['pop'] == 0)]
		Total_untagged_Patients = sum(state_invalid_data['no_pat'])
		distribution_df = pd.DataFrame()

		if Total_untagged_Patients != 0:
			if Total_untagged_Patients < number_of_Missing_district:
				state_data_valid.iloc[0, 2] = state_data_valid.iloc[0, 2] + Total_untagged_Patients

			else:
				Patient_distribution = int(round(Total_untagged_Patients / number_of_Missing_district))
				for eachdistrict in missing_districts:
					district_dist_df = {}
					lat_long_df_for_pericular_dist = lat_long_df[
						(lat_long_df['detectedstate'] == eachState) & (lat_long_df['detecteddistrict'] == eachdistrict)]
					district_dist_df['detectedstate'] = [eachState]
					district_dist_df['detecteddistrict'] = [eachdistrict]
					district_dist_df['no_pat'] = [Patient_distribution]
					district_dist_df['pop'] = [lat_long_df_for_pericular_dist.iloc[0, 2]]
					district_dist_df['long'] = [lat_long_df_for_pericular_dist.iloc[0, 3]]
					district_dist_df['lat'] = [lat_long_df_for_pericular_dist.iloc[0, 4]]
					distribution_df = distribution_df.append(pd.DataFrame.from_dict(district_dist_df))

		if distribution_df.empty == True:
			full_State_df = state_data_valid
		else:
			full_State_df = state_data_valid.append(distribution_df)

		India_Final_Merge_Data = India_Final_Merge_Data.append(full_State_df)

	India_Final_Merge_Data.columns = ['no_pat', 'District', 'State', 'lat', 'long', 'pop']
	India_district = lat_long_df[['detecteddistrict', 'detectedstate', 'lat', 'long', 'population']]
	India_district.columns = ['District', 'State', 'lat', 'long', 'pop']
	India_Final_Merge_Data['no_pat'] = India_Final_Merge_Data.groupby(['State','District'])['no_pat'].apply(lambda x: x.cummax())
	India_Final_Merge_Data['new_pat'] = India_Final_Merge_Data.groupby(['lat','long'])['no_pat'].diff()
	India_Final_Merge_Data = India_Final_Merge_Data.dropna()
	India_Final_Merge_Data.to_csv(tgtdir + '/India_Covid_Patient.csv', index=False)
	India_district.to_csv(tgtdir + '/India_district.csv', index=False)
	print(' India Patient Data Created under Directory :' + tgtdir)