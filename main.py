#%tensorflow_version 1.x
from covid19spatiotemporal.dataprep import *
from covid19spatiotemporal.model import *
from covid19spatiotemporal.getdata import *

def getdata(country,dir):
	if country == 'USA':
		fetch_usa_patientdata(dir)
	elif country == 'Italy':
		fetch_italy_patientdata(dir)
	elif country == 'India':
		fetch_india_patientdata(dir)
		
def train(country,dir,span,epoch,hiddenlayers,ensembles=5):
	country_dataprep(dir,country=country,testspan=span, channel = 2,minframe=10,margin=4,pixelsize=8)
	train_country_ensemble(src_dir=dir,country=country,epochs =epoch,hiddenlayers=hiddenlayers,ensembles = ensembles)
	
def test(country,dir,span):
	KL_div,MAPE,_errorframe,MAPE_countrytotal,cumulative_predicttotal_day,predicttotal_country = test_country_ensemble(src_dir=dir,country=country,span=span,margin=4)
	return (KL_div,MAPE,_errorframe,MAPE_countrytotal,cumulative_predicttotal_day,predicttotal_country)
	
def forecast(country,dir,span):
	forecast=forecast_country_cases(src_dir=dir,country=country,span=span)
	return (forecast)
		
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Set the run parameters.')
	parser.add_argument('--run', type=str, help='Specify function to run {getdata,train,test,forecast}')
	parser.add_argument('--country', default = 'India',type=str, help='Specify country name {USA,Italy,India}')
	parser.add_argument('--dir', type=str, help='Specify directory to store all intermediate files')
	parser.add_argument('--span', default = 5, type=int, help='Specify forecast period')
	parser.add_argument('--hiddenlayers', default = 3, type=int, help='Specify number of  hiddenlayers')
	parser.add_argument('--epoch', default = 2, type=int, help='Specify epoch')
	
	args = vars(parser.parse_args())
	
	if args['run'] == 'getdata':
		getdata(args['country'],args['dir'])

	elif args['run'] == 'train':
		train(args['country'],args['dir'],args['span'],args['epoch'],args['hiddenlayers'])
		
	elif args['run'] == 'test':
		KL_div,MAPE,_errorframe,MAPE_countrytotal,cumulative_predicttotal_day,predicttotal_country = test(args['country'],args['dir'],args['span'])
		print ("KL divergence:", KL_div, "Pixel MAPE:", MAPE,"Country MAPE:",MAPE_countrytotal)
		
	elif args['run'] == 'forecast':
		forecast=forecast(args['country'], args['dir'],args['span'])
		print (forecast)
		
	else:
		print ("Unknown parameter given")
		
		
