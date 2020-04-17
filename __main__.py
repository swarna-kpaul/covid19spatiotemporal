
from covid19spatiotemporal import *

### Asks for the Country for which you want to prepare data
getCountry = input('Please enter the country name for which you want to prepare the data : (USA, Italy, China)')
### Asks for the directory path you want to save the file
getDirectory = input('Please mention the directory path you want save file:')
directory = getDirectory+'/'

if lower(getCountry) == 'italy':
	fetch_italy_patientdata(directory)
elif lower(getCountry) == 'usa' or lower(getCountry) == 'united states' or lower(getCountry) == 'america':
	fetch_us_patientdata(directory)
elif lower(getCountry) == 'china':
	fetch_china_patientdata(directory)
elif lower(getCountry) == 'india':
	fetch_india_patientdata(directory)
