# covid19-spatiotemporal

The high R-naught factor of SARS-CoV-2 has created a race against time for mankind and it necessitates rapid containment actions to control the spread. In such scenario short term accurate spatiotemporal predictions can help understanding the dynamics of the spread in a geographic region and identify hotspots. We propose an ensemble of convolutional LSTM based spatiotemporal model to forecast spread of the epidemic with high resolution and accuracy in a large geographic region. A data preparation method is proposed to convert spatial causal features into set of 2D images with or without temporal component. The model has been trained with available data for USA and Italy. It achieved 5.57% and 0.3% mean absolute percent error for total number of predicted infection cases in a 5day prediction period for USA and Italy respectively.

Modelling can also be done for India

tensorflow_version 1.x is required to run the code

## Running an example
To fetch latest data for a country run the following
```
python main.py --run getdata --country India --dir /home/covid19data/
```
To train a model for a country run the following. Here the span is period of testing data. To use complete data for training span can be set to 0.
```
python main.py --run train --country India --dir /home/covid19data/ --span 5 --hiddenlayers 1 --epoch 20
```
To test the model run the following. Please make sure to keep the testing span same as that of training
```
python main.py --run test --country India --dir /home/covid19data/ --span 5
```
To forecast run the following. Here span can be any time period.
```
python main.py --run forecast --country India --dir /home/covid19data/ --span 10
```
