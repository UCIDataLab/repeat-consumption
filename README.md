# repeat-consumption
This is the code used for the paper *Predicting Consumption Patterns with Repeatedand Novel Events.*

## Data
The data used for this project can be found in the UCI Machine Learning repository [here](https://archive.ics.uci.edu/ml/datasets/Repeat+Consumption+Matrices). 
 
You can download the data and put in a ./data directory for the code to work as is.

If you want to change directory for where the data is loaded/saved you need to change util.paths file.

## Reproducing Results
In order to reproduce the results from the paper you can run 

~~~~
cd repeat_consumption
python experiments/run.py
~~~~

(The Data need to have been downloaded and be in the correct directory)


## Using the Code
In order to use the Mixture Model described in the paper you can use the method 
`train_mixture_model` which is in `models/mixture_model.py`. 

The method contains comments on how to use it.

