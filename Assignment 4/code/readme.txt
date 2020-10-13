to run core and completion, just type python core.py or pyton completion.py while in the same directory
as the files
to run challenge, type python simple_linear_regression.py which will default to BGD+MSE while in 
the same directory as the file
to change parameters, type python simple_linear_regression.py -o "optimizer" -m "metric"
e.g python simple_linear_regression.py -o MSE -m MiniBGD
options for optimizer: BGD, PSO, MiniBGD
options for metric: MSE, RMSE, MAE, R2