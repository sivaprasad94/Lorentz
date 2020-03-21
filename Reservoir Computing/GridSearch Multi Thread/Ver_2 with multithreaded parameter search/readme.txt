1.Parameter description of prediction model
initLen = 500 # number of time steps during which internal activations are washed-out during training
# we consider trainLen including the warming-up period (i.e. internal activations that are washed-out when training)
trainLen = initLen + 1300 # number of time steps during which we train the network£¬If you want to change the training data length, please modify here
testLen = 1000 # number of time steps during which we test/run the network£¬If you need to modify the test data length, please modify here
data = np.loadtxt('MackeyGlass_t17.txt')
n_inputs = 1 # Number of input nodes
input_bias = True # add a constant input to 1
n_outputs = 2 # Number of output nodes
n_reservoir = 300 # number of recurrent units
leak_rate = 0.3 # leaking rate (=1/time_constant_of_neurons)
spectral_radius = 1.25 # Scaling of recurrent matrix
input_scaling = 1. # Scaling of input matrix
proba_non_zero_connec_W = 0.2 # Sparsity of recurrent matrix: Perceptage of non-zero connections in W matrix
proba_non_zero_connec_Win = 1. # Sparsity of input matrix
proba_non_zero_connec_Wfb = 1. # Sparsity of feedback matrix
regularization_coef =  1e-8 #None # regularization coefficient, if None, pseudo-inverse is use instead of ridge regression
num=0 # Number of forecast data

The above configuration parameters show that there are 500 data as initial period, 1300 data as training period and 1000 data as test data. By inputting one node to predict the change curve of two nodes, Num = 0 means to predict Y & Z with X.


2.Simulation model parameters
initLen = 500 # number of time steps during which internal activations are washed-out during training
# we consider trainLen including the warming-up period (i.e. internal activations that are washed-out when training)
trainLen = initLen + 1300 # number of time steps during which we train the network£¬If you want to change the training data length, please modify here
testLen = 1000 # number of time steps during which we test/run the network£¬If you need to modify the test data length, please modify here
data = np.loadtxt('MackeyGlass_t17.txt')
n_inputs = 3 # Number of input nodes£¬In the prediction model, n _inputs should be equal to n_outputs
input_bias = True # add a constant input to 1
n_outputs = 3 # Number of output nodes,In the prediction model, n _inputs should be equal to n_outputs
n_reservoir = 300 # number of recurrent units
leak_rate = 0.3 # leaking rate (=1/time_constant_of_neurons)
spectral_radius = 1.25 # Scaling of recurrent matrix
input_scaling = 1. # Scaling of input matrix
proba_non_zero_connec_W = 0.2 # Sparsity of recurrent matrix: Perceptage of non-zero connections in W matrix
proba_non_zero_connec_Win = 1. # Sparsity of input matrix
proba_non_zero_connec_Wfb = 1. # Sparsity of feedback matrix
regularization_coef =  1e-8 #None # regularization coefficient, if None, pseudo-inverse is use instead of ridge regression
N = n_reservoir#100
dim_inp = n_inputs #26

The above configuration parameters show that there are 500 data as initial period, 1300 data as training period and 1000 data as test data. Through the input of three nodes to simulate their respective change curve.It should be noted that in the simulation model n _inputs should be equal to n_outputs


3.Function
The file name of the input data is mackeyglass_t17.txt. If you have new data, please input it into the corresponding file according to the sample format in the program. We also support you to embed your data into the model. You only need to modify the
Data = NP. Loadtext ('mackeyglass_t17.txt ')
The read file can be modified by the file name in.

After processing the data, if you want to run the simulation model, you only need to run the simulation model.py file. If you want to run the prediction model, you only need to run the prediction model.py file.

