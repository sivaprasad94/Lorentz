import numpy as np
import matplotlib.pyplot as plt
#from pyESN import ESN
import ESN
import ESN2
# import ESNold as ESN
# import reservoir as ESN
from  sklearn.model_selection import GridSearchCV
import threading
import concurrent.futures

def set_seed(seed=None):
    """Making the seed (for random values) variable if None"""

    # Set the seed
    if seed is None:
        import time
        seed = int((time.time()*10**6) % 4294967295)
    try:
        np.random.seed(seed)
    except Exception as e:
        print( "!!! WARNING !!!: Seed was not set correctly.")
        print( "!!! Seed that we tried to use: "+str(seed))
        print( "!!! Error message: "+str(e))
        seed = None
    print( "Seed used for random values:", seed)
    return seed

## Set a particular seed for the random generator (for example seed = 42), or use a "random" one (seed = None)
# NB: reservoir performances should be averaged accross at least 30 random instances (with the same set of parameters)
seed = 42 #None #42

set_seed(seed) #random.seed(seed)

## load the data and select which parts are used for 'warming', 'training' and 'testing' the reservoir
# 30 seems to be enough for initLen with leak_rate=0.3 and reservoir size (resSize) = 300
initLen = 1000 # number of time steps during which internal activations are washed-out during training
# we consider trainLen including the warming-up period (i.e. internal activations that are washed-out when training)
trainLen = initLen + 3000 # number of time steps during which we train the network
testLen = 4000 # number of time steps during which we test/run the network

data = np.loadtxt('2.txt')
print( "data dimensions", data.shape)



# generate the ESN reservoir
# inSize = outSize = 1 #input/output dimension
# resSize = 300 #reservoir size (for prediction)
# resSize = 1000 #reservoir size (for generation)
# spectral_radius = 1.25
# input_scaling = 1.
def getpap(list1):
    list2=[]
    a=list1[0]
    while a<=list1[1]:
        list2.append(a)
        a=a+list1[2]
    return list2

n_threads = 8
n_reservoir_op =getpap([100,500,100])   # [startValut, endValute, step]
leak_rate_op=getpap([0.1,0.5,0.1])
proba_non_zero_connec_W_op=getpap([0.1,0.5,0.1])
regularization_coef_op=[1e-4,1e-5,1e-6,1e-7,1e-8]
n_inputs =25
input_bias = True # add a constant input to 1
n_outputs = 25
#n_reservoir =300 # number of recurrent units300
#leak_rate = 0.3 # leaking rate (=1/time_constant_of_neurons)
spectral_radius = 1.25 # Scaling of recurrent matrix
input_scaling = 1. # Scaling of input matrix
#proba_non_zero_connec_W = 0.2 # Sparsity of recurrent matrix: Perceptage of non-zero connections in W matrix
proba_non_zero_connec_Win = 1. # Sparsity of input matrix
proba_non_zero_connec_Wfb = 1. # Sparsity of feedback matrix
#regularization_coef =  1e-5 #None # regularization coefficient, if None, pseudo-inverse is use instead of ridge regression
# out_func_activation = lambda x: x
minmse =100


def get_parameter(i,j,t,r):
    n_reservoir = n_reservoir_op[i]
    leak_rate = leak_rate_op[j]
    proba_non_zero_connec_W = proba_non_zero_connec_W_op[t]
    regularization_coef = regularization_coef_op[r]
    N = n_reservoir  # 100
    dim_inp = n_inputs  # 26

    line = np.arange(0, trainLen + testLen, 1)
    W = np.random.rand(N, N) - 0.5
    if input_bias:
        Win = np.random.rand(N, dim_inp + 1) - 0.5
    else:
        Win = np.random.rand(N, dim_inp) - 0.5
    Wfb = np.random.rand(N, n_outputs) - 0.5


    ## delete the fraction of connections given the sparsity (i.e. proba of non-zero connections):
    mask = np.random.rand(N, N)  # create a mask Uniform[0;1]
    W[mask > proba_non_zero_connec_W] = 0  # set to zero some connections given by the mask
    mask = np.random.rand(N, Win.shape[1])
    Win[mask > proba_non_zero_connec_Win] = 0
    # mask = np.random.rand(N,Wfb.shape[1])
    # Wfb[mask > proba_non_zero_connec_Wfb] = 0

    ## SCALING of matrices
    # scaling of input matrix
    Win = Win * input_scaling
    # scaling of recurrent matrix
    # compute the spectral radius of these weights:
    #print('Computing spectral radius...')
    original_spectral_radius = np.max(np.abs(np.linalg.eigvals(W)))
    # TODO: check if this operation is quicker: max(abs(linalg.eig(W)[0])) #from scipy import linalg
    #print("default spectral radius before scaling:", original_spectral_radius)
    # rescale them to reach the requested spectral radius:
    W = W * (spectral_radius / original_spectral_radius)
    #print("spectral radius after scaling", np.max(np.abs(np.linalg.eigvals(W))))

    reservoir = ESN.ESN(lr=leak_rate, W=W, Win=Win, input_bias=input_bias, ridge=regularization_coef,
                        Wfb=None, fbfunc=None)

    """mean=np.mean(data,0)
    var=np.var(data,0)
    print(mean)
    print(var)
    for i in range(len(mean)):
        data[:,i]=data[:,i]-mean[i]
        data[:,i]=data[:,i]/var[i]
    print(data)"""

    train_in = data[0:trainLen, :]
    train_out = data[1:trainLen + 1, :]
    test_in = data[trainLen:trainLen + testLen, :]
    test_out = data[trainLen + 1:trainLen + testLen + 1, :]
    #print(train_in.shape)

    # train_in, train_out =  np.vstack([data[0:trainLen],data[0:trainLen]]), np.vstack([data[0+1:trainLen+1], data[0+1:trainLen+1]])
    # test_in, test_out =  np.vstack([data[trainLen:trainLen+testLen],data[trainLen:trainLen+testLen]]) , np.vstack([data[trainLen+1:trainLen+testLen+1],data[trainLen+1:trainLen+testLen+1]])

    # train_in, train_out =  np.atleast_2d(data[0:trainLen]), np.atleast_2d(data[0+1:trainLen+1])
    # test_in, test_out =  np.atleast_2d(data[trainLen:trainLen+testLen]), np.atleast_2d(data[trainLen+1:trainLen+testLen+1])

    # rearange inputs in correct dimensions
    # train_in, train_out = train_in.T, train_out.T
    # test_in, test_out = test_in.T, test_out.T

    # Dimensions of input/output train/test data
    #print("train_in, train_out dimensions", train_in.shape, train_out.shape)
    #print("test_in, test_out dimensions", test_in.shape, test_out.shape)

    internal_trained = reservoir.train(inputs=[train_in, ], teachers=[train_out, ],
                                       wash_nr_time_step=initLen, verbose=False)
    # print(internal_trained)
    output_pred, internal_pred = reservoir.run(inputs=[test_in, ], reset_state=False)

    errorLen = len(test_out[:])  # testLen #2000

    mse = np.mean((test_out[:] - output_pred[0]) ** 2)
    """if mse < minmse[-1][-1]:
        minmse1 = mse
        n_reservoir_bst = n_reservoir
        proba_non_zero_connec_W_bst = proba_non_zero_connec_W
        regularization_coef_bst = regularization_coef
        leak_rate_bst = leak_rate
        minmse.append([n_reservoir_bst,proba_non_zero_connec_W_bst,regularization_coef_bst,leak_rate_bst,minmse1])"""

    rmse = np.sqrt(mse)  # Root Mean Squared Error: see https://en.wikipedia.org/wiki/Root-mean-square_deviation for more info
    nmrse_mean = abs(rmse / np.mean(test_out[:]))  # Normalised RMSE (based on mean)
    nmrse_maxmin = rmse / abs(np.max(test_out[:]) - np.min(test_out[:]))  # Normalised RMSE (based on max - min)
    str_1=str("n_reservoir=" + str(n_reservoir) +" leak_rate=" + str(leak_rate) + " proba_non_zero_connec_W=" + str(proba_non_zero_connec_W) + " regularization_coef=" + str(regularization_coef) + " mse=" + str(mse))
    #print(str_1)
    return mse,str_1

pool = concurrent.futures.ThreadPoolExecutor(n_threads)  #set the number of threads
str_end=""
for i in range(len(n_reservoir_op)):
    for j in range(len(leak_rate_op)):
        for t in range(len(proba_non_zero_connec_W_op)):
            for r in range(len(regularization_coef_op)):
                yu=pool.submit(get_parameter, i, j, t, r)
                mse_1,str_2=yu.result()
                if mse_1<minmse:
                    str_end=str_2
                    minmse=mse_1
print("********************\n")
print("min_mse= "+str(minmse))
print(str_end)




