import numpy as np
import matplotlib.pyplot as plt
import ESN
import pandas as pd


def ESN_TUNE(n_reservoir, leak_rate, regularization_coef, spectral_radius, n_inputs,n_outputs, initLen, train_in, train_out, valid_in, valid_out):
    Best_Parameters = []
    input_bias = True
    n_r = n_reservoir
    l_r = leak_rate
    r_c = regularization_coef
    s_r = spectral_radius
    dim_inp = n_inputs
    n_outputs = n_outputs
    dim_inp = n_inputs
    initLen = initLen
    for N in n_r:
        for L in l_r:
            for R in r_c:
                for i in range(len(s_r)):
                    parameters = []
                    N =  N#100
                    spectral_radius = s_r[i]
                    input_scaling = 1. # Scaling of input matrix
                    proba_non_zero_connec_W = 0.2 # Sparsity of recurrent matrix: Perceptage of non-zero connections in W matrix
                    proba_non_zero_connec_Win = 1. # Sparsity of input matrix
                    proba_non_zero_connec_Wfb = 1.
                        ### Generating random weight matrices with custom method
                    W = np.random.rand(N,N) - 0.5
                    if input_bias:
                        Win = np.random.rand(N,dim_inp+1) - 0.5
                    else:
                        Win = np.random.rand(N,dim_inp) - 0.5
                    Wfb = np.random.rand(N,n_outputs) - 0.5
                    mask = np.random.rand(N,N) # create a mask Uniform[0;1]
                    W[mask > proba_non_zero_connec_W] = 0 # set to zero some connections given by the mask
                    mask = np.random.rand(N,Win.shape[1])
                    Win[mask > proba_non_zero_connec_Win] = 0
                    Win = Win * input_scaling
                    original_spectral_radius = np.max(np.abs(np.linalg.eigvals(W)))
                    W = W * (spectral_radius / original_spectral_radius)
                    reservoir = ESN.ESN(lr=L, W=W, Win=Win, input_bias=input_bias, ridge=R, Wfb=None, fbfunc=None)
                    internal_trained = reservoir.train(inputs=[train_in], teachers=[train_out], wash_nr_time_step=initLen, verbose=False)
                    output_pred, internal_pred = reservoir.run(inputs=[valid_in,], reset_state=False)
                    df_pred = pd.DataFrame(output_pred[0])
                    df_test_out = pd.DataFrame(valid_out)
                    X_MSE = np.mean((df_test_out[0][:] - df_pred[0])**2)
                    Y_MSE = np.mean((df_test_out[1][:] - df_pred[1])**2)
                    Z_MSE = np.mean((df_test_out[2][:] - df_pred[2])**2)
#                     print('For ', N,L,R,s_r[i], 'parameters:: ', 'X, Y, Z MSE = ', X_MSE, Y_MSE, Z_MSE)
#                     print('RMSE = ', np.sqrt(X_MSE**2+Y_MSE**2+Z_MSE**2))
                    RMSE = np.sqrt(X_MSE**2+Y_MSE**2+Z_MSE**2)
                    parameters.append({'n_reservoir':N,'leaky_rate':L,'regularization_coef':R,'spectral_radius' :s_r[i]})
                    Best_Parameters.append([parameters, RMSE, X_MSE])
    df = pd.DataFrame(Best_Parameters)
    return df.iloc[np.argmin(np.array(df[1]))][0][0]

