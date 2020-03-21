import numpy as np
from scipy import linalg
import sklearn.linear_model as sklm
import tensorflow as tf

class ESN():
    def __init__(self, lr, W, Win, input_bias=True, ridge=None, Wfb=None, fbfunc=None,
                typefloat=np.float64, reg_model=None):
        #TODO : add check if fbfunc is not set when Wfb is not None
        """
        Dimensions of matrices:
            - W : (nr_neurons, nr_neurons)
            - Win : (nr_neurons, dim_input)
            - Wout : (dim_output, nr_neurons)

        Inputs:
            - reg_model: sklearn.linear_model regression object.
                If it is defined, the provided model is used to learn output weights.
                ridge coefficient must not be defined at the same time
                Examples : reg_model = sklearn.linear_model.Lasso(1e-8)
                            reg_model = sklearn.linear_model.Ridge(1e-8)

        """
        self.W = W # reservoir matrix
        self.Win = Win # input matrix

        self.typefloat = typefloat
        self.lr = lr # leaking rate

        self.Wout = None
        self.Wfb = Wfb
        self.fbfunc = fbfunc
        self.N = self.W.shape[1] # nr of neurons
        self.in_bias = input_bias
        self.dim_inp = self.Win.shape[1] # dimension of inputs (including the bias at 1)

        self.ridge = None
        self.reg_model = None
        #self.update_regression_model(ridge, reg_model)

        if self.in_bias:
            pass
        else:
            pass
        if self.Wfb is not None:
            self.dim_out = self.Wfb.shape[1] # dimension of outputs
        else:
            self.dim_out = None
        #self.autocheck_nan()


    def train(self, inputs, teachers, wash_nr_time_step, reset_state=True, verbose=False):
        #x = tf.constant(0, tf.float32, [self.N, 1])
        with tf.variable_scope("train",reuse=tf.AUTO_REUSE):
            u=tf.placeholder(tf.float32,[self.dim_inp,1])
            y=tf.placeholder(tf.float32,[self.dim_inp,1])
            x=tf.placeholder(tf.float32,[self.N,1])

            A=tf.constant(self.W,tf.float32,[self.N, self.N])
            Win=tf.constant(self.Win,tf.float32,[self.N, self.dim_inp])
            Wout=tf.get_variable("Wout",[self.dim_inp, self.N],dtype= tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            x_op=(1-self.lr) * x  +  self.lr*tf.tanh(tf.matmul(A,x)+tf.matmul(Win,u))
            #x_op = tf.tanh(tf.matmul(A, x) + tf.matmul(Win, u))
            out=tf.matmul(Wout,x_op)
            loss=tf.norm(out-y, ord='euclidean')
            train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())  # 初始化变量
            for (j, (inp, tea)) in enumerate(zip(inputs, teachers)):
                inp = inp[:,:,np.newaxis]
                tea = tea[:,:,np.newaxis]
                input=np.zeros([self.N,1])
                for t in range(inp.shape[0]):
                    x_out,_=sess.run((x_op,train_step),feed_dict={u:inp[t,:],y:tea[t,:],x:input})
                    input=np.array(x_out)
                    #print(x_out)
            return x_out

    def run(self, inputs,x_in, reset_state=True, verbose=False):
        all_outputs = [None] * len(inputs)
        #x = tf.constant(x_in, tf.float32, [self.N, 1])

        with tf.variable_scope("test",reuse=tf.AUTO_REUSE):
            u = tf.placeholder(tf.float32,[self.dim_inp, 1] )
            x = tf.placeholder(tf.float32, [self.N, 1])
            A = tf.constant(self.W, tf.float32, [self.N, self.N])
            Win = tf.constant(self.Win, tf.float32, [self.N, self.dim_inp])
            Wout = tf.get_variable("Wout", [self.dim_inp, self.N], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            x_op=(1-self.lr) * x+self.lr*tf.tanh(tf.matmul(A,x)+tf.matmul(Win,tf.matmul(Wout,x)))
            #x_op = tf.tanh(tf.matmul(A,x)+tf.matmul(Win,tf.matmul(Wout,x)))
            out=tf.matmul(Wout,x_op)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())  # 初始化变量
                for (j, inp) in enumerate(inputs):
                    all_outputs[j] = np.zeros((self.dim_inp, inp.shape[0]), dtype=self.typefloat)
                    input=x_in
                    for t in range(inp.shape[0]):
                        x_op1,out1=sess.run((x_op,out),feed_dict={x:input})
                        input=x_op1
                        all_outputs[j][:, t] = out1.reshape(-1, )
            return [st.T for st in all_outputs],0















