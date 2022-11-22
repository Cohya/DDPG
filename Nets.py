import tensorflow as tf 
import numpy as np



class DenseLayer(object):
    def __init__(self, M1, M2, activation = tf.nn.relu, i_d = 0, bais = True):
        
        W0 = np.random.rand(M1, M2).astype(np.float32) * np.sqrt(2/float(M1))
        b0 = np.zeros(shape = M2).astype(np.float32)
        
        self.W = tf.Variable(initial_value = W0, name = 'W_dense_%i' % i_d)
        
        if bais :
            self.b = tf.Variable(initial_value = b0, name = 'b_%i' % i_d)
            
            
        self.trainable_params = [self.W, self.b]
        
        self.f = activation
        
    def forward(self,X):
        
        Z = tf.matmul(X, self.W) + self.b
        return self.f(Z)
    

class ANN():
    def __init__(self, input_dims, output_dims, hidden_layer_sizes, last_layer_activation = tf.identity):
        
        """
        input_dims = is the input dimension 
        output_dims = is the output dimension 
        hidden_layer_sizes = is a vector with the hidden layer size e.g. [100,100]
        
        """

        self.layers = []
        
        # Let's build the layers
        M1 = input_dims
        id_counter = 0
        for M2 in hidden_layer_sizes:
            layer = DenseLayer(M1 = M1, M2 = M2, i_d = id_counter)
            
            self.layers.append(layer)
            id_counter += 1
            M1 = M2
            
        
        ## last layer 
        
        last_layer = DenseLayer(M1,
                                M2 = output_dims,
                                i_d = id_counter, 
                                activation = last_layer_activation)
        
        self.layers.append(last_layer)
        
        ## collect all the trainable params 
        self.trainable_params = []
        
        for layer in self.layers:
            self.trainable_params += layer.trainable_params
            
        
    def forward(self,X):
        Z = X 
        
        for layer in self.layers:
            Z = layer.forward(Z)
            
        return Z
       
            