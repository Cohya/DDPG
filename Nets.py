import tensorflow as tf 
import numpy as np



class DenseLayer(object):
    def __init__(self, M1, M2, activation = tf.nn.relu, i_d = 0, bais = True, initializer = False):
        
        if initializer == 'Uniform':
            W0 = tf.random.uniform(shape = (M1,M2), minval= - 0.003, maxval=0.003)
        else:
            W0 = np.random.randn(M1, M2).astype(np.float32) * np.sqrt(6.0/(float(M1+M2)))
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
       
class ActorNet():
    def __init__(self,observation_dims, action_dims):
        
        self.hidden_layer_dims = [256,256]
        
        self.layers = []
        M1 = observation_dims
        counter_id = 0
        
        for M2 in self.hidden_layer_dims:
            layer = DenseLayer(M1 = M1, M2 = M2, activation = tf.nn.relu, i_d = counter_id , bais = True)
            self.layers.append(layer)
            M1 = M2
            
            counter_id += 1
            
        # last layer
      
        last_layer = DenseLayer(M1 = M1, M2 = action_dims, activation=tf.nn.tanh,
                                i_d= counter_id, bais=True, initializer = 'Uniform')
        
        self.layers.append(last_layer)
        
        
        #  collect trainable params 
        
        self.trainable_params = []
        
        for layer in self.layers:
            self.trainable_params += layer.trainable_params
            
        
    def forward(self, x):
        Z = x
        
        for layer in self.layers:
            Z = layer.forward(Z)
        
        #  for the pendulum the upper action is 2.0 and the lower is -2.0
        Z = 2.0 * Z
        return Z
    

class CriticNet():
    def __init__(self,observation_dims, action_dims):
        # for the state as input 
        layer0 = DenseLayer(M1  = observation_dims, M2 = 16, activation= tf.nn.relu,   i_d = 0, bais=True)
        layer1 = DenseLayer(M1  = 16, M2 = 32, activation= tf.nn.relu,   i_d = 1, bais=True)
        
        # For action as input 
        
        layer_action = DenseLayer( M1 = action_dims, M2 = 32, activation=tf.nn.relu, i_d= 2, bais= True)
        
        
        
        ## layers after concatenation 
        layer_after_0 = DenseLayer(M1  = 64, M2 = 256, activation= tf.nn.relu,   i_d = 3, bais=True)
        layer_after_1 = DenseLayer(M1  = 256, M2 = 256, activation= tf.nn.relu,   i_d = 4, bais=True)
        
        # last layer 
        last_layer = DenseLayer(M1  = 256, M2 = 1, activation= tf.identity,   i_d = 5, bais=True)
        
        
        self.layers = [ layer0, layer1, layer_action, layer_after_0, layer_after_1, last_layer]
        
        ## collect trainable params
        
        self.trainable_params = [] 
        
        for layer in self.layers:
            self.trainable_params += layer.trainable_params
            
        
        
    def forward(self,state, action):
        # print(state.shape, action.shape, type(action))
        # print(action)
        s = state
        
        for i in range(2):
            s = self.layers[i].forward(s)
        
        a = self.layers[2].forward(action)
        
        ## concatanate 
        
        x = tf.concat((s, a), axis = 1)
        
        for j in range(3,6,1):
            x = self.layers[j].forward(x)
            
        return x
        
        
        
            
            
        
        