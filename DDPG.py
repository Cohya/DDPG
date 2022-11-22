import tensorflow as tf 
import numpy as np 
from Nets import ANN

def orgenize_dims(x):
    n,h,w,c = x.shape
    x = tf.reshape(x, shape=(n,h,c,w))
    x = tf.transpose(x,perm = (0,2,1,3))
    x = tf.reshape(x, shape = (n, h*w*c))
    return x 

def copy_weights(main_net, copy_net):
    for w, w2 in zip(main_net.trainable_params, copy_net.trainable_params):
        w2.assign(w)
    
    
class DDPG(object):
    def __init__(self, action_dims, observation_dims, args, replay_memory, action_clip = [], 
                 gamma = 0.99,
                 decay = 0.995):
        
        self.replay_memory = replay_memory
        self.action_clip = action_clip
        ## mu net which is the Actor net 
        self.mu_Net = ANN(input_dims = observation_dims,
                          output_dims = action_dims,
                          hidden_layer_sizes= args.hidden_layer_sizes_mu,
                          last_layer_activation = tf.nn.tanh)# The output activation can be also identity 
        
        self.gamma = gamma
        self.decay = decay
        # The critic 
        self.q_Net = ANN(input_dims = int(action_dims + observation_dims),
                         output_dims = 1,
                         hidden_layer_sizes = args.hidden_layer_sizes_Q,
                         last_layer_activation = tf.identity)
        
        #### We should hold a copy from each one of them 
        
        self.mu_Net_targ = ANN(input_dims = observation_dims,
                          output_dims = action_dims,
                          hidden_layer_sizes= args.hidden_layer_sizes_mu,
                          last_layer_activation = tf.nn.tanh)
        
        self.q_Net_targ = ANN(input_dims = int(action_dims + observation_dims),
                         output_dims = 1,
                         hidden_layer_sizes = args.hidden_layer_sizes_Q,
                         last_layer_activation = tf.identity)
        
        ## copy weights 
        copy_weights(self.q_Net, self.q_Net_targ)
        copy_weights(self.mu_Net, self.mu_Net_targ)
        
        
        self.mu_optimizer = tf.keras.optimizers.Adam(lr = 1e-3)
        self.q_optimizer = tf.keras.optimizers.Adam(lr = 1e-3)
        
      # ""I think we should normelized the observation""  
    def get_action(self,state, noise_scale):
        # calculate mu, which suppose to be the best action  
        a = self.mu_Net.forward(state) * 2.0  ## this is the action max of the pendulum 
        # print(state)
        # Add Gaussian noise (during training noise_scale = 0.1 or other value higher than 0)
        a += tf.random.normal(shape = a.shape) * noise_scale
        
        if len(self.action_clip) != 0:
            a = tf.clip_by_value(a, self.action_clip[0], self.action_clip[1])
        # print(a.numpy())
        assert (a[0][0].numpy() <= 2.0),"Oh no! This assertion failed!" 
        assert (a[0][0].numpy() >= -2.0),"Oh no! This assertion failed!" 
        return a
    
    def q_targets_fun(self,next_states, rewards, dones):
        # print("in q_traget_fun, next_state_dims is:", next_states.shape)
        next_states = orgenize_dims(next_states)
        # print("in q_traget_fun, next_state_dims after orgenize is:", next_states.shape)
        
        a = self.mu_Net_targ.forward(next_states) *2.0
        
        if len(self.action_clip) != 0:
             a = tf.clip_by_value(a, self.action_clip[0], self.action_clip[1])
             
        x = tf.concat((next_states, a), axis = 1)
        q_next = self.q_Net_targ.forward(x)
        
        q_targets = tf.stop_gradient(rewards + self.gamma * (1-dones) * q_next)
        
        return q_targets
    
    def get_q(self, states, a):
        
        a = np.expand_dims(a, axis = 1 )
        states = orgenize_dims(states)
        x = tf.concat((states, a), axis = 1)
        q = self.q_Net.forward(x)
        
        return q 
    
    ## or try teh first one
    def get_q2(self, states):
        states = orgenize_dims(states)
        a = self.mu_Net.forward(states)*2.0
        if len(self.action_clip) != 0:
             a = tf.clip_by_value(a, self.action_clip[0], self.action_clip[1])

        x = tf.concat((states, a), axis = 1)
        q = self.q_Net.forward(x)
        
        return q 
    
    def q_loss(self, states, next_states, actions, rewards, dones):
        q_targets = self.q_targets_fun(next_states = next_states,
                                       rewards = rewards, 
                                       dones = dones)
        
        q = self.get_q(states = states, a = actions) # option 1 
        # q = self.get_q2(states = states) # option 2 
        loss = tf.reduce_mean((q - q_targets)**2)
        
        return loss
    
    def mu_loss(self, states):
        # Here we are tying to maximize the Q(s, mu(s))
        states = orgenize_dims(states)
        a = self.mu_Net.forward(states) * 2.0
        
        if len(self.action_clip) != 0:
             a = tf.clip_by_value(a, self.action_clip[0], self.action_clip[1])
        
        # print(a.shape)
        # print(states.shape)
        x = tf.concat((states, a), axis = 1)
        q_mu = self.q_Net.forward(x)
        
        loss = - tf.reduce_mean(q_mu) # The minus for minimization!! 
        
        return loss 
        
        
    def train_q_net(self,states, next_states, actions, rewards, dones):
        with tf.GradientTape(watch_accessed_variables = False) as tape:
            tape.watch(self.q_Net.trainable_params)
            loss_q = self.q_loss( states, next_states, actions, rewards, dones)
        
        gradients = tape.gradient(loss_q, self.q_Net.trainable_params)
        
        self.q_optimizer.apply_gradients(zip(gradients, self.q_Net.trainable_params))
        
        return loss_q
    
    def train_mu_net(self, states):
        
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.mu_Net.trainable_params)
            loss_mu = self.mu_loss(states)
            
        gradients = tape.gradient(loss_mu, self.mu_Net.trainable_params)
        self.mu_optimizer.apply_gradients(zip(gradients, self.mu_Net.trainable_params))
        
        return loss_mu 
    
    
    ## Update target netwok
    
    def update_targets_nets(self):
        
        for w_mu, w_mu_target in zip(self.mu_Net.trainable_params, self.mu_Net_targ.trainable_params):
            smoothed_w = self.decay * w_mu_target + (1-self.decay) * w_mu
            w_mu_target.assign(smoothed_w)
            
        for w_q, w_q_target in zip(self.q_Net.trainable_params, self.q_Net_targ.trainable_params):
            smoothed_w = self.decay * w_q_target + (1-self.decay) * w_q
        
        
   
    def train(self):
        s, a,r, s_tag, dones = self.replay_memory.get_minibatch()
        
        q_loss = self.train_q_net(states = s,
                                  next_states = s_tag,
                                  actions=a,
                                  rewards=r,
                                  dones=dones)
        
        mu_loss = self.train_mu_net(states = s)
        
        self.update_targets_nets()
        
        return q_loss, mu_loss
            
        
    
    
        
    
    
    
    
        


