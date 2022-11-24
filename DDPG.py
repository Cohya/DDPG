import tensorflow as tf 
import numpy as np 
import os 
import pickle


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
    def __init__(self, mu_Net, mu_Net_targ, q_Net, q_Net_targ,  replay_memory, action_clip = [], 
                 gamma = 0.99,
                 decay = 0.995 ):
        
        #action_dims, observation_dims, args,
        std_dev = 0.2
        self.ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))
        self.replay_memory = replay_memory
        self.action_clip = action_clip
        
        # Define the orginal nets and their copies
        self.mu_Net = mu_Net
        self.mu_Net_targ = mu_Net_targ
        self.q_Net = q_Net
        self.q_Net_targ = q_Net_targ
        
        self.gamma = gamma
        self.decay = decay

        
        ## copy weights 
        copy_weights(self.q_Net, self.q_Net_targ)
        copy_weights(self.mu_Net, self.mu_Net_targ)
        
        
        self.mu_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
        self.q_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.002)
        
      # ""I think we should normelized the observation""  
    def get_action(self,state, noise_scale = 'OU'):
        # calculate mu, which suppose to be the best action  
        a = self.mu_Net.forward(state)  ## this is the action max of the pendulum 
        # print(state)
        # Add Gaussian noise (during training noise_scale = 0.1 or other value higher than 0)
        # a += tf.random.normal(shape = a.shape) * noise_scale
        
        ## USe OUAActionNOise 
        if noise_scale is not False:
            if type(noise_scale) == float:
                a += tf.random.normal(shape = a.shape) * noise_scale
                
            elif  noise_scale == 'OU':
                a += self.ou_noise()
        

        if len(self.action_clip) != 0:
            a = tf.clip_by_value(a, self.action_clip[0], self.action_clip[1])

        assert (a[0][0].numpy() <= 2.0),"Oh no! This assertion failed!" 
        assert (a[0][0].numpy() >= -2.0),"Oh no! This assertion failed!" 

        return a
    
    def q_targets_fun(self,next_states, rewards, dones):

        a = self.mu_Net_targ.forward(next_states) 
        
        if len(self.action_clip) != 0:
             a = tf.clip_by_value(a, self.action_clip[0], self.action_clip[1])
             
        # x = tf.concat((next_states, a), axis = 1)
        q_next = self.q_Net_targ.forward(next_states, a)
        bz = q_next.shape[0]
        q_targets = tf.stop_gradient(np.reshape(rewards, newshape=(bz,1))
                                     + self.gamma 
                                     * np.array((1-dones)).reshape(bz,1)
                                     * q_next)
        
        return q_targets
    
    def get_q(self, states, a):
        
        a = np.expand_dims(a, axis = 1 )
        q = self.q_Net.forward(states, a)
        
        return q 
    
    
    def q_loss(self, states, next_states, actions, rewards, dones):
        q_targets = self.q_targets_fun(next_states = next_states,
                                       rewards = rewards, 
                                       dones = dones)
        
        q = self.get_q(states = states, a = actions) # option 1 
        loss = tf.math.reduce_mean(tf.math.square(q - q_targets))
        
        return loss
    
    def mu_loss(self, states):
        # Here we are tying to maximize the Q(s, mu(s))
        a = self.mu_Net.forward(states) 
        
        if len(self.action_clip) != 0:
             a = tf.clip_by_value(a, self.action_clip[0], self.action_clip[1])

        q_mu = self.q_Net.forward(states, a)
        
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
            tape.watch(self.mu_Net.trainable_params) # to activate it set wathc to false
            loss_mu = self.mu_loss(states)
            
        gradients = tape.gradient(loss_mu, self.mu_Net.trainable_params)
        self.mu_optimizer.apply_gradients(zip(gradients, self.mu_Net.trainable_params))
        
        return loss_mu 
    
    
    ## Update target netwok
    def check_weights_distance(self,w_tar,w_org):
        vals = []
        
        for w, w2 in zip(w_tar, w_org):
            val =  np.mean(w - w2)
            vals.append(val)
            
        return np.mean(vals)
    
    def update_targets_nets(self):
        
        for w_mu, w_mu_target in zip(self.mu_Net.trainable_params, self.mu_Net_targ.trainable_params):
            smoothed_w = self.decay * w_mu_target.numpy() + (1-self.decay) * w_mu.numpy()
            w_mu_target.assign(smoothed_w)

        for w_q, w_q_target in zip(self.q_Net.trainable_params, self.q_Net_targ.trainable_params):
            smoothed_w = self.decay * w_q_target.numpy() + (1-self.decay) * w_q.numpy()
            w_q_target.assign(smoothed_w)
            
        
   
    def train(self):
        s, a,r, s_tag, dones = self.replay_memory.get_minibatch()
        s = orgenize_dims(s)
        s_tag = orgenize_dims(s_tag)
        q_loss = self.train_q_net(states = s,
                                  next_states = s_tag,
                                  actions=a,
                                  rewards=r,
                                  dones=dones)
        
        mu_loss = self.train_mu_net(states = s)
        
        self.update_targets_nets()
        
        return q_loss, mu_loss
    
    def save_weghts(self):
        # save the actor (mu_Net)
        weights_mu_Nets = []
        weights_q_Net = []
        for w_mu, w_q in zip(self.mu_Net.trainable_params, self.q_Net.trainable_params):
            weights_mu_Nets.append(w_mu.numpy())
            weights_q_Net.append(w_q.numpy())
            
        
        
        if not os.path.isdir('weights'):
            print("Creating Weights folder")
            os.mkdir('weights')
            
        with open('weights/weights_mu_Net.pk', 'wb') as file:
            pickle.dump(weights_mu_Nets, file)
        
        with open('weights/Weights_q_Net.pk', 'wb') as file:
            pickle.dump(weights_q_Net, file)
            
        
        print("Weights saved successfully!")
        
    def load_weights(self, mu_Net_weights_file, q_Net_weights_file):
        
        with open(mu_Net_weights_file, 'rb') as file:
            mu_net_weights = pickle.load(file)
            
        with open(q_Net_weights_file, 'rb') as file:
            q_Net_weights = pickle.load(file)
            
        
        for w, wc in zip(self.mu_Net.trainable_params, mu_net_weights):
            w.assign(wc)
            
        for w, wc in zip(self.q_Net.trainable_params, q_Net_weights):
            w.assign(wc)
            
        
        
        ## copy weights 
        copy_weights(self.q_Net, self.q_Net_targ)
        copy_weights(self.mu_Net, self.mu_Net_targ)
        
        print("Weights were load successfully!")
            
        
        
        
            
class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)
              
    
        
    
    
    
    
        


