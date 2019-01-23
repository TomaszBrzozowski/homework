import os
import tensorflow as tf
import numpy as np

C = 1e-13

# Predefined function to build a feedforward neural network
def build_mlp(input_placeholder, 
              output_size,
              scope, 
              n_layers=2, 
              size=500, 
              activation=tf.tanh,
              output_activation=None
              ):
    out = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            out = tf.layers.dense(out, size, activation=activation)
        out = tf.layers.dense(out, output_size, activation=output_activation)
    return out

class NNDynamicsModel():
    def __init__(self, 
                 env, 
                 n_layers,
                 size, 
                 activation, 
                 output_activation, 
                 normalization,
                 batch_size,
                 iterations,
                 learning_rate,
                 sess
                 ):
        """ Note: Be careful about normalization """
        self.mean_obs,self.std_obs,self.mean_deltas,self.std_deltas,self.mean_actions,self.std_actions = normalization
        self.obs_dim = env.observation_space.shape[0]
        self.actions_dim = env.action_space.shape[0]
        self.in_states_acts= tf.placeholder(tf.float32,[None,self.obs_dim + self.actions_dim],name='states_actions')
        self.out_states_deltas = tf.placeholder(tf.float32,[None,self.obs_dim],name='states_deltas')
        self.epochs = iterations
        self.gstep = tf.Variable(0, dtype=tf.int32,trainable=False, name='global_step')
        self.pred_delt = build_mlp(self.in_states_acts,self.obs_dim,"pred_state_delta",n_layers,size,activation,output_activation)
        self.batch_size = batch_size
        self.lr = learning_rate
        self.loss = tf.losses.mean_squared_error(self.out_states_deltas,self.pred_delt)
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.sess=sess

    def fit(self, data):
        """
        a function to take in a dataset of (unnormalized)states, (unnormalized)actions, (unnormalized)next_states and fit the dynamics model going from normalized states, normalized actions to normalized state differences (s_t+1 - s_t)
        """
        obs = np.vstack([path['observations'] for path in data])
        actions = np.vstack([path['actions'] for path in data])
        next_obs = np.vstack([path['next_observations'] for path in data])

        norm_obs = (obs - self.mean_obs) / (self.std_obs + C)
        norm_actions = (actions - self.mean_actions) / (self.std_actions + C)
        norm_delta = (next_obs - self.mean_deltas) / (self.std_deltas + C)
        obs_actions = np.vstack((norm_obs,norm_actions))

        n_batches = obs.shape[0]//self.batch_size+1

        for ep in range(self.epochs):
            perm_ids = np.random.choice(obs.shape[0])
            tl=0.

            for st in range(n_batches):
                start_id = st*self.batch_size
                perms_ids_batch = perm_ids[start_id:start_id+self.batch_size]
                in_batch = obs_actions[perms_ids_batch:]
                out_batch = norm_delta[perms_ids_batch:]
                l,_ = self.sess.run([self.loss,self.opt],feed_dict={self.in_states_acts:in_batch,self.out_states_deltas:out_batch})
                tl+=l

            print("Epoch {0}/{1}: Train_loss = {2:.6f}".format(ep,self.epochs,tl/n_batches))

    def predict(self, states, actions):
        """ a function to take in a batch of (unnormalized) states and (unnormalized) actions and return the (unnormalized) next states as predicted by using the model """

        norm_obs = (states - self.mean_obs) / (self.std_obs + C)
        norm_actions = (actions - self.mean_actions) / (self.std_actions + C)
        obs_actions = np.vstack((norm_obs,norm_actions))

        pred_states_deltas = self.sess.run([self.pred_delt],feed_dict={self.in_states_acts: obs_actions})

        unnormalized =  states + self.mean_deltas + pred_states_deltas*self.std_deltas
        return unnormalized
