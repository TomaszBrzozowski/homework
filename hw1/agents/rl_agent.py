import tensorflow as tf
import numpy as np
import gym
from gym import wrappers

import load_policy
import tf_util
import pickle
import os

from sklearn.model_selection import train_test_split
from models.model import Model
from util import mkdir_rel,pbar


class RlAgent:
    def __init__(self,output_dir=os.getcwd()):
        self.experience = None
        self.output_dir = output_dir
        self.model = None
        self.timestep = None
        self.episode = None

    def gather_experience(self,env,num_rollouts,max_steps=None,render=False,sess=None,progbar=False,cout=False):
        returns = []
        observations = []
        actions = []
        if (progbar or cout): print('Running {} roll-outs (num_rollouts = {})...'.format(self.__class__.__name__,num_rollouts))
        if progbar: pb = pbar(num_rollouts)
        for i in (range(num_rollouts)):
            if cout: print('Roll-out nr: ',i)
            obs = env.reset()
            totalr = 0.
            steps = 0
            for _ in (range(max_steps)):
                # action = self.policy_fn(obs[None, :])
                action = self.policy(obs[None, :],sess)
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if render:
                    env.render()
                if cout and (steps % 100) == 0: print("    steps %i/%i"%(steps, max_steps))
                if done:
                    break
            returns.append(totalr)
            if progbar: pb.update()
        mean_returns, std_returns = np.mean(returns), np.std(returns)
        return {'observations': np.array(observations), 'actions': np.array(actions), 'mean': mean_returns,
                       'std': std_returns}
    def save_experience(self,filename,output_dir=None):
        raise NotImplementedError
    def policy(self,observations,sess=None):
        raise NotImplementedError
    def get_model(self,x_train=None, observations_dim=None, actions_dim=None, checkpoint_dir=None, learning_rate=None, batch_size=None,sess=None):
        raise NotImplementedError
    def save_model(self,sess):
        raise NotImplementedError
    def restore_model(self,sess):
        raise NotImplementedError

class ExpertAgent(RlAgent):
    def __init__(self,output_dir=None):
        super().__init__(output_dir=output_dir)
        self.expert_policy_fn=None
    def save_experience(self,filename,output_dir=None):
        if output_dir is not None:
            self.output_dir=output_dir
        mkdir_rel(self.output_dir)
        with open(self.output_dir + '/' + filename, 'wb') as f:
            pickle.dump(self.experience, f)
    def policy(self,observations,sess=None):
        return self.expert_policy_fn(observations)
    def get_model(self, x_train=None, observations_dim=None, actions_dim=None, checkpoint_dir=None, learning_rate=None,
                  batch_size=None, sess=None):
        pass
    def load_policy(self,expert_policy_filename):
        self.expert_policy_fn = load_policy.load_policy(expert_policy_filename)
    def generate_experience(self,env_name,expert_policy_filename,num_rollouts,render=False,max_timesteps=None,output_dir=None,cout=False,record=False):
        if output_dir is not None:
            self.output_dir=output_dir

        env = gym.make(env_name)
        if record: env = wrappers.Monitor(env,self.output_dir,force=True,mode='evaluation')
        with tf.Session():
            tf_util.initialize()
            self.load_policy(expert_policy_filename)
            self.experience = self.gather_experience(env,num_rollouts,max_timesteps,render,progbar=(not cout),cout=cout)
            print('mean return:', self.experience['mean'], 'std of return:', self.experience['std'])
        self.save_experience('{}-{}_rollouts.pkl'.format(env_name, num_rollouts))

class ImitationAgent(RlAgent):
    def save_experience(self,filename,output_dir=None):
        if output_dir is not None:
            self.output_dir=output_dir
        mkdir_rel(output_dir)
        with open(output_dir + '/' + filename) as f:
            f.write(self.experience['mean']+''+self.experience['std'])
    def get_model(self,x_train=None, observations_dim=None, actions_dim=None, checkpoint_dir=None, learning_rate=None, batch_size=None,sess=None,restore=True):
        self.model = Model(x_train, observations_dim, actions_dim, checkpoint_dir, learning_rate, batch_size)
        if restore:
            return self.model.restore(sess)
        return 0,0
    def policy(self,observations,sess=None):
        return self.model.predict(observations,sess)
    def save_model(self,sess):
        self.model.save(sess)
    def restore_model(self,sess):
        self.model.restore(sess)
    def load_experience(self,expert_experience_filename):
        with open(expert_experience_filename, 'rb') as f:
            return pickle.loads(f.read())
    def clone_expert(self,env_name,expert_experience_filename,render=None,max_timesteps=None,num_rollouts=None,output_dir=None):
        raise NotImplementedError

class CloningAgent(ImitationAgent):
    def clone_expert(self,env_name,expert_experience_filename,model_output_dir=None,learning_rate=0.001,drop_prob=0.75,batch_size=64,num_epochs=10,
                     num_test_rollouts=10,restore=True,render=None,max_timesteps=None,save_exp=False, progbar=False, cout=False,record=True):
        if model_output_dir == None:
            model_output_dir = self.output_dir
        env = gym.make(env_name)
        max_steps = max_timesteps or env.spec.timestep_limit

        expert_experience = self.load_experience(expert_experience_filename)
        X,y = expert_experience['observations'], expert_experience['actions']
        y = np.reshape(y,(y.shape[0],y.shape[1]*y.shape[2]))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

        tf.reset_default_graph()
        with tf.Session() as sess:
            self.epoch, self.timestep = self.get_model(X_train, X_train.shape[1], y_train.shape[1], model_output_dir, learning_rate, batch_size, sess, restore)
            print("Env = {} | Current epoch = {} | timestep = {}".format(env_name,self.epoch, self.timestep))
            if self.epoch<num_epochs:
                writer = tf.summary.FileWriter(model_output_dir, sess.graph)
            tf_util.initialize()
            if progbar: pb = pbar(num_epochs)
            if record: envw = wrappers.Monitor(env,self.output_dir,force=True,mode='evaluation')
            while self.epoch<num_epochs:
                if progbar: pb.update()
                self.timestep, t_loss = self.model.run_model(sess, writer, self.epoch, self.timestep,X_train,y_train,'train',drop_prob)
                self.timestep, v_loss = self.model.run_model(sess, writer, self.epoch, self.timestep,X_test,y_test,'eval')
                reward_curr = self.gather_experience(env,1,max_steps,render,sess,progbar=False)['mean']
                print("Epoch {0}/{1}: Train_loss = {2:.6f} | Val_loss = {3:.6f} | Reward = {4:.5f}".format(self.epoch,num_epochs-1,t_loss,v_loss,reward_curr))
                self.epoch+=1
            self.experience = self.gather_experience(envw,num_test_rollouts,max_steps,render,sess,progbar=(not cout),cout=cout)
            if save_exp:
                self.save_experience("{}-{}_rollouts.txt".format(env_name, num_test_rollouts), self.output_dir)
        try: writer.close()
        except: pass
        return self.epoch,self.experience['mean'], self.experience['std'], expert_experience['mean'], expert_experience['std']

class DaggerAgent(ImitationAgent):
    def clone_expert(self,env_name,expert_experience_filename,render=None,max_timesteps=None,num_rollouts=None,output_dir=None):
        pass