import tensorflow as tf
import numpy as np
import gym
import load_policy
import tf_util
import pickle
import os

from util import mkdir_rel

class RlAgent:
    def __init__(self,output_dir=os.getcwd()):
        self.experience = None
        self.output_dir = output_dir
        self.model = None
        self.timestep = None
        self.episode = None
    def gather_experience(self,env_name,num_rollouts,max_timesteps=None,render=False,sess=None):
        env = gym.make(env_name)
        max_steps = max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                # action = self.policy_fn(obs[None, :])
                action = self.policy(obs[None, :],sess)
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if render:
                    env.render()
                if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        mean_returns, std_returns = np.mean(returns), np.std(returns)
        print('returns', returns)
        print('mean return', mean_returns)
        print('std of return', std_returns)

        self.experience = {'observations': np.array(observations), 'actions': np.array(actions), 'mean': mean_returns,
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
    def generate_experience(self,env_name,expert_policy_filename,num_rollouts,render=False,max_timesteps=None,output_dir=None):
        if output_dir is not None:
            self.output_dir=output_dir

        with tf.Session():
            tf_util.initialize()
            self.load_policy(expert_policy_filename)
            self.gather_experience(env_name,num_rollouts,max_timesteps,render)
        self.save_experience('{}-{}_rollouts.pkl'.format(env_name, num_rollouts))


class CloningAgent(RlAgent):
    def save_experience(self,filename,output_dir=None):
        pass
    def load_experience(self,expert_experience_filename):
        pass
    def clone_expert_bahavior(self):
        pass

class BehaviorCloningAgent(CloningAgent):
    def clone_expert_bahavior(self):
        pass

class DaggerAgent(CloningAgent):
    def clone_expert_bahavior(self):
        pass