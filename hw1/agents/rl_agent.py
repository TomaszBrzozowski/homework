import tensorflow as tf
import numpy as np
import gym
import load_policy
import tf_util
import pickle
import util

class RlAgent:
    def __init__(self,output_dir=None):
        self.experience = None
        self.policy_fn = None
        self.output_dir = output_dir
        self.model = self.get_model()
        self.timestep = None
        self.episode = None
    def gather_experience(self,env_name,expert_policy_filename=None,render=None,max_timesteps=None,num_rollouts=None,output_dir=None):
        raise NotImplementedError
    def save_experience(self,output_dir,filename):
        raise NotImplementedError
    def save_model(self):
        raise NotImplementedError
    def restore_model(self):
        raise NotImplementedError
    def get_model(self):
        raise NotImplementedError

class ExpertAgent(RlAgent):
    def gather_experience(self,env_name,expert_policy_filename=None,render=None,max_timesteps=None,num_rollouts=None,output_dir=None):
        # always give this output dir check at gather_experience
        if output_dir is not None:
            self.output_dir=output_dir
        print('loading and building expert policy')
        self.policy_fn = self.load_policy(expert_policy_filename)
        print('loaded and built')

        with tf.Session():
            tf_util.initialize()

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
                    action = self.policy_fn(obs[None, :])
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
            if self.output_dir is not 'None':
                self.save_experience(self.output_dir,'{}-{}_rollouts.pkl'.format(env_name, num_rollouts))
    def save_experience(self,output_dir,filename):
        util.mkdir_rel(output_dir)
        with open(output_dir + '/' + filename, 'wb') as f:
            pickle.dump(self.experience, f)
    def load_policy(self,expert_policy_filename):
        return load_policy.load_policy(expert_policy_filename)
    def save_model(self):
        pass
    def restore_model(self):
        pass
    def get_model(self):
        pass

class CloningAgent(RlAgent):
    def save_experience(self,output_dir,filename):
        pass
    def load_experience(self,expert_experience_filename):
        pass
    def gather_experience(self):
        pass
    def clone_expert_bahavior(self):
        pass

class BehaviorCloningAgent(CloningAgent):
    def gather_experience(self):
        pass
    def clone_expert_bahavior(self):
        pass

class DaggerAgent(CloningAgent):
    def gather_experience(self):
        pass
    def clone_expert_bahavior(self):
        pass