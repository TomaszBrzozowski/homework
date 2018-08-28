import sys

import tensorflow as tf
import numpy as np
import gym
import load_policy
import tf_util
import pickle
import os
from gym import wrappers
from sklearn.model_selection import train_test_split
from models.model import Model
from util import mkdir_rel,pbar
from math import ceil


class RlAgent:
    def __init__(self,output_dir=os.getcwd()):
        self.experience = None
        self.output_dir = output_dir
        self.model = None
        self.epoch = None
        self.timestep = None
        self.episode = None

    def gather_experience(self,env,num_rollouts,max_steps=None,render=False,progbar=False,cout=False,**kwargs):
        max_steps = max_steps or env.spec.timestep_limit
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
                action = self.policy(obs[None, :],**kwargs)
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
    def policy(self,observations,**kwargs):
        raise NotImplementedError
    def get_model(self,**kwargs):
        raise NotImplementedError
    def save_model(self,sess):
        raise NotImplementedError
    def restore_model(self,sess):
        raise NotImplementedError


class ExpertAgent(RlAgent):
    def __init__(self,output_dir=None,expert_policy_filename=None):
        super().__init__(output_dir=output_dir)
        if expert_policy_filename is not None: self.load_policy(expert_policy_filename)

    def save_experience(self,filename,output_dir=None):
        if output_dir is not None:
            self.output_dir=output_dir
        mkdir_rel(self.output_dir)
        with open(self.output_dir + '/' + filename, 'wb') as f:
            pickle.dump(self.experience, f)

    def policy(self,observations,**kwargs):
        return self.expert_policy_fn(observations)

    def get_model(self,**kwargs): pass

    def load_policy(self,expert_policy_filename):
        self.expert_policy_fn = load_policy.load_policy(expert_policy_filename)

    def generate_experience(self,env_name,expert_policy_filename,num_rollouts,render=False,max_timesteps=None,
                            output_dir=None,cout=False,record=False):
        if output_dir is not None:
            self.output_dir=output_dir

        env = gym.make(env_name)
        if record: env = wrappers.Monitor(env,self.output_dir,force=True,mode='evaluation')
        with tf.Session():
            tf_util.initialize()
            self.load_policy(expert_policy_filename)
            print('Env = {} | '.format(env_name))
            self.experience = self.gather_experience(env,num_rollouts,max_timesteps,render,progbar=(not cout),cout=cout)
            print('mean return:', self.experience['mean'], 'std of return:', self.experience['std'])
        self.save_experience('{}-{}_rollouts.pkl'.format(env_name, num_rollouts))


class CloningAgent(RlAgent):
    def save_experience(self,filename,output_dir=None):
        if output_dir is not None:
            self.output_dir=output_dir
        mkdir_rel(output_dir)
        with open(output_dir + '/' + filename) as f:
            f.write(self.experience['mean']+''+self.experience['std'])

    def get_model(self,x_train,obs_dim,actions_dim,batch_size,layers,opt,mod_out_dir,restore=True,sess=None,env_name=None,y=None,
                  lr=None,lr_sched=None,n_batches=None,lr_show=False,**kwargs):
        if lr:
            mod_out_dir+='_lr={}'.format(lr)
        self.model = Model(x_train,obs_dim,actions_dim,batch_size,layers,opt,mod_out_dir)
        ep, st = None, None
        if restore:
            ep, st =  self.model.restore(sess)
        print("Env = {} | {} | Current epoch = {} | timestep = {}".format(env_name,self.__class__.__name__,ep,st))
        lr = self.model.init_model(lr,sess,x_train,y,n_batches,lr_show,**kwargs)
        if ep is None:
            if not lr:
                self.model.checkpoint_dir=mod_out_dir+'_lr={}'.format(lr)
            sess.run(tf.global_variables_initializer())
        return (ep,st,mod_out_dir) if ep is not None else (0,0,mod_out_dir)

    def policy(self,observations,**kwargs):
        return self.model.predict(observations,**kwargs)

    def save_model(self,sess):
        self.model.save(sess)

    def restore_model(self,sess):
        self.model.restore(sess)

    def load_experience(self,expert_experience_filename):
        with open(expert_experience_filename, 'rb') as f:
            return pickle.loads(f.read())

    def clone_expert(self,env_name,expert_experience_filename,n_epochs=10,mod_out_dir=None,lr=None,batch_size=64,restore=True,layers=[64],net_name=None
                     ,keep_prob=1.0,opt='adam',lr_show=False,lr_sched='const',lr_exp_base=100,lr_exp_decay=0.99,lr_maxmin = 10,lr_c_t_ratio = 9,
                     num_test_rollouts=10,max_steps=None,progbar=False,render=None,cout=False,record=True,save_exp=False):
        tf.reset_default_graph()
        if net_name == None: net_name = '-'.join(str(e) for e in layers)
        if mod_out_dir == None:
            mod_out_dir = self.output_dir
        env = envw =  gym.make(env_name)
        model_params = "nn={}_kp={}_lr={}_lrs={}".\
            format(net_name,keep_prob,lr if lr else 'find',lr_sched)
        mod_out_dir = os.path.join(mod_out_dir,model_params)

        expert_experience = self.load_experience(expert_experience_filename)
        X,y = expert_experience['observations'], expert_experience['actions']
        if X.shape[0]==1: X = np.squeeze(X,axis=(0,))
        y = np.reshape(y,(y.shape[0],y.shape[1]*y.shape[2]))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
        n_batches = ceil(X_train.shape[0]/float(batch_size))

        with tf.Session() as sess:
            self.epoch, self.timestep, out_dir = self.get_model(X_train, X_train.shape[1], y_train.shape[1],batch_size,
                                                                layers,opt,mod_out_dir,restore,sess,env_name,y_train,
                                                                lr,lr_sched,n_batches,lr_show,keep_prob=keep_prob)
            if self.epoch<n_epochs:
                writer = tf.summary.FileWriter(out_dir, sess.graph)
            if progbar: pb = pbar(n_epochs)
            if record: envw = wrappers.Monitor(env,self.output_dir,force=True,mode='evaluation')

            best_val_loss=sys.float_info.max
            while self.epoch<n_epochs:
                if progbar: pb.update()
                self.timestep, t_loss = self.model.run_model(sess,writer,self.epoch,self.timestep,X_train,y_train,'train',
                                                             keep_prob,lr_sched,lr_exp_base,lr_exp_decay,lr_maxmin,lr_c_t_ratio,n_batches,n_epochs)
                self.timestep, v_loss = self.model.run_model(sess, writer, self.epoch, self.timestep,X_test,y_test,'eval')
                imit_exp = self.gather_experience(env,1,max_steps,render,progbar=False,sess=sess)
                X_train,y_train = self.dagger(X_train,y_train,imit_exp,env_name)
                print("Epoch {0}/{1}: Train_loss = {2:.6f} | Val_loss = {3:.6f} | Reward = {4:.5f}"
                      .format(self.epoch,n_epochs-1,t_loss,v_loss,imit_exp['mean']))
                self.epoch+=1
                if v_loss<best_val_loss:
                    self.save_model(sess)

            self.experience = self.gather_experience(envw,num_test_rollouts,max_steps,render,progbar=(not cout),cout=cout,sess=sess)
            if save_exp:
                self.save_experience("env={}_model-param={}.txt".format(env_name,mod_out_dir),self.output_dir)
        if self.epoch<n_epochs: writer.close()
        return expert_experience['mean'], expert_experience['std'],self.experience['mean'], self.experience['std'], \
               self.epoch, self.timestep, out_dir.split('\\')[-1]

    def dagger(self,X_train=None,y_train=None,imit_exp=None,env_name=None):
        return X_train, y_train


class DaggerAgent(CloningAgent):
    def dagger(self,X_train=None,y_train=None,imit_exp=None,env_name=None):
        exp_policy_file = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'experts',env_name+'.pkl'))
        X = imit_exp['observations']
        if X.shape[0]==1: X = np.squeeze(X,axis=(0,))
        dagg_actions = ExpertAgent(expert_policy_filename=exp_policy_file).policy(X)
        X_train = np.concatenate((X_train,X),axis=0),
        y_train = np.concatenate((y_train,dagg_actions),axis=0)
        return np.squeeze(X_train,axis=0), y_train