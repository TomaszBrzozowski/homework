import numpy as np
import tensorflow as tf
import gym
import logz
import scipy.signal
import os
import time
import inspect
from gym import wrappers
from pathos.multiprocessing import ProcessingPool
from concurrent.futures import ThreadPoolExecutor,as_completed

CONST = 1e-13
#============================================================================================#
# Utilities
#============================================================================================#

def build_mlp(
        input_placeholder,
        output_size,
        scope,
        n_layers=2,
        size=64,
        activation=tf.tanh,
        output_activation=None
        ):
    #========================================================================================#
    #                           ----------SECTION 3----------
    # Network building
    #
    # Your code should make a feedforward neural network (also called a multilayer perceptron)
    # with 'n_layers' hidden layers of size 'size' units.
    #
    # The output layer should have size 'output_size' and activation 'output_activation'.
    #
    # Hint: use tf.layers.dense
    #========================================================================================#
    with tf.variable_scope(scope):
        out = input_placeholder
        for i in range(n_layers):
            out=tf.layers.dense(out,size,activation,kernel_initializer=tf.contrib.layers.xavier_initializer(),name='l_h_%d'%i)
        out = tf.layers.dense(out,output_size,output_activation,kernel_initializer=tf.contrib.layers.xavier_initializer(),name='l_out_%d'%i)
        return out

def pathlength(path):
    return len(path["reward"])

class EnvList(gym.Env):
    def __init__(self, env_name, n_envs, logdir=None, record=False):
        self.envs = [gym.make(env_name) for _ in range(n_envs)]
        if record and logdir:
            self.envs = [wrappers.Monitor(self.envs[i],logdir,force=True,mode='evaluation') for i in self.envs]
        self.action_space = self.envs[0].action_space.n if self.discrete() else self.envs[0].action_space.shape[0]
        self.observation_space = self.envs[0].observation_space.shape[0]
    def discrete(self):
        return isinstance(self.envs[0].action_space,gym.spaces.Discrete)
    def reset(self,i=0):
        return self.envs[i].reset()
    def render(self,i=0,**kwargs):
        return self.envs[i].render(**kwargs)
    def step(self, action,i=0):
        return self.envs[i].step(action)

class PathCollector():
    def __init__(self,sess,symbolic_ac,symbolic_ob,max_path_length,):
        self.sess = sess
        self.symbolic_ac = symbolic_ac
        self.symbolic_ob = symbolic_ob
        self.max_path_length = max_path_length
    def __call__(self,env,animate=False):
        ob = env.reset()
        obs,acs,rewards = [],[],[]
        steps = 0
        while (True):
            if animate:
                env.render()
                time.sleep(0.05)
            obs.append(ob)
            ac = self.sess.run(self.symbolic_ac,feed_dict={self.symbolic_ob: ob[None]})
            ac = ac[0]
            acs.append(ac)
            ob,rew,done,_ = env.step(ac)
            rewards.append(rew)
            steps += 1
            if done or steps > self.max_path_length:
                break
        path = {"observation": np.array(obs),"reward": np.array(rewards),"action": np.array(acs)}
        env.close()
        return path

#============================================================================================#
# Policy Gradient
#============================================================================================#

def train_PG(exp_name='',
             env_name='CartPole-v0',
             n_iter=100,
             gamma=1.0,
             min_timesteps_per_batch=1000,
             max_path_length=None,
             learning_rate=5e-3,
             reward_to_go=True,
             animate=True,
             logdir=None,
             normalize_advantages=True,
             nn_baseline=False,
             seed=0,
             gae=True,
             lambd=1.0,
             threads=1,
             max_threads_pool=16,
             thread_timeout=None,
             record=False,
             # network arguments
             n_layers=1,
             size=32,
             ):

    def n_threads_to_run(timesteps_this_batch):
        tsteps_left = min_timesteps_per_batch - timesteps_this_batch
        max_threads = int(np.ceil((tsteps_left) / max_path_length))
        if threads < 1 or threads>max_threads:
            return max_threads
        else:
            return threads

    start = time.time()

    # Configure output directory for logging
    logz.configure_output_dir(logdir)

    # Log experimental parameters
    args = inspect.getargspec(train_PG)[0]
    # args = inspect.signature(train_PG).parameters
    locals_ = locals()
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_params(params)

    # Set random seeds
    tf.set_random_seed(seed)
    np.random.seed(seed)

    # Maximum length for episodes
    max_path_length = max_path_length or gym.make(env_name).spec.max_episode_steps

    # Make the gym environment
    env = EnvList(env_name,n_threads_to_run(0),logdir,record)

    # Is this env continuous, or discrete?
    discrete = env.discrete()

    #========================================================================================#
    # Notes on notation:
    #
    # Symbolic variables have the prefix sy_, to distinguish them from the numerical values
    # that are computed later in the function
    #
    # Prefixes and suffixes:
    # ob - observation
    # ac - action
    # _no - this tensor should have shape (batch size /n/, observation dim)
    # _na - this tensor should have shape (batch size /n/, action dim)
    # _n  - this tensor should have shape (batch size /n/)
    #
    # Note: batch size /n/ is defined at runtime, and until then, the shape for that axis
    # is None
    #========================================================================================#

    # Observation and action sizes
    ob_dim = env.observation_space
    ac_dim = env.action_space

    #========================================================================================#
    #                           ----------SECTION 4----------
    # Placeholders
    #
    # Need these for batch observations / actions / advantages in policy gradient loss function.
    #========================================================================================#

    sy_ob_no = tf.placeholder(shape=[None, ob_dim], name="ob", dtype=tf.float32)
    if discrete:
        sy_ac_na = tf.placeholder(shape=[None], name="ac", dtype=tf.int32)
    else:
        sy_ac_na = tf.placeholder(shape=[None, ac_dim], name="ac", dtype=tf.float32)

    # Define a placeholder for advantages
    sy_adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32)

    #========================================================================================#
    #                           ----------SECTION 4----------
    # Networks
    #
    # Make symbolic operations for
    #   1. Policy network outputs which describe the policy distribution.
    #       a. For the discrete case, just logits for each action.
    #
    #       b. For the continuous case, the mean / log std of a Gaussian distribution over
    #          actions.
    #
    #      Hint: use the 'build_mlp' function you defined in utilities.
    #
    #      Note: these ops should be functions of the placeholder 'sy_ob_no'
    #
    #   2. Producing samples stochastically from the policy distribution.
    #       a. For the discrete case, an op that takes in logits and produces actions.
    #
    #          Should have shape [None]
    #
    #       b. For the continuous case, use the reparameterization trick:
    #          The output from a Gaussian distribution with mean 'mu' and std 'sigma' is
    #
    #               mu + sigma * z,         z ~ N(0, I)
    #
    #          This reduces the problem to just sampling z. (Hint: use tf.random_normal!)
    #
    #          Should have shape [None, ac_dim]
    #
    #      Note: these ops should be functions of the policy network output ops.
    #
    #   3. Computing the log probability of a set of actions that were actually taken,
    #      according to the policy.
    #
    #      Note: these ops should be functions of the placeholder 'sy_ac_na', and the
    #      policy network output ops.
    #
    #========================================================================================#

    if discrete:
        sy_logits_na = build_mlp(sy_ob_no,ac_dim,'disc_policy',n_layers,size)
        sy_sampled_ac = tf.squeeze(tf.multinomial(sy_logits_na,1),axis=1)
        sy_logprob_n = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sy_ac_na,logits=sy_logits_na)

    else:
        sy_mean = build_mlp(sy_ob_no,ac_dim,'cont_policy',n_layers=n_layers,size=size)
        sy_logstd = tf.get_variable('logstd',shape=[ac_dim],dtype=np.float32)
        sy_std = tf.exp(sy_logstd)
        sy_sampled_ac = sy_mean + tf.multiply(tf.random_normal(shape=tf.shape(sy_mean)),sy_std)
        mvn = tf.contrib.distributions.MultivariateNormalDiag(loc=sy_mean,scale_diag=sy_std)
        sy_logprob_n = -mvn.log_prob(sy_ac_na)

    #========================================================================================#
    #                           ----------SECTION 4----------
    # Loss Function and Training Operation
    #========================================================================================#

    loss = tf.reduce_mean(tf.multiply(sy_logprob_n,sy_adv_n))
    update_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    #========================================================================================#
    #                           ----------SECTION 5----------
    # Optional Baseline
    #========================================================================================#

    if gae: nn_baseline = True
    if nn_baseline:
        baseline_prediction = tf.squeeze(build_mlp(
                                sy_ob_no,
                                1,
                                "nn_baseline",
                                n_layers=n_layers,
                                size=size))
        # Define placeholders for targets, a loss function and an update op for fitting a
        # neural network baseline. These will be used to fit the neural network baseline.
        sy_bl_target_n = tf.placeholder(shape=[None], name="bl_target", dtype=tf.float32)
        baseline_loss = tf.losses.mean_squared_error(sy_bl_target_n,baseline_prediction)
        baseline_update_op = tf.train.AdamOptimizer(learning_rate).minimize(baseline_loss)

    #========================================================================================#
    # Tensorflow Engineering: Config, Session, Variable initialization
    #========================================================================================#

    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)

    sess = tf.Session(config=tf_config)
    sess.__enter__() # equivalent to `with sess:`
    tf.global_variables_initializer().run() #pylint: disable=E1101

    #========================================================================================#
    # Training Loop
    #========================================================================================#

    total_timesteps = 0
    col = PathCollector(sess,sy_sampled_ac,sy_ob_no,max_path_length)

    for itr in range(n_iter):
        print("********** Iteration %i ************"%itr)

        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            n_threads = n_threads_to_run(timesteps_this_batch)
            if threads==1:
                path = col.__call__(env,animate=(animate and len(paths) == 0 and itr%10))
                paths.append(path)
            else:
                with ThreadPoolExecutor(max_threads_pool) as exec:
                    futures = [exec.submit(col.__call__,e) for e in env.envs[:n_threads]]
                    for future in as_completed(futures,timeout=thread_timeout):
                        paths.append(future.result())
            col_paths = paths[-n_threads:]
            timesteps_this_batch += sum([pathlength(path) for path in col_paths])
            if timesteps_this_batch >= min_timesteps_per_batch:
                break
        total_timesteps += timesteps_this_batch
        # Build arrays for observation, action for the policy gradient update by concatenating
        # across paths
        ob_no = np.concatenate([path["observation"] for path in paths if pathlength(path) >0])
        ac_na = np.concatenate([path["action"] for path in paths if pathlength(path) >0])

        #====================================================================================#
        #                           ----------SECTION 4----------
        # Computing Q-values
        #
        # Your code should construct numpy arrays for Q-values which will be used to compute
        # advantages (which will in turn be fed to the placeholder you defined above).
        #
        # Recall that the expression for the policy gradient PG is
        #
        #       PG = E_{tau} [sum_{t=0}^T grad log pi(a_t|s_t) * (Q_t - b_t )]
        #
        # where
        #
        #       tau=(s_0, a_0, ...) is a trajectory,
        #       Q_t is the Q-value at time t, Q^{pi}(s_t, a_t),
        #       and b_t is a baseline which may depend on s_t.
        #
        # You will write code for two cases, controlled by the flag 'reward_to_go':
        #
        #   Case 1: trajectory-based PG
        #
        #       (reward_to_go = False)
        #
        #       Instead of Q^{pi}(s_t, a_t), we use the total discounted reward summed over
        #       entire trajectory (regardless of which time step the Q-value should be for).
        #
        #       For this case, the policy gradient estimator is
        #
        #           E_{tau} [sum_{t=0}^T grad log pi(a_t|s_t) * Ret(tau)]
        #
        #       where
        #
        #           Ret(tau) = sum_{t'=0}^T gamma^t' r_{t'}.
        #
        #       Thus, you should compute
        #
        #           Q_t = Ret(tau)
        #
        #   Case 2: reward-to-go PG
        #
        #       (reward_to_go = True)
        #
        #       Here, you estimate Q^{pi}(s_t, a_t) by the discounted sum of rewards starting
        #       from time step t. Thus, you should compute
        #
        #           Q_t = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
        #
        #
        # Store the Q-values for all timesteps and all trajectories in a variable 'q_n',
        # like the 'ob_no' and 'ac_na' above.
        #
        #====================================================================================#

        q_ns =[]
        for path in paths:
            path_len = pathlength(path)
            rews = path['reward']
            discs = np.power(gamma,np.arange(path_len))
            if reward_to_go:
                qn = [np.sum(discs[:path_len-t]*rews[t:]) for t in range(path_len)]
            else:
                qn = np.sum(discs*rews)*np.ones(path_len)
            q_ns.append(qn)
        q_n = np.concatenate(q_ns)

        #====================================================================================#
        #                           ----------SECTION 5----------
        # Computing Baselines
        #====================================================================================#

        if nn_baseline:
            # If nn_baseline is True, use your neural network to predict reward-to-go
            # at each timestep for each trajectory, and save the result in a variable 'b_n'
            # like 'ob_no', 'ac_na', and 'q_n'.
            #
            # Hint #bl1: rescale the output from the nn_baseline to match the statistics
            # (mean and std) of the current or previous batch of Q-values. (Goes with Hint
            # #bl2 below.)

            b_n = np.array(sess.run(baseline_prediction,feed_dict={sy_ob_no: ob_no}))
            b_n = b_n * np.std(q_n) + np.mean(q_n)

            if gae:
                adv_ns = []
                for i,path in enumerate(paths):
                    path_len = pathlength(path)
                    rews = path['reward']
                    gamma_discs = np.power(gamma,np.arange(path_len))
                    gamma_lambda_discs = np.multiply(gamma_discs,np.power(lambd,np.arange(path_len)))
                    deltas = rews[:-1] + gamma * b_n[i + 1:i + path_len] - b_n[i:i + path_len - 1]
                    adv_n = [np.sum(gamma_lambda_discs[:path_len - 1 - t] * deltas[t:]) for t in
                             range(path_len - 1)] + [0]
                    adv_ns.append(adv_n)
                adv_n = np.concatenate(adv_ns)
                q_gae = np.array(adv_n + b_n)
            else:
                adv_n = q_n - b_n
        else:
            adv_n = q_n.copy()

        #====================================================================================#
        #                           ----------SECTION 4----------
        # Advantage Normalization
        #====================================================================================#

        if normalize_advantages:
            # On the next line, implement a trick which is known empirically to reduce variance
            # in policy gradient methods: normalize adv_n to have mean zero and std=1.
            adv_n = (adv_n - np.mean(adv_n)) / (np.std(adv_n) + CONST)

        #====================================================================================#
        #                           ----------SECTION 5----------
        # Optimizing Neural Network Baseline
        #====================================================================================#
        if nn_baseline:
            # ----------SECTION 5----------
            # If a neural network baseline is used, set up the targets and the inputs for the
            # baseline.
            #
            # Fit it to the current batch in order to use for the next iteration. Use the
            # baseline_update_op you defined earlier.
            #
            # Hint #bl2: Instead of trying to target raw Q-values directly, rescale the
            # targets to have mean zero and std=1. (Goes with Hint #bl1 above.)
            # experiment with different targets
            # q_n = (q_n - np.mean(q_n)) / (np.std(q_n) + CONST)
            q_n = (q_n - np.mean(q_gae)) / (np.std(q_gae) + CONST)
            # q_n = (q_gae - np.mean(q_gae)) / (np.std(q_gae) + CONST)
            # q_n = (q_gae - np.mean(q_n)) / (np.std(q_n) + CONST)
            # q_n = (b_n-np.mean(q_n))/(np.std(q_n)+CONST)
            # q_n = (b_n-np.mean(q_gae))/(np.std(q_gae)+CONST)
            _ = sess.run([baseline_update_op],feed_dict={sy_ob_no: ob_no,sy_bl_target_n: q_n})

        #====================================================================================#
        #                           ----------SECTION 4----------
        # Performing the Policy Update
        #====================================================================================#

        # Call the update operation necessary to perform the policy gradient update based on
        # the current batch of rollouts.
        #
        # For debug purposes, you may wish to save the value of the loss function before
        # and after an update, and then log them below.

        l,_ = sess.run([loss,update_op],feed_dict={sy_ob_no:ob_no,sy_ac_na:ac_na,sy_adv_n:adv_n})
        l_upd = sess.run(loss,feed_dict={sy_ob_no:ob_no,sy_ac_na:ac_na,sy_adv_n:adv_n})

        # Log diagnostics
        returns = [path["reward"].sum() for path in paths]
        ep_lengths = [pathlength(path) for path in paths]
        logz.log_tabular("Time", time.time() - start)
        logz.log_tabular("Iteration", itr)
        logz.log_tabular("AverageReturn", np.mean(returns))
        logz.log_tabular("StdReturn", np.std(returns))
        logz.log_tabular("MaxReturn", np.max(returns))
        logz.log_tabular("MinReturn", np.min(returns))
        logz.log_tabular("EpLenMean", np.mean(ep_lengths))
        logz.log_tabular("EpLenStd", np.std(ep_lengths))
        logz.log_tabular("TimestepsThisBatch", timesteps_this_batch)
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        logz.log_tabular("Loss", l)
        logz.log_tabular("Loss updated", l_upd)
        logz.dump_tabular(prec=8)
        logz.pickle_tf_vars()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--logdir', '-dir', type=str, default='data')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--nn_baseline', '-bl', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=1)
    parser.add_argument('--size', '-s', type=int, default=32)
    parser.add_argument('--gae','-gae',action='store_true')
    parser.add_argument('--lambd','-ld',type=float,default=1.0)
    parser.add_argument('--threads', '-th', type=int, default=1)
    parser.add_argument('--max_threads_pool', '-max_tp', type=int, default=16)
    parser.add_argument('--thread_timeout', '-th_to', type=int, default=None)
    parser.add_argument('--record', action='store_true')
    args = parser.parse_args()

    if not(os.path.exists(args.logdir)):
        os.makedirs(args.logdir)
    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(args.logdir, logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    max_path_length = args.ep_len if args.ep_len > 0 else None
    start = time.time()

    for e in range(args.n_experiments):
        seed = args.seed + 10*e
        print('Running experiment with seed %d'%seed)
        def train_func():
            train_PG(
                exp_name=args.exp_name,
                env_name=args.env_name,
                n_iter=args.n_iter,
                gamma=args.discount,
                min_timesteps_per_batch=args.batch_size,
                max_path_length=max_path_length,
                learning_rate=args.learning_rate,
                reward_to_go=args.reward_to_go,
                animate=args.render,
                logdir=os.path.join(logdir,'%d'%seed),
                normalize_advantages=not(args.dont_normalize_advantages),
                nn_baseline=args.nn_baseline,
                seed=seed,
                n_layers=args.n_layers,
                size=args.size,
                gae=args.gae,
                lambd=args.lambd,
                threads=args.threads,
                max_threads_pool=args.max_threads_pool,
                thread_timeout=args.thread_timeout,
                record=args.record
                )
        # Awkward hacky process runs, because Tensorflow does not like
        # repeatedly calling train_PG in the same thread.
        # p = Process(target=train_func)
        # p.start()
        # p.join()
        p = ProcessingPool(1)
        p.apipe(train_func).get()
        p.clear()
    print('All training took: {:.3f}s'.format(time.time()-start))

if __name__ == "__main__":
    main()