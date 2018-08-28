import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from util import mkdir_rel,param_sched,pbar
from tensorflow.contrib import slim
from math import ceil


class Model(object):
    def __init__(self,x_train,obs_dim,actions_dim,batch_size,layers,optimizer='adam',checkpoint_dir=None):
        self.mean_obs, self.var_obs = x_train.mean(axis=0), x_train.std(axis=0)
        self.observations_dim = obs_dim
        self.actions_dim = actions_dim
        self.input_obs = tf.placeholder(tf.float32,[None,obs_dim],name='observations')
        self.input_actions = tf.placeholder(tf.float32, [None, actions_dim], name='actions')
        self.keep_prob = tf.placeholder(tf.float32 , name='keep_probability')
        self.gstep = tf.Variable(0, dtype=tf.int32,trainable=False, name='global_step')
        self.epoch = tf.Variable(0, dtype=tf.int32,trainable=False, name='epoch')
        self.batch_size = batch_size
        self.input_lr = tf.placeholder(tf.float32,name='learning_rate')
        self.x, self.y = None, None
        self.iter = self.get_dataset_iterator()
        self.pred = self.get_model(layers)
        self.loss = self.get_loss()
        self.opt = self.get_optimizer(optimizer)
        self.checkpoint_dir = checkpoint_dir
        self.saver = tf.train.Saver()
        self.summ_train_loss, self.summ_val_loss = self.var_summary("training"), self.var_summary("validation")

    def get_dataset_iterator(self):
        with tf.name_scope('data'):
            dataset = tf.data.Dataset.from_tensor_slices((self.input_obs, self.input_actions)).batch(self.batch_size)
            iter = dataset.make_initializable_iterator()
            self.x, self.y = iter.get_next()
            self.x = tf.reshape(self.x, shape=[-1, self.observations_dim])
            self.y = tf.reshape(self.y, shape=[-1, self.actions_dim])
            return iter

    def save(self,sess):
        mkdir_rel(self.checkpoint_dir)
        self.saver.save(sess, self.checkpoint_dir +'/'+ 'model', global_step=self.gstep)

    def restore(self,sess):
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Restoring checkpoint: ",ckpt.model_checkpoint_path)
            self.saver.restore(sess, ckpt.model_checkpoint_path)
            return self.epoch.eval(), self.gstep.eval()
        return None, None

    def get_model(self,layer_sizes):
        obs_normalized = (self.x - self.mean_obs) / (self.var_obs+1e-8)
        layer_sizes = [int(s) if s.isdigit() else s for s in layer_sizes]
        fc_nr = 1
        nn = slim.fully_connected(obs_normalized,layer_sizes[0],scope='fc' + str(fc_nr),activation_fn=tf.nn.relu)
        for num_outputs in layer_sizes[1:]:
            if num_outputs != 'd':
                fc_nr+=1
                nn = slim.fully_connected(nn,num_outputs,scope='fc'+str(fc_nr),activation_fn=tf.nn.relu)
            else:
                nn = slim.dropout(nn,self.keep_prob)
        return slim.fully_connected(nn,self.actions_dim,scope='pred',activation_fn=None)

    def get_loss(self):
        with tf.name_scope('loss'):
            return tf.losses.mean_squared_error(self.y,self.pred)

    def get_optimizer(self,optimizer='adam'):
        if optimizer == "adam":
            return tf.train.AdamOptimizer(self.input_lr).minimize(self.loss,global_step=self.gstep)
        elif optimizer == "rmsprop":
            return tf.train.RMSPropOptimizer(self.input_lr).minimize(self.loss,global_step=self.gstep)
        elif optimizer == "adagrad":
            return tf.train.AdagradOptimizer(self.input_lr).minimize(self.loss,global_step=self.gstep)

    def train(self, lr, keep_prob, sess):
        l, summ_train_loss, _ = sess.run([self.loss, self.summ_train_loss, self.opt],feed_dict={self.keep_prob: keep_prob,
                                                                                                self.input_lr: lr})
        return l, summ_train_loss

    def evaluate(self, keep_prob,sess):
        return sess.run([self.loss, self.summ_val_loss],feed_dict={self.keep_prob: keep_prob})

    def predict(self, x, sess):
        y = np.zeros((x.shape[0],self.actions_dim),dtype=np.float32)
        sess.run(self.iter.initializer , feed_dict={self.input_obs: x, self.input_actions: y})
        return sess.run([self.pred], feed_dict={self.keep_prob: 1.0})

    def lr_find(self,n_batches,fn_train,lr_final=10.,lr_init=1e-6,beta=0.98,
                masm=None,masm_auto=None,sm=None,skip=0,show=False,lr_co=True,**kwargs):
        def choose_lr(lrs,losses,masm=masm,sm=sm,skip=skip):
            assert len(lrs)==len(losses), 'lrs/losses must be same length'
            if (masm_auto and masm >= len(lrs) / masm_auto) or masm is None:
                masm=ceil(len(lrs) / masm_auto)
            else:
                assert masm < len(lrs),'moving average smooth must be smaller than num of lrs/losses'
            lrs, losses = lrs[skip:], losses[skip:]
            mode = 'interp'
            if sm==None: sm=ceil(n_batches/2)//2*2+1
            if sm<=n_batches: mode = 'constant'
            derivatives = [0] * (masm + 1)
            for i in range(1 + masm,len(lrs)):
                derivatives.append((losses[i] - losses[i - masm]) / masm)
            smoothed_ders = savgol_filter(np.array(derivatives), sm, min(4,sm-1),mode=mode)
            wh = np.argmin(smoothed_ders)
            lr = lrs[wh]
            if lr_co: print(" Found lr = {0:.6f}".format(lr))
            if show:
                assert skip < len(lrs), 'skip must be smaller than num of train batches lrs/losses to show'
                plt.ylabel("d/loss")
                plt.xlabel("learning rate (log scale)")
                plt.plot(lrs[skip:],smoothed_ders[skip:],'b')
                plt.xscale('log')
                plt.plot(lrs[wh],smoothed_ders[wh],'ro',ms=8)
                plt.show()
            return lr
        mult = (lr_final / lr_init) ** (1 / n_batches)
        lr = lr_init
        avg_loss = 0.
        best_loss = 0.
        batch_num = 0
        losses = []
        lrs = []
        tot_train_loss=0.
        try:
            if lr_co:
                print('Finding lr...')
                pb = pbar(n_batches)
            while True:
                loss = fn_train(lr,**kwargs)[0]
                batch_num += 1
                avg_loss = beta * avg_loss + (1 - beta) * loss
                smoothed_loss = avg_loss / (1 - beta ** batch_num)
                if batch_num > 1 and smoothed_loss > 4 * best_loss:
                    return choose_lr(lrs,losses), batch_num, tot_train_loss
                if smoothed_loss < best_loss or batch_num == 1:
                    best_loss = smoothed_loss
                losses.append(smoothed_loss)
                lrs.append(lr)
                tot_train_loss += loss
                lr *= mult
                if lr_co: pb.update()
        except tf.errors.OutOfRangeError:
            pass
        return choose_lr(lrs,losses), batch_num, tot_train_loss

    def init_model(self,lr=None,sess=None,x=None,y=None,n_batches=None,lr_show=False,**kwargs):
        if not lr:
            sess.run(tf.global_variables_initializer())
            sess.run(self.iter.initializer,feed_dict={self.input_obs: x,self.input_actions: y})
            lr,_,_ = self.lr_find(n_batches,self.train,show=lr_show,masm=ceil(n_batches / 50),masm_auto=50,
                                           sess=sess,**kwargs)
        self.lr=lr
        return round(self.lr,6)

    def run_model(self,sess,writer,epoch,step,x,y,mode,keep_prob=1.0,lr_sched='const',lr_exp_base=100,lr_exp_dec=0.99,
                  lr_maxmin=10,c_t_ratio=9,n_batches=None,n_epochs=None):
        assert (mode is "train") or (mode is "eval"), "mode must be `train` or `eval`"
        sess.run(self.iter.initializer,feed_dict={self.input_obs: x,self.input_actions: y})
        total_loss = 0
        batch_num = 0
        try:
            while True:
                if mode == "train":
                    if lr_sched=='cycle': lr = param_sched(step,n_batches*n_epochs,self.lr / lr_maxmin,self.lr,'cycle',c_t_ratio)
                    elif lr_sched=='exp_dec': lr = param_sched(step,self.lr,mode='exp_dec',
                                                                         exp_base=lr_exp_base,exp_dec_rate=lr_exp_dec)
                    else: lr = self.lr
                    l, summaries = self.train(lr,keep_prob,sess)
                    step += 1
                elif mode == "eval":
                    l, summaries = self.evaluate(keep_prob,sess)
                writer.add_summary(summaries, global_step=step)
                total_loss += l
                batch_num += 1
        except tf.errors.OutOfRangeError:
            pass
        self.epoch.load(epoch+1,sess)
        self.gstep.load(step,sess)
        return step, total_loss / batch_num

    def var_summary(self, name):
        tf.summary.scalar("{} loss".format(name), self.loss)
        tf.summary.histogram("{} loss histogram".format(name), self.loss)
        return tf.summary.merge_all()