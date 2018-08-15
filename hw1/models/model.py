import tensorflow as tf
import time
from util import mkdir_rel
import numpy as np

class Model(object):
    def __init__(self, x_train, observations_dim, actions_dim, checkpoint_dir, learning_rate, batch_size):
        self.observations_dim = observations_dim
        self.actions_dim = actions_dim
        self.input_obs = tf.placeholder(tf.float32, [None, observations_dim], name='observations')
        self.input_actions = tf.placeholder(tf.float32, [None, actions_dim], name='actions')
        self.drop_prob = tf.placeholder(tf.float32 , name='drop_probability')
        # self.keep_prob =1.0
        self.mean_obs, self.var_obs = x_train.mean(axis=0), x_train.std(axis=0)
        self.gstep = tf.Variable(0, dtype=tf.int32,trainable=False, name='global_step')
        self.epoch = tf.Variable(0, dtype=tf.int32, name='epoch')
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir
        self.lr = tf.train.exponential_decay(learning_rate,self.gstep,100,0.99)
        self.x, self.y = None, None
        self.iter = self.get_dataset_iterator()
        self.pred = self.get_model()
        self.loss = self.get_loss()
        self.opt = self.get_optimizer()
        self.saver = tf.train.Saver()
        self.summ_train_loss, self.summ_val_loss = self.var_summary("training"), self.var_summary("validation")

    def get_dataset_iterator(self):
        with tf.name_scope('data'):
            dataset = tf.data.Dataset.from_tensor_slices((self.input_obs, self.input_actions)).batch(self.batch_size)
            iter = dataset.make_initializable_iterator()
            self.x, self.y = iter.get_next()
            # self.obs = tf.expand_dims(self.obs,0)
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
        return 0,0

    def get_model(self):
        obs_normalized = (self.x - self.mean_obs) / self.var_obs
        nn = tf.layers.dense(obs_normalized, 64, tf.nn.relu,kernel_initializer=
            tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=False),name='fc1')
        nn = tf.layers.dropout(nn , self.drop_prob , name='drop1')
        return tf.layers.dense(nn, self.actions_dim, name='pred')

    def get_loss(self):
        with tf.name_scope('loss'):
            return tf.losses.mean_squared_error(self.y,self.pred)

    def get_optimizer(self):
        return tf.train.AdamOptimizer(learning_rate=self.lr).minimize(
            self.loss, global_step=self.gstep)

    def train(self, sess, drop_prob):
        l, summ_train_loss, _ = sess.run([self.loss, self.summ_train_loss, self.opt],feed_dict={self.drop_prob: drop_prob})
        return l, summ_train_loss

    def evaluate(self, sess, drop_prob):
        return sess.run([self.loss, self.summ_val_loss],feed_dict={self.drop_prob: drop_prob})

    def predict(self, x, sess):
        y = np.zeros((x.shape[0],self.actions_dim),dtype=np.float32)
        sess.run(self.iter.initializer , feed_dict={self.input_obs: x, self.input_actions: y})
        return sess.run([self.pred], feed_dict={self.drop_prob: 1.0})

    def run_model(self, sess, writer, epoch, step,x,y, mode,drop_prob=1.0):
        assert (mode is "train") or (mode is "eval"), "mode must be `train` or `eval`"
        feed_dict = {}
        if mode == "train":
            feed_dict = {self.input_obs: x, self.input_actions: y}
        elif mode == "eval":
            feed_dict = {self.input_obs: x, self.input_actions: y}
        sess.run(self.iter.initializer,feed_dict=feed_dict)
        total_loss = 0
        n_batches = 0
        try:
            while True:
                l, summaries = None,None
                if mode == "train":
                    l, summaries = self.train(sess,drop_prob)
                    step += 1
                elif mode == "eval":
                    l, summaries = self.evaluate(sess,drop_prob)
                if writer is not None:
                    writer.add_summary(summaries, global_step=step)
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        # save next epoch and step (not ones finished)
        self.epoch.load(epoch+1,sess)
        self.gstep.load(step,sess)
        self.save(sess)
        # print('Average {0} loss at epoch {1}: {2:.10f}  Took: {3:.2f}s'.format(mode,epoch, total_loss / n_batches,
        #                                                                time.time() - start_time))
        return step, total_loss / n_batches

    def var_summary(self, name):
        tf.summary.scalar("{} loss".format(name), self.loss)
        tf.summary.histogram("{} loss histogram".format(name), self.loss)
        return tf.summary.merge_all()