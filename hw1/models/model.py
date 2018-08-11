import tensorflow as tf
import os
import time
from util import mkdir_rel


class Model(object):
    def __init__(self, x_train, num_observations, num_actions, checkpoint_dir, learning_rate, batch_size):

        self.num_obs = num_observations
        self.num_actions = num_actions
        self.obs = tf.placeholder(tf.float32, [None, num_observations],name='observations')
        self.actions = tf.placeholder(tf.float32, [None, num_actions],name='actions')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_probability')
        self.mean_obs, self.var_obs = x_train.mean(axis=0), x_train.std(axis=0)
        self.gstep = tf.Variable(0, dtype=tf.int32,trainable=False, name='global_step')
        self.loss = self.loss()
        self.opt = self.get_optimizer()
        self.pred = self.get_model()
        self.checkpoint_dir = checkpoint_dir
        self.saver = tf.train.Saver()
        self.lr = tf.train.exponential_decay(learning_rate,self.gstep,100,0.99)
        self.skip_step = 20
        self.summ_train_loss, self.summ_val_loss = self.var_summary("training"), self.var_summary("validation")
        self.batch_size = batch_size
        self.x, self.y = None, None
        self.iter = self.get_dataset_iterator()

    def get_dataset_iterator(self):
        with tf.name_scope('data'):
            dataset = tf.data.Dataset.from_tensor_slices((self.obs, self.actions)).batch(self.batch_size).repeat()
            iter = dataset.make_initializable_iterator()
            self.x, self.y = iter.get_next()
            # self.obs = tf.expand_dims(self.obs,0)
            self.x = tf.reshape(self.x,shape=[-1, self.num_obs])
            self.y = tf.reshape(self.actions, shape=[-1, self.num_actions])
            return iter

    def save(self,sess):
        mkdir_rel(self.checkpoint_dir)
        self.saver.save(sess, self.checkpoint_dir +'/'+ 'model', global_step=self.gstep)

    def restore(self,sess):
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.checkpoint_dir))
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(sess, ckpt.model_checkpoint_path)

    def get_model(self):
        obs_normalized = (self.x - self.mean_obs) / self.var_obs
        with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
            nn = tf.layers.dense(obs_normalized, 64, tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer,name='fc1')
            nn = tf.nn.dropout(nn,self.keep_prob,name='drop1')
            return tf.layers.dense(nn, self.num_actions, name='pred')

    def get_loss(self):
        with tf.name_scope('loss'):
            return tf.losses.mean_squared_error(self.y,self.pred)

    def get_optimizer(self):
        return tf.train.AdamOptimizer(learning_rate=self.lr).minimize(
            self.loss, global_step=self.gstep)

    def train(self, sess, x, y, keep_prob):
        l, summ_train_loss, _ = sess.run([self.loss, self.summ_train_loss, self.opt],
                                            feed_dict={self.obs: x, self.actions: y,
                                                       self.keep_prob: keep_prob})
        return l, summ_train_loss

    def evaluate(self, sess, x, y):
        return sess.run([self.loss, self.summ_val_loss],
                        feed_dict={self.obs: x, self.actions: y, self.keep_prob: 1})

    def predict(self, sess, x):
        return sess.run(self.pred, feed_dict={self.obs: x, self.keep_prob: 1})

    def run_model(self, sess, writer, epoch, step,x,y,keep_prob, mode):
        assert mode is ("train" or "evaluate"), "mode must be `train` or `evaluate`"
        start_time = time.time()
        sess.run(self.iter.initializer)
        total_loss = 0
        n_batches = 0
        try:
            while True:
                l, summaries = None,None
                if mode == "train":
                    l, summaries = self.train(sess,x,y,keep_prob)
                elif mode == "evaluate":
                    l, summaries = self.evaluate(sess,x,y)
                writer.add_summary(summaries, global_step=step)
                if (step + 1) % self.skip_step == 0:
                    print('Loss at step {0}: {1}'.format(step, l))
                step += 1
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        self.save(sess)
        print('Average {} loss at epoch {}: {}'.format(mode,epoch, total_loss / n_batches))
        print('Took {0} sec'.format(time.time() - start_time))
        return step

    def var_summary(self, name):
        with tf.name_scope("summaries"):
            tf.summary.scalar("{} loss".format(name), self.loss)
            tf.summary.histogram("{} loss histogram".format(name), self.loss)
            return tf.summary.merge_all()