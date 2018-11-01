import tensorflow as tf
from tfac.queue_input import QueueInput


class ModeKeys(object):
    TRAIN = 'train'
    PREDICT = 'predict'


class NN(object):
    def __init__(self, sample_fn, need_summary=True):
        self.sample_fn = sample_fn
        self.need_summary = need_summary
        self.summary = []
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.build_input()
            idx, batch_features, batch_labels = self.qi.build_op(32)
            self.train_net = self.build_q_net(batch_features['s'], 'origin',
                                              False)
            self.target_net = self.build_q_net(batch_features['s_'], 'target',
                                               False, False)
            self.train_op = self.build_train_op(batch_features, batch_labels)
            self.update_op = self.build_update_op(self.train_net['weights'],
                                                  self.target_net['weights'])
            if self.need_summary:
                self.merged = tf.summary.merge(self.summary)
            self.pred_net = self.build_q_net(self.features_predict['s'],
                                             'target', tf.AUTO_REUSE, False)
            self.sess = tf.InteractiveSession()
            self.saver = tf.train.Saver(max_to_keep=5)
            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)
            self.run()
            self.sess.graph.finalize()
            if self.need_summary:
                self.sw_step = 0
                self.summary_writer = tf.summary.FileWriter(
                    'tf_log/summary', self.sess.graph)

    def load(self, load_path):
        checkpoint = tf.train.get_checkpoint_state(load_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            return True
        return False

    def save(self, save_path):
        self.saver.save(self.sess, save_path, self.global_step)

    def update_param(self):
        self.sess.run(self.update_op)

    def train(self):
        if self.need_summary:
            _, loss, merged, self.global_step = self.sess.run(
                [self.train_op, self.train_net['loss'],
                 self.merged, self.global_step_op]
            )
            self.summary_writer.add_summary(merged, self.sw_step)
            self.sw_step += 1
        else:
            _, loss, self.global_step = self.sess.run(
                [self.train_op, self.train_net['loss'],
                 self.global_step_op]
            )
        return loss

    def predict(self, s):
        return self.sess.run(self.pred_net['pred']['q'],
                             feed_dict={
                                 self.features_predict['s']: s})

    def build_input(self):
        self.features = {
            's': tf.placeholder(tf.uint8, [None, 9, 9, 12 * 3], 's'),
            'a': tf.placeholder(tf.uint8, [None, ], 'a'),
            'r': tf.placeholder(tf.float32, [None, ], 'r'),
            't': tf.placeholder(tf.uint8, [None, ], 't'),
            's_': tf.placeholder(tf.uint8, [None, 9, 9, 12 * 3], 's_'),
        }

        self.features_predict = {
            's': tf.placeholder(tf.uint8, [None, 9, 9, 12 * 3], 's_pred')
        }

        self.qi = QueueInput(self.features, {}, [400, 400], 'qi')

    def build_q_net(self, x, prefix, reuse, trainable=True):
        with tf.variable_scope(prefix, reuse=reuse):
            w_init = tf.contrib.layers.xavier_initializer()
            b_init = tf.contrib.layers.xavier_initializer()
            x = tf.cast(x, tf.float32)

            x = tf.layers.conv2d(
                x, 32, 4, 2, activation=tf.nn.relu, name='conv_1',
                kernel_initializer=w_init, bias_initializer=b_init,
                trainable=trainable)
            x = tf.layers.conv2d(
                x, 64, 3, 1, activation=tf.nn.relu, name='conv_2',
                kernel_initializer=w_init, bias_initializer=b_init,
                trainable=trainable)
            x = tf.layers.flatten(x)
            x = tf.layers.dense(x, 512, activation=tf.nn.relu,
                                name='dense_1',
                                trainable=trainable)
            q = tf.layers.dense(x, 9, activation=tf.nn.relu,
                                name='dense_2',
                                trainable=trainable)

            pred = {
                'q': tf.identity(q, name='q')
            }

            weights = [v for v in tf.global_variables() if prefix in v.name]
            net = {'pred': pred, 'weights': weights}
            return net

    def build_train_op(self, features, labels):
        q = self.train_net['pred']['q']
        q_ = self.target_net['pred']['q']
        a = features['a']
        r = features['r']
        t = tf.cast(features['t'], tf.float32)
        q_value = tf.reduce_sum(q * tf.one_hot(a, 9), 1)
        q_max = tf.reduce_max(q_, axis=1, name='q_max_s_')
        q_target = r + (1. - t) * 0.99 * q_max
        mse_loss = tf.losses.mean_squared_error(
            q_target, q_value)
        l2_loss = 1e-6 * tf.add_n([
            tf.nn.l2_loss(v)
            for v in tf.trainable_variables()])
        loss = mse_loss + l2_loss
        self.train_net['loss'] = loss
        self.summary.append(tf.summary.scalar('mse_loss', mse_loss))
        self.summary.append(tf.summary.scalar('l2_loss', l2_loss))
        self.summary.append(tf.summary.scalar('loss', loss))
        self.global_step_op = tf.train.get_or_create_global_step()
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-5)
        grads = optimizer.compute_gradients(loss)
        self.summary.extend([
            tf.summary.histogram('%s-grad' % g[1].name, g[0])
            for g in grads if g[0] is not None])
        train_op = optimizer.apply_gradients(
            grads, global_step=self.global_step_op)
        return train_op

    def build_update_op(self, o_weights, t_weights):
        update_op = [tf.assign(t, o) for t, o in zip(t_weights, o_weights)]
        update_op = tf.group(*update_op, name='update')
        return update_op

    def run(self):
        self.qi.run(self.sess, [self.sample_fn])

    def close(self):
        self.qi.close()
        self.sess.close()
