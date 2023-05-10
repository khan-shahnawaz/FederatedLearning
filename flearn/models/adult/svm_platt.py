import numpy as np
import tensorflow as tf
from tqdm import trange

from flearn.utils.model_utils import batch_data, gen_batch
from flearn.utils.tf_utils import graph_size
from flearn.utils.tf_utils import process_grad


class Model(object):    
    def __init__(self, num_classes, q, optimizer, seed=1):

        # params
        self.num_classes = num_classes

        # create computation graph        
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(123+seed)
            self.features, self.labels, self.train_op, self.grads, self.eval_metric_ops, self.loss, self.predictions, self.y_pred = self.create_model(q, optimizer)
            self.saver = tf.train.Saver()
        self.sess = tf.Session(graph=self.graph)

        # find memory footprint and compute cost of the model
        self.size = graph_size(self.graph)
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            metadata = tf.RunMetadata()
            opts = tf.profiler.ProfileOptionBuilder.float_operation()
            self.flops = tf.profiler.profile(self.graph, run_meta=metadata, cmd='scope', options=opts).total_float_ops
            
    def create_platt_variables(self):
        with self.graph.as_default():
            platt_a = tf.Variable(0.0, dtype=tf.float32, name='platt_a')
            platt_b = tf.Variable(0.0, dtype=tf.float32, name='platt_b')
            return platt_a, platt_b

    def apply_platt_scaling(self, logits, a, b):
        scaled_logits = a + b * logits
        return tf.sigmoid(scaled_logits)

    def platt_loss(self, logits, labels):
        probabilities = self.apply_platt_scaling(logits, self.platt_a, self.platt_b)
        nll = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
        return nll

    def fit_platt_scaling(self, logits, labels, num_epochs=100, learning_rate=0.01):
        self.platt_a, self.platt_b = self.create_platt_variables()

        with self.graph.as_default():
            logits_placeholder = tf.placeholder(tf.float32, shape=[None], name='logits')
            labels_placeholder = tf.placeholder(tf.float32, shape=[None], name='labels')

            loss = self.platt_loss(logits_placeholder, labels_placeholder)

            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            train_op = optimizer.minimize(loss)

        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(num_epochs):
                _, current_loss = sess.run([train_op, loss], feed_dict={logits_placeholder: logits, labels_placeholder: labels})

        return sess.run(self.platt_a), sess.run(self.platt_b)

            
    def create_model(self, q, optimizer):
        """Model function for Logistic Regression."""
        features = tf.placeholder(tf.float32, shape=[None, 62], name='features')
        labels = tf.placeholder(tf.float32, shape=[None, 1], name='labels')
        
        W = tf.Variable(tf.zeros([62, 1]))
        b = tf.Variable(tf.zeros([1]))
        y_pred = tf.matmul(features, W) + b

        
        loss = 0.01 * tf.reduce_sum(tf.square(W)) + tf.reduce_mean(
            tf.maximum(tf.zeros_like(labels), (1 - labels * y_pred))
        )

        grads_and_vars = optimizer.compute_gradients(loss)
        grads, _ = zip(*grads_and_vars)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())
        eval_metric_ops = tf.count_nonzero(tf.equal(labels, tf.sign(y_pred)))
        return features, labels, train_op, grads, eval_metric_ops, loss, tf.sign(y_pred), y_pred


    def set_params(self, model_params=None):
        if model_params is not None:
            with self.graph.as_default():
                all_vars = tf.trainable_variables()
                for variable, value in zip(all_vars, model_params):
                    variable.load(value, self.sess)

    def get_params(self):
        with self.graph.as_default():
            model_params = self.sess.run(tf.trainable_variables())
        return model_params

    def get_gradients(self, data, model_len):

        grads = np.zeros(model_len)
        num_samples = len(data['y'])
        
        with self.graph.as_default():
            model_grads = self.sess.run(self.grads,
                feed_dict={self.features: data['x'], self.labels: data['y']})
            grads = process_grad(model_grads)

        return num_samples, grads

    def get_loss(self, data):
        with self.graph.as_default():
            loss = self.sess.run(self.loss, feed_dict={self.features: data['x'], self.labels: data['y']})
        return loss
    
    def solve_inner(self, data, num_epochs=1, batch_size=32):
        '''Solves local optimization problem'''
        for _ in range(num_epochs):
            for X, y in batch_data(data, batch_size):
                with self.graph.as_default():
                    _, pred = self.sess.run([self.train_op, self.predictions],
                        feed_dict={self.features: X, self.labels: y})
        soln = self.get_params()
        comp = num_epochs * (len(data['y'])//batch_size) * batch_size * self.flops
        return soln, comp

    def solve_sgd(self, mini_batch_data):
        with self.graph.as_default():
            grads, loss, _ = self.sess.run([self.grads, self.loss, self.train_op],
                                    feed_dict={self.features: mini_batch_data[0], self.labels: mini_batch_data[1]})

        weights = self.get_params()
        return grads, loss, weights
    def test(self, data):
        '''
        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        '''
        if len(data['y']) == 0: # if the client does not have any data
            return 0, 0
        
        with self.graph.as_default():
            tot_correct, loss = self.sess.run([self.eval_metric_ops, self.loss], 
                feed_dict={self.features: data['x'], self.labels: data['y']})
        
        
        return tot_correct, loss
    
    
    def test_with_mce(self, data):
        '''
        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        '''
        if len(data['y']) == 0: # if the client does not have any data
            return 0, 0
        
        with self.graph.as_default():
            tot_correct, loss, y_pred = self.sess.run([self.eval_metric_ops, self.loss, self.y_pred], 
                feed_dict={self.features: data['x'], self.labels: data['y']})

        #Get the confidence probabilites for MCE from y_pred
        pred_prob  = np.zeros(len(y_pred))
        for i in range(len(y_pred)):
            pred_prob[i] = max(1/(1+np.exp(-y_pred[i])), 1/(1+np.exp(y_pred[i])))   #Ensure correct percentage for negative predictions also
        
        platt_a, platt_b = self.fit_platt_scaling(y_pred, data['y'])
        
        #Calculate the Maximum calibration error
        bins = np.linspace(0, 1, 11)
        bin_index = np.digitize(pred_prob, bins)
        bin_index = bin_index - 1
        bin_correct = np.zeros(10)
        bin_total = np.zeros(10)
        for i in range(len(bin_index)):
            bin_total[bin_index[i]] = bin_total[bin_index[i]] + 1
            if data['y'][i] == np.sign(y_pred[i]):
                bin_correct[bin_index[i]] = bin_correct[bin_index[i]] + 1
        for i in range(len(bin_correct)):
            if bin_total[i] != 0:
                bin_correct[i] = bin_correct[i]/bin_total[i]
        mce = 0
        #Calculate average confidence for each bin
        avg_conf = np.zeros(10)
        for i in range(len(bin_index)):
            avg_conf[bin_index[i]] = avg_conf[bin_index[i]] + pred_prob[i]
        for i in range(len(bin_correct)):
            if bin_total[i] != 0:
                avg_conf[i] = avg_conf[i]/bin_total[i]
        #Calculate MCE
        for i in range(len(bin_correct)):
            mce = max(mce, abs(bin_correct[i] - avg_conf[i]))
        return tot_correct, loss,mce
    
    def close(self):
        self.sess.close()