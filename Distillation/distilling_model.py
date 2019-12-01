'''
    原模型：
        [Test] Loss 1.60000~1.70000, Accuracy 0.8300~0.8600
        对真值y拟合的同时，也对软标签进行拟合。
    蒸馏后:
        [Test1] Accuracy 0.94250
        [Test2] Accuracy 0.93250
        [Test3] Accuracy 0.94650
'''


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
X, y = mnist['data'], mnist['target']

y_train_prob = np.load("./npy/y_prob.npy")
y_val_prob = y_train_prob[0:60000:6]

X_train = np.array(X[:60000], dtype=int)
y_train = np.array(y[:60000], dtype=int)
X_val = np.array(X[0:60000:6], dtype=int)
y_val = np.array(y[0:60000:6], dtype=int)
X_test = np.array(X[60000:], dtype=int)
y_test = np.array(y[60000:], dtype=int)
y_train_new = np.zeros((60000, 10))
y_val_new = np.zeros((10000, 10))
y_test_new = np.zeros((10000, 10))
print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(X_test.shape, y_test.shape)
print(sum(y_train[:6000]), sum(y_train[6000:12000]), sum(y_train[12000:18000]))
p = np.random.permutation(10000)
X_test = X_test[p] / 127.5 - 1
y_test = y_test[p]


def one_hot(data1, data2):
    for i in range(len(data2)):
        data1[i][data2[i]] = 1
    return data1

y_train_new = one_hot(y_train_new, y_train)
y_val_new = one_hot(y_val_new, y_val)
y_test_new = one_hot(y_test_new, y_test)


class Help:
    def __init__(self, data, labels, y_prob, need_shuffle):
        self._data = data
        self._data = self._data / 127.5 - 1
        self._labels = labels
        self._y_prob = y_prob
        print(self._data.shape)
        print(self._labels.shape)
        print(self._y_prob.shape)

        self._num_examples = len(self._data)
        self._need_shuffle = need_shuffle
        self._indicator = 0
        if self._need_shuffle:
            self._shuffle_data()

    def _shuffle_data(self):
        # [0,1,2,3,4,5] -> [5,3,2,4,0,1]
        p = np.random.permutation(self._num_examples)
        self._data = self._data[p]
        self._labels = self._labels[p]
        self._y_prob = self._y_prob[p]

    def next_batch(self, batch_size):
        """return batch_size examples as a batch."""
        end_indicator = self._indicator + batch_size
        if end_indicator > self._num_examples:
            if self._need_shuffle:
                self._shuffle_data()
                self._indicator = 0
                end_indicator = batch_size
            else:
                raise Exception("have no more examples")
        if end_indicator > self._num_examples:
            raise Exception("batch size is larger than all examples")
        batch_data = self._data[self._indicator: end_indicator]
        batch_labels = self._labels[self._indicator: end_indicator]
        batch_y_prob = self._y_prob[self._indicator: end_indicator]
        self._indicator = end_indicator
        return batch_data, batch_labels, batch_y_prob

train_data = Help(X_train, y_train_new, y_train_prob, True)
val_data = Help(X_val, y_val_new, y_val_prob, True)

with tf.name_scope("build_network"):

    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])
    y_p = tf.placeholder(tf.float32, [None, 10])
    y_prob = tf.nn.softmax(y_p / 20)
    hidden1 = tf.layers.dense(x, 20, activation=tf.nn.relu)
    y_ = tf.nn.softmax(tf.layers.dense(hidden1, 10))
    hard_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_))
    soft_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_prob, logits=y_))
    loss = 0.2*hard_loss + 0.8*20*20*soft_loss
    predict = tf.argmax(y_, 1)
    print(predict.dtype, y.dtype)
    correct_prediction = tf.equal(predict, tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
    with tf.name_scope('train_op'):
        train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)


with tf.name_scope("train"):
    init = tf.global_variables_initializer()
    batch_size = 30
    train_steps = 2000
    epochs = 30

    with tf.Session() as sess:
        sess.run(init)
        train_loss = []
        train_loss_all = []
        for epoch in range(epochs):
            train_acc, val_acc = [], []
            for i in range(train_steps):
                batch_data, batch_labels, batch_y_prob = train_data.next_batch(batch_size)
                loss_train, acc_train, _ = sess.run(
                    [loss, accuracy, train_op],
                    feed_dict={
                        x: batch_data,
                        y: batch_labels,
                        y_p: batch_y_prob})
                train_loss.append(loss_train)
                if (i + 1) % 100 == 0:
                    train_acc.append(acc_train)
                    print('[Train] Epoch: %d Step: %d, loss: %4.5f'
                          % (epoch+1, i + 1, loss_train))
            train_loss_all.append(np.mean(train_loss))
            for i in range(5):
                val_x, val_labels, val_y_prob = val_data.next_batch(batch_size)
                loss_val, acc_val, _ = sess.run(
                    [loss, accuracy, train_op], feed_dict={
                        x: val_x,
                        y: val_labels,
                        y_p: val_y_prob
                    })
                val_acc.append(acc_val)
                print('[Val] Epoch: %d Step: %d, loss: %4.5f'
                      % (epoch+1, i + 1, loss_val))
            print("Epoch %d Train_Acc: %4.5f, Val_Acc: %4.5f" %
                (epoch+1, np.mean(train_acc), np.mean(val_acc)))
        acc_test, _ = sess.run([accuracy, train_op],
                                          feed_dict={
                                              x: X_test,
                                              y: y_test_new,
                                              y_p: np.zeros((10000, 10))
                                          })
        print("[Test] Accuracy %4.5f" % (acc_test))

        plt.plot(train_loss_all)
        plt.title("Train Loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.show()













