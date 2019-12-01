'''
    About Model2
    [Test] Loss 1.50000~1.60000, Accuracy 0.9300~0.9400

'''











import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data" , one_hot = True)
## 每个批次的大小
batch_size = 100

# 计算一共有多少批次
n_batch = mnist.train.num_examples // batch_size

with tf.name_scope("build_network"):

    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])
    hidden1 = tf.layers.dense(x, 20, activation=tf.nn.relu)
    y_ = tf.nn.softmax(tf.layers.dense(hidden1, 10))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_))
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
                batch_data, batch_labels = mnist.train.next_batch(batch_size)
                loss_train, acc_train, _ = sess.run(
                    [loss, accuracy, train_op],
                    feed_dict={
                        x: batch_data,
                        y: batch_labels})
                train_loss.append(loss_train)
                if (i + 1) % 100 == 0:
                    train_acc.append(acc_train)
                    print('[Train] Epoch: %d Step: %d, loss: %4.5f'
                          % (epoch+1, i + 1, loss_train))
            train_loss_all.append(np.mean(train_loss))
            for i in range(5):
                val_x, val_y = mnist.validation.next_batch(batch_size)
                loss_val, acc_val, _ = sess.run(
                    [loss, accuracy, train_op], feed_dict={
                        x: val_x,
                        y: val_y
                    })
                val_acc.append(acc_val)
                print('[Val] Epoch: %d Step: %d, loss: %4.5f'
                      % (epoch+1, i + 1, loss_val))
            print("Epoch %d Train_Acc: %4.5f, Val_Acc: %4.5f" %
                (epoch+1, np.mean(train_acc), np.mean(val_acc)))
        test_x, test_y = mnist.test.next_batch(10000)
        loss_test, acc_test, _ = sess.run([loss, accuracy, train_op],
                                          feed_dict={
                                              x: test_x,
                                              y: test_y
                                          })
        print("[Test] Loss %4.5f, Accuracy %4.5f" % (loss_test, acc_test))

        plt.plot(train_loss_all)
        plt.title("Train Loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.show()






