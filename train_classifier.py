#encoding=utf-8

import tensorflow as tf
import os
from models import Classifier
import tools

model_path = './data/classifier'
lr_init = 0.001
batch_size = 64
epoch = 2000
iter = 10

def train():
    input = tf.placeholder(shape=(None, 224, 224, 3), dtype=tf.float32)
    label = tf.placeholder(shape=(None, 2), dtype=tf.float32)
    global_step = tf.Variable(0)

    _, logits = Classifier(input)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))
    lr = tf.train.exponential_decay(learning_rate=lr_init, global_step=global_step, decay_steps=100,
                                    decay_rate=0.98, staircase=True)
    opt = tf.train.AdamOptimizer(lr).minimize(loss, global_step)
    logits = tf.nn.softmax(logits)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(label, 1), tf.argmax(logits, 1)), tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver = tf.train.Saver(max_to_keep=1)
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            pass
        for i in range(1, epoch+1):
            for j in range(iter):
                img, labels = tools.get_classifier_pic(batch_size, i*epoch+j)
                sess.run([opt], feed_dict={input:img, label:labels})
            l, a = sess.run([loss, accuracy], feed_dict={input:img, label:labels})
            print 'Epoch ' + str(i) + ' :  Loss = ' + str(l) + ' ; Accuracy = ' + str(a)
            saver.save(sess, os.path.join(model_path, 'model.ckpt'))

if __name__ == '__main__':
    train()