#encoding=utf-8

import os
import tensorflow as tf
from models import Generative, Discriminative, Encoder, Classifier, Generative2
import tools

learning_rate = 0.000001
batch_size = 32
model_path = './data/model/GAN'
epoch = 10000
iter = 10

def train():
    seal = tf.placeholder(shape=(None, 224, 224, 3), dtype=tf.float32)
    noseal = tf.placeholder(shape=(None, 224, 224, 3), dtype=tf.float32)
    seal_label = tf.placeholder(shape=(None, 2), dtype=tf.float32)
    noseal_label = tf.placeholder(shape=(None, 2), dtype=tf.float32)

    # z = Encoder(seal)
    fake = Generative2(seal)

    feature_fake, logits_fake = Discriminative(fake)
    feature_real_seal, logits_real_seal = Discriminative(seal, reuse=True)
    feature_real_noseal, logits_real_noseal = Discriminative(noseal, reuse=True)

    feature_fake_noseal, logits_fake_noseal = Classifier(fake)
    feature_noseal, logits_noseal = Classifier(noseal, reuse=True)
    _, logits_seal = Classifier(seal, reuse=True)

    loss_D = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=seal_label, logits=logits_real_seal) + \
            tf.nn.softmax_cross_entropy_with_logits(labels=seal_label, logits=logits_real_noseal) + \
            tf.nn.softmax_cross_entropy_with_logits(labels=noseal_label, logits=logits_fake))

    loss_C = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=seal_label, logits=logits_seal) + \
                tf.nn.softmax_cross_entropy_with_logits(labels=noseal_label, logits=logits_noseal) + \
                tf.nn.softmax_cross_entropy_with_logits(labels=seal_label, logits=logits_fake_noseal))

    loss_GD = tools.get_loss_GD(feature_real_noseal, feature_real_seal, feature_fake)

    loss_GC = tools.get_loss_GC(feature_noseal, feature_fake_noseal)

    loss_G = tools.get_loss_G(seal, fake)

    # loss_KL = tools.get_loss_KL(z)

    all_var = tf.trainable_variables()

    var_C = [var for var in all_var if var.name.startswith('Classifier')]

    # var_E = [var for var in all_var if var.name.startswith('Encoder')]

    var_G = [var for var in all_var if var.name.startswith('Generative')]

    var_D = [var for var in all_var if var.name.startswith('Discriminative')]

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    opt_C = optimizer.minimize(loss_C, var_list=var_C)

    # opt_E = optimizer.minimize(3*loss_KL + loss_G, var_list=var_E)

    opt_G = optimizer.minimize(0.001*loss_G + loss_GC + loss_GD, var_list=var_G)

    opt_D = optimizer.minimize(loss_D, var_list=var_D)

    accuracy = 0.5*(tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.nn.softmax(logits_seal), 1),
                                                    tf.argmax(tf.nn.softmax(seal_label), 1)), tf.float32)) + \
                    tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.nn.softmax(logits_noseal), 1),
                                                    tf.argmax(tf.nn.softmax(noseal_label), 1)), tf.float32)))


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
                seal_img, seal_labels = tools.get_seal_set(batch_size/2, i*iter+j)
                noseal_img, noseal_labels = tools.get_noseal_set(batch_size/2, i*iter+j)
                # sess.run(opt_E, feed_dict={seal:seal_img})
                sess.run(opt_G, feed_dict={seal:seal_img, seal_label:seal_labels, noseal:noseal_img,
                                           noseal_label:noseal_labels})
                sess.run(opt_D, feed_dict={seal:seal_img, seal_label:seal_labels, noseal:noseal_img,
                                           noseal_label:noseal_labels})
                sess.run(opt_C, feed_dict={seal:seal_img, seal_label:seal_labels, noseal:noseal_img,
                                           noseal_label:noseal_labels})
            l_c, l_g, l_gc, l_gd, l_d, a = sess.run([loss_C, loss_G, loss_GC, loss_GD, loss_D, accuracy],
                                                          feed_dict={seal:seal_img, seal_label:seal_labels,
                                                                     noseal:noseal_img,noseal_label:noseal_labels})
            print 'Epoch ' + str(i) + ' : L_C = ' + str(l_c) + ' ; L_G = ' + str(l_g) + \
                ' ; L_GC = ' + str(l_gc) + ' ; L_GD = ' + str(l_gd) + ' ; L_D = ' + str(l_d) + ' ; acc = ' + str(a)
            if i % 10 == 0:
                saver.save(sess, os.path.join(model_path, 'model.ckpt'))
                seal_img = tools.get_test()
                fake_img = sess.run(fake, feed_dict={seal:seal_img})
                tools.save(fake_img[0, :, : ,:], i)

if __name__ == '__main__':
    train()