from gan3d.progressive_model import generator
from gan3d.progressive_model import dicriminator
from gan3d.utils import plotLineGraph
from gan3d.utils import visImage
from gan3d.utils import visInterpolation
from gan3d.utils import createPath

import time
import tensorflow as tf
import numpy as np

def visualization(d_layers,
                  g_layers,
                  batch_size,
                  no_batches,
                  pixel_shape,
                  d_lr,
                  g_lr,
                  beta,
                  d_thresh,
                  g_thresh,
                  pro_growth,
                  no_vis_mul = 4,
                  pre_trained_models = None):
    
    tf.reset_default_graph()
    
    # Define input
    z_vector = tf.placeholder(shape=[batch_size, 200],dtype=tf.float32) 
    x_vector = tf.placeholder(shape=[batch_size, pixel_shape, pixel_shape, pixel_shape,1], dtype=tf.float32)
    
    # Define model
    gen_out = generator(z_vector, g_layers, batch_size, pro_growth, training=True)
    gen_out_test = generator(z_vector, g_layers, batch_size, pro_growth, training=False, reuse=True)
    dis_out, dis_out_no = dicriminator(x_vector, d_layers, batch_size, pro_growth, training=True)
    dis_gen_out, dis_gen_out_no = dicriminator(gen_out, d_layers, batch_size, pro_growth, training=True, reuse=True)
    
    # Get stats + loss
    nx = tf.reduce_sum(tf.cast(dis_out > 0.5, tf.int32))
    nz = tf.reduce_sum(tf.cast(dis_gen_out < 0.5, tf.int32))
    d_acc = tf.divide(nx + nz, 2 * batch_size)
    
    # Loss
    d_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_out_no, 
                                                     labels=tf.ones_like(dis_out_no)) + \
                tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_gen_out_no, labels=tf.zeros_like(dis_gen_out_no))
    g_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_gen_out_no, labels=tf.ones_like(dis_gen_out_no))
    d_loss = tf.reduce_mean(d_loss)
    g_loss = tf.reduce_mean(g_loss)

    # Optimizer
    gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN_GEN_")
    disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN_DIS_")
    optimizer_op_d = tf.train.AdamOptimizer(learning_rate=d_lr,beta1=beta).minimize(d_loss, var_list=disc_vars)
    optimizer_op_g = tf.train.AdamOptimizer(learning_rate=g_lr,beta1=beta).minimize(g_loss, var_list=gen_vars)

    # Visualization
    createPath('output/')
    
    for pre_trained_model in pre_trained_models.keys():
        cur_model_name = pre_trained_model
        print('Model name: ' + cur_model_name)

        createPath('output/' + cur_model_name + '/vis_test/')
        
        for pre_trained_model_step in pre_trained_models[pre_trained_model]:
            with tf.Session() as sess:
                #merge = tf.summary.merge_all()
                #writer = tf.summary.FileWriter("log/{}".format(cur_model_name), sess.graph)
                
                createPath('output/' + cur_model_name + '/vis_test/' + pre_trained_model_step)
                createPath('output/' + cur_model_name + '/vis_test/' + pre_trained_model_step + '/output_test')
                createPath('output/' + cur_model_name + '/vis_test/' + pre_trained_model_step + '/output_train')

                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()

                if pre_trained_model_step is not None:
                    print('Pretrianed model: ' + pre_trained_model_step)
                    try:
                        print('Load the model from: ' + 'output/' + pre_trained_model + '/model/')
                        print('Load iteration: ' + 'output/' + pre_trained_model + '/model/')
                        saver.restore(sess, 'output/' + pre_trained_model + '/model/' + pre_trained_model_step)
                    except Exception:
                        raise ValueError("Load model Failed!")
                
                no_vis = batch_size * no_vis_mul
                z_total = np.random.normal(0, 0.33, size=[no_vis, 200]).astype(np.float32)
                np.save('output/' + cur_model_name + '/vis_test/' + pre_trained_model_step + '/data.npy', z_total)
                
                for i in range(no_vis_mul):
                    z = z_total[(0 + batch_size*i):((1+i)*batch_size)]
                    output_train = sess.run([gen_out], feed_dict={z_vector:z})
                    output_test = sess.run([gen_out_test], feed_dict={z_vector:z})
                    for k in range(batch_size):
                        visImage(output_test[0][k], 
                                 'output/' + 
                                 cur_model_name + 
                                 '/vis_test/' + 
                                 pre_trained_model_step + 
                                 '/output_test/'  + str(0).replace('.', '') + '_' + 
                                 str((0 + batch_size*i) + k).zfill(5))
                        visImage(output_train[0][k], 
                                 'output/' + 
                                 cur_model_name + 
                                 '/vis_test/' + 
                                 pre_trained_model_step + 
                                 '/output_train/'  + str(0).replace('.', '') + '_' + 
                                 str((0 + batch_size*i) + k).zfill(5))
                        for treshhold in [0.9]:
                            print(str(treshhold).replace('.', '') + '_' + str((0 + batch_size*i) + k).zfill(5))
                            visImage(output_test[0][k]>treshhold, 
                                     'output/' + 
                                     cur_model_name + 
                                     '/vis_test/' + 
                                     pre_trained_model_step + 
                                     '/output_test/'  + str(treshhold).replace('.', '') + '_' + 
                                     str((0 + batch_size*i) + k).zfill(5))
                            visImage(output_train[0][k]>treshhold, 
                                     'output/' + 
                                     cur_model_name + 
                                     '/vis_test/' + 
                                     pre_trained_model_step + 
                                     '/output_train/'  + str(treshhold).replace('.', '') + '_' + 
                                     str((0 + batch_size*i) + k).zfill(5))


def interpolation(v1, 
                  v2,
                  no_steps,
                  d_layers,
                  g_layers,
                  batch_size,
                  no_batches,
                  pixel_shape,
                  d_lr,
                  g_lr,
                  beta,
                  d_thresh,
                  g_thresh,
                  pro_growth,
                  no_vis_mul = 4,
                  pre_trained_models = None):
    
    batch_size = no_steps
    tf.reset_default_graph()
    
    # Define input
    z_vector = tf.placeholder(shape=[batch_size, 200],dtype=tf.float32) 
    x_vector = tf.placeholder(shape=[batch_size, pixel_shape, pixel_shape, pixel_shape,1], dtype=tf.float32)
    
    # Define model
    gen_out = generator(z_vector, g_layers, batch_size, pro_growth, training=True)
    gen_out_test = generator(z_vector, g_layers, batch_size, pro_growth, training=False, reuse=True)
    dis_out, dis_out_no = dicriminator(x_vector, d_layers, batch_size, pro_growth, training=True)
    dis_gen_out, dis_gen_out_no = dicriminator(gen_out, d_layers, batch_size, pro_growth, training=True, reuse=True)
    
    # Get stats + loss
    nx = tf.reduce_sum(tf.cast(dis_out > 0.5, tf.int32))
    nz = tf.reduce_sum(tf.cast(dis_gen_out < 0.5, tf.int32))
    d_acc = tf.divide(nx + nz, 2 * batch_size)
    
    # Loss
    d_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_out_no, 
                                                     labels=tf.ones_like(dis_out_no)) + \
                tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_gen_out_no, labels=tf.zeros_like(dis_gen_out_no))
    g_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_gen_out_no, labels=tf.ones_like(dis_gen_out_no))
    d_loss = tf.reduce_mean(d_loss)
    g_loss = tf.reduce_mean(g_loss)

    # Optimizer
    gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN_GEN_")
    disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN_DIS_")
    optimizer_op_d = tf.train.AdamOptimizer(learning_rate=d_lr,beta1=beta).minimize(d_loss, var_list=disc_vars)
    optimizer_op_g = tf.train.AdamOptimizer(learning_rate=g_lr,beta1=beta).minimize(g_loss, var_list=gen_vars)

    # Visualization
    createPath('output/')
    
    for pre_trained_model in pre_trained_models.keys():
        cur_model_name = pre_trained_model
        print('Model name: ' + cur_model_name)

        createPath('output/' + cur_model_name + '/vis_test/')
        
        for pre_trained_model_step in pre_trained_models[pre_trained_model]:
            with tf.Session() as sess:
                #merge = tf.summary.merge_all()
                #writer = tf.summary.FileWriter("log/{}".format(cur_model_name), sess.graph)
                
                createPath('output/' + cur_model_name + '/vis_test/' + pre_trained_model_step)
                createPath('output/' + cur_model_name + '/vis_test/' + pre_trained_model_step + '/interpolation')

                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()

                if pre_trained_model_step is not None:
                    print('Pretrianed model: ' + pre_trained_model_step)
                    try:
                        print('Load the model from: ' + 'output/' + pre_trained_model + '/model/')
                        print('Load iteration: ' + 'output/' + pre_trained_model + '/model/')
                        saver.restore(sess, 'output/' + pre_trained_model + '/model/' + pre_trained_model_step)
                    except Exception:
                        raise ValueError("Load model Failed!")
                
                zs = []
                steps = np.linspace(0, 1, no_steps)
                for step in steps:
                    # Latent space interpolation
                    z = v1*(1-step) + v2*step
                    zs.append(z)
                
                z = np.array(zs)
                np.save('output/' + cur_model_name + '/vis_test/' + pre_trained_model_step + '/interpolation.npy', z)
                
                output_train = sess.run([gen_out], feed_dict={z_vector:z})
                output_test = sess.run([gen_out_test], feed_dict={z_vector:z})
                
                visInterpolation(output_train,
                                 'output/' + cur_model_name + '/vis_test/' + 
                                 pre_trained_model_step + '/interpolation/' + 'plot.png')


def interpolation_jupyter(v1, 
                          v2,
                          no_steps,
                          d_layers,
                          g_layers,
                          batch_size,
                          no_batches,
                          pixel_shape,
                          d_lr,
                          g_lr,
                          beta,
                          d_thresh,
                          g_thresh,
                          pro_growth,
                          no_vis_mul = 4,
                          pre_trained_models = None):
    
    batch_size = no_steps
    tf.reset_default_graph()
    
    # Define input
    z_vector = tf.placeholder(shape=[batch_size, 200],dtype=tf.float32) 
    x_vector = tf.placeholder(shape=[batch_size, pixel_shape, pixel_shape, pixel_shape,1], dtype=tf.float32)
    
    # Define model
    gen_out = generator(z_vector, g_layers, batch_size, pro_growth, training=True)
    gen_out_test = generator(z_vector, g_layers, batch_size, pro_growth, training=False, reuse=True)
    dis_out, dis_out_no = dicriminator(x_vector, d_layers, batch_size, pro_growth, training=True)
    dis_gen_out, dis_gen_out_no = dicriminator(gen_out, d_layers, batch_size, pro_growth, training=True, reuse=True)
    
    # Get stats + loss
    nx = tf.reduce_sum(tf.cast(dis_out > 0.5, tf.int32))
    nz = tf.reduce_sum(tf.cast(dis_gen_out < 0.5, tf.int32))
    d_acc = tf.divide(nx + nz, 2 * batch_size)
    
    # Loss
    d_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_out_no, 
                                                     labels=tf.ones_like(dis_out_no)) + \
                tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_gen_out_no, labels=tf.zeros_like(dis_gen_out_no))
    g_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_gen_out_no, labels=tf.ones_like(dis_gen_out_no))
    d_loss = tf.reduce_mean(d_loss)
    g_loss = tf.reduce_mean(g_loss)

    # Optimizer
    gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN_GEN_")
    disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN_DIS_")
    optimizer_op_d = tf.train.AdamOptimizer(learning_rate=d_lr,beta1=beta).minimize(d_loss, var_list=disc_vars)
    optimizer_op_g = tf.train.AdamOptimizer(learning_rate=g_lr,beta1=beta).minimize(g_loss, var_list=gen_vars)

    # Visualization
    createPath('output/')
    
    for pre_trained_model in pre_trained_models.keys():
        cur_model_name = pre_trained_model
        print('Model name: ' + cur_model_name)

        createPath('output/' + cur_model_name + '/vis_test/')
        
        for pre_trained_model_step in pre_trained_models[pre_trained_model]:
            with tf.Session() as sess:
                #merge = tf.summary.merge_all()
                #writer = tf.summary.FileWriter("log/{}".format(cur_model_name), sess.graph)
                
                createPath('output/' + cur_model_name + '/vis_test/' + pre_trained_model_step)
                createPath('output/' + cur_model_name + '/vis_test/' + pre_trained_model_step + '/interpolation')

                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()

                if pre_trained_model_step is not None:
                    print('Pretrianed model: ' + pre_trained_model_step)
                    try:
                        print('Load the model from: ' + 'output/' + pre_trained_model + '/model/')
                        print('Load iteration: ' + 'output/' + pre_trained_model + '/model/')
                        saver.restore(sess, 'output/' + pre_trained_model + '/model/' + pre_trained_model_step)
                    except Exception:
                        raise ValueError("Load model Failed!")
                
                zs = []
                steps = np.linspace(0, 1, no_steps)
                for step in steps:
                    # Latent space interpolation
                    z = v1*(1-step) + v2*step
                    zs.append(z)
                
                z = np.array(zs)
                np.save('output/' + cur_model_name + '/vis_test/' + pre_trained_model_step + '/interpolation.npy', z)
                
                output_train = sess.run([gen_out], feed_dict={z_vector:z})
                output_test = sess.run([gen_out_test], feed_dict={z_vector:z})                
    return(output_train, output_test)

def getExample(d_layers,
               g_layers,
               batch_size,
               no_batches,
               pixel_shape,
               d_lr,
               g_lr,
               beta,
               d_thresh,
               g_thresh,
               pro_growth,
               no_vis_mul = 4,
               pre_trained_models = None):
    
    tf.reset_default_graph()
    
    # Define input
    z_vector = tf.placeholder(shape=[batch_size, 200],dtype=tf.float32) 
    x_vector = tf.placeholder(shape=[batch_size, pixel_shape, pixel_shape, pixel_shape,1], dtype=tf.float32)
    
    # Define model
    gen_out = generator(z_vector, g_layers, batch_size, pro_growth, training=True)
    gen_out_test = generator(z_vector, g_layers, batch_size, pro_growth, training=False, reuse=True)
    dis_out, dis_out_no = dicriminator(x_vector, d_layers, batch_size, pro_growth, training=True)
    dis_gen_out, dis_gen_out_no = dicriminator(gen_out, d_layers, batch_size, pro_growth, training=True, reuse=True)
    
    # Get stats + loss
    nx = tf.reduce_sum(tf.cast(dis_out > 0.5, tf.int32))
    nz = tf.reduce_sum(tf.cast(dis_gen_out < 0.5, tf.int32))
    d_acc = tf.divide(nx + nz, 2 * batch_size)
    
    # Loss
    d_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_out_no, 
                                                     labels=tf.ones_like(dis_out_no)) + \
                tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_gen_out_no, labels=tf.zeros_like(dis_gen_out_no))
    g_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_gen_out_no, labels=tf.ones_like(dis_gen_out_no))
    d_loss = tf.reduce_mean(d_loss)
    g_loss = tf.reduce_mean(g_loss)

    # Optimizer
    gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN_GEN_")
    disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN_DIS_")
    optimizer_op_d = tf.train.AdamOptimizer(learning_rate=d_lr,beta1=beta).minimize(d_loss, var_list=disc_vars)
    optimizer_op_g = tf.train.AdamOptimizer(learning_rate=g_lr,beta1=beta).minimize(g_loss, var_list=gen_vars)

    # Visualization
    output_train = []
    output_test = []
    
    for pre_trained_model in pre_trained_models.keys():
        cur_model_name = pre_trained_model
        print('Model name: ' + cur_model_name)
        
        for pre_trained_model_step in pre_trained_models[pre_trained_model]:
            with tf.Session() as sess:
                #merge = tf.summary.merge_all()
                #writer = tf.summary.FileWriter("log/{}".format(cur_model_name), sess.graph)

                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()

                if pre_trained_model_step is not None:
                    print('Pretrianed model: ' + pre_trained_model_step)
                    try:
                        print('Load the model from: ' + 'output/' + pre_trained_model + '/model/')
                        print('Load iteration: ' + 'output/' + pre_trained_model + '/model/')
                        saver.restore(sess, 'output/' + pre_trained_model + '/model/' + pre_trained_model_step)
                    except Exception:
                        raise ValueError("Load model Failed!")
                
                no_vis = batch_size * no_vis_mul
                z_total = np.random.normal(0, 0.33, size=[no_vis, 200]).astype(np.float32)
                
                for i in range(no_vis_mul):
                    z = z_total[(0 + batch_size*i):((1+i)*batch_size)]
                    output_train_tmp = sess.run([gen_out], feed_dict={z_vector:z})
                    output_test_tmp = sess.run([gen_out_test], feed_dict={z_vector:z})
                    output_train = output_train + output_train_tmp
                    output_test = output_test + output_test_tmp
    return(output_train, output_test)
