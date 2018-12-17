from gan3d.progressive_model import generator
from gan3d.progressive_model import dicriminator
from gan3d.utils import plotLineGraph
from gan3d.utils import visImage
from gan3d.utils import createPath

import time
import tensorflow as tf
import numpy as np

def training(volumes,
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
             model_name,
             pro_growth,
             pre_trained_model = None,
             pre_trained_model_version = None,
             use_timestamp = True,
             no_models = 5,
             full_load_pre_trained = False
            ):
    
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
    
    load_gen_vars = [tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GAN_GEN_/UP_BLOCK_" + str(i)) for i in range(pro_growth)]
    load_disc_vars = [tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GAN_DIS_/DOWN_BLOCK_" + str(i)) for i in range(pro_growth)]
    
    if not(full_load_pre_trained):
        load_gen_vars = [y for x in load_gen_vars for y in x]
        load_disc_vars = [y for x in load_disc_vars for y in x]
        load_var = load_gen_vars + load_disc_vars

    # Training
    createPath('output/')
    if use_timestamp:
        cur_model_name_time = model_name + '_' + str(int(time.time()))
    else:
        cur_model_name_time = model_name
    
    for abc in range(no_models):
        if abc > 0:
            cur_model_name = cur_model_name_time + '_' + str(abc)
        else:
            cur_model_name = cur_model_name_time

        print('Model name: ' + cur_model_name)

        createPath('output/' + cur_model_name + '/vis/')
        createPath('output/' + cur_model_name + '/model/')
        createPath('output/' + cur_model_name + '/loss/')

        hist_d_loss = []
        hist_g_loss = []
        hist_d_acc = []
        hist_d_acc_var = []
        hist_g_gen_loss_unique = []

        with tf.Session() as sess:
            #merge = tf.summary.merge_all()
            #writer = tf.summary.FileWriter("log/{}".format(cur_model_name), sess.graph)

            sess.run(tf.global_variables_initializer())
            if not(full_load_pre_trained):
                loader = tf.train.Saver(max_to_keep=100, var_list=load_var)
            else:
                loader = tf.train.Saver(max_to_keep=100)
            saver = tf.train.Saver(max_to_keep=100)

            if pre_trained_model is not None:
                print('Pretrianed model: ' + pre_trained_model)
                try:
                    print('Load the model from: ' + 'output/' + pre_trained_model + '/model/' + pre_trained_model_version)
                    print('Load iteration: ' + 'output/' + pre_trained_model + '/model/')
                    loader.restore(sess, 'output/' + pre_trained_model + '/model/' + pre_trained_model_version)
                    open('output/' + cur_model_name + '/' + pre_trained_model_version, 'a').close()
                except Exception:
                    raise ValueError("Load model Failed!")

            # Epochs are iterations
            for epoch in range(no_batches):
                print(epoch)

                # Getting inputs
                idx = np.random.randint(len(volumes), size=batch_size)
                x = volumes[idx]
                z = np.random.normal(0, 0.33, size=[batch_size, 200]).astype(np.float32)

                # Getting loss and accuracy
                discriminator_loss, generator_loss, d_accuracy, n_x, n_z = sess.run([d_loss, g_loss, d_acc, nx, nz], feed_dict={z_vector:z, x_vector:x})

                hist_d_loss = hist_d_loss + [discriminator_loss]
                hist_g_loss = hist_g_loss + [generator_loss]
                hist_d_acc = hist_d_acc + [d_accuracy]
                if epoch > 20:
                    d_acc_var = np.var(list(reversed(hist_d_acc))[0:20])
                    d_gen_loss_unique = np.unique(np.round(list(reversed(hist_g_loss))[0:20],3)).shape[0]
                else:
                    d_acc_var = 0
                    d_gen_loss_unique = 0
                hist_d_acc_var = hist_d_acc_var + [d_acc_var]
                hist_g_gen_loss_unique = hist_g_gen_loss_unique + [d_gen_loss_unique]

                print(d_accuracy)

                # Train discriminator
                if d_accuracy < d_thresh:
                    sess.run([optimizer_op_d],feed_dict={z_vector:z, x_vector:x})
                    print('Discriminator Training ', "epoch: ", epoch,', d_loss:', discriminator_loss,'g_loss:', generator_loss, "d_acc: ", d_accuracy)

                # Train generator
                if d_accuracy > g_thresh:
                    sess.run([optimizer_op_g],feed_dict={z_vector:z})
                    print('Generator Training ', "epoch: ",epoch,', d_loss:', discriminator_loss,'g_loss:', generator_loss, "d_acc: ", d_accuracy)

                # Visualize and safe
                if epoch % 200 == 0 and epoch > 0:
                    #merge_result  = sess.run([merge], feed_dict={z_vector:z, x_vector:x})
                    #writer.add_summary(merge_result, epoch)
                    saver.save(sess, 'output/' + cur_model_name + '/model/' + cur_model_name + '_' + str(epoch))

                    plotLineGraph({'D_Loss': hist_d_loss,
                                  'G_Loss': hist_g_loss}, 'output/' + cur_model_name + '/loss/' + '01_loss' + str(epoch) + '.png')

                    plotLineGraph({'D_Acc': hist_d_acc}, 'output/' + cur_model_name + '/loss/' + '01_acc' + str(epoch) + '.png')
                    plotLineGraph({'D_Var_Acc': hist_d_acc_var}, 'output/' + cur_model_name + '/loss/' + '01_var_acc' + str(epoch) + '.png')
                    plotLineGraph({'G_LOSS_UNIQUE': hist_g_gen_loss_unique}, 'output/' + cur_model_name + '/loss/' + '01_g_loss_unique' + str(epoch) + '.png')

                    z = np.random.normal(0, 0.33, size=[batch_size, 200]).astype(np.float32)
                    output = sess.run([gen_out_test], feed_dict={z_vector:z})
                    for i in range(5):
                        visImage(output[0][i], 'output/' + cur_model_name + '/vis/' + cur_model_name + '_' + str(epoch) + '_' + str(i) + '_')
                        visImage(output[0][i]>0.9, 'output/' + cur_model_name + '/vis/' + cur_model_name + '_' + str(epoch) + '_' + str(i) + '_treshhold09')

                    output = sess.run([gen_out], feed_dict={z_vector:z})
                    for i in range(5):
                        visImage(output[0][i], 'output/' + cur_model_name + '/vis/GEN_OUT_' + cur_model_name + '_' + str(epoch) + '_' + str(i) + '_')
                        visImage(output[0][i]>0.9, 'output/' + cur_model_name + '/vis/GEN_OUT_' + cur_model_name + '_' + str(epoch) + '_' + str(i) + '_treshhold09')




def training_jupyter(volumes,
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
                     model_name,
                     pro_growth,
                     pre_trained_model = None,
                     pre_trained_model_version = None,
                     use_timestamp = True,
                     no_models = 5,
                     full_load_pre_trained = False
                    ):
    
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
    
    load_gen_vars = [tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GAN_GEN_/UP_BLOCK_" + str(i)) for i in range(pro_growth)]
    load_disc_vars = [tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GAN_DIS_/DOWN_BLOCK_" + str(i)) for i in range(pro_growth)]
    
    if not(full_load_pre_trained):
        load_gen_vars = [y for x in load_gen_vars for y in x]
        load_disc_vars = [y for x in load_disc_vars for y in x]
        load_var = load_gen_vars + load_disc_vars

    # Training
    createPath('output/')
    if use_timestamp:
        cur_model_name_time = model_name + '_' + str(int(time.time()))
    else:
        cur_model_name_time = model_name
    
    for abc in range(no_models):
        if abc > 0:
            cur_model_name = cur_model_name_time + '_' + str(abc)
        else:
            cur_model_name = cur_model_name_time

        print('Model name: ' + cur_model_name)

        createPath('output/' + cur_model_name + '/vis/')
        createPath('output/' + cur_model_name + '/model/')
        createPath('output/' + cur_model_name + '/loss/')

        hist_d_loss = []
        hist_g_loss = []
        hist_d_acc = []
        hist_d_acc_var = []
        hist_g_gen_loss_unique = []

        with tf.Session() as sess:
            #merge = tf.summary.merge_all()
            #writer = tf.summary.FileWriter("log/{}".format(cur_model_name), sess.graph)

            sess.run(tf.global_variables_initializer())
            if not(full_load_pre_trained):
                loader = tf.train.Saver(max_to_keep=100, var_list=load_var)
            else:
                loader = tf.train.Saver(max_to_keep=100)
            saver = tf.train.Saver(max_to_keep=100)

            if pre_trained_model is not None:
                print('Pretrianed model: ' + pre_trained_model)
                try:
                    print('Load the model from: ' + 'output/' + pre_trained_model + '/model/' + pre_trained_model_version)
                    print('Load iteration: ' + 'output/' + pre_trained_model + '/model/')
                    loader.restore(sess, 'output/' + pre_trained_model + '/model/' + pre_trained_model_version)
                    open('output/' + cur_model_name + '/' + pre_trained_model_version, 'a').close()
                except Exception:
                    raise ValueError("Load model Failed!")

            # Epochs are iterations
            for epoch in range(no_batches):
                print(epoch)

                # Getting inputs
                idx = np.random.randint(len(volumes), size=batch_size)
                x = volumes[idx]
                z = np.random.normal(0, 0.33, size=[batch_size, 200]).astype(np.float32)

                # Getting loss and accuracy
                discriminator_loss, generator_loss, d_accuracy, n_x, n_z = sess.run([d_loss, g_loss, d_acc, nx, nz], feed_dict={z_vector:z, x_vector:x})

                hist_d_loss = hist_d_loss + [discriminator_loss]
                hist_g_loss = hist_g_loss + [generator_loss]
                hist_d_acc = hist_d_acc + [d_accuracy]
                if epoch > 20:
                    d_acc_var = np.var(list(reversed(hist_d_acc))[0:20])
                    d_gen_loss_unique = np.unique(np.round(list(reversed(hist_g_loss))[0:20],3)).shape[0]
                else:
                    d_acc_var = 0
                    d_gen_loss_unique = 0
                hist_d_acc_var = hist_d_acc_var + [d_acc_var]
                hist_g_gen_loss_unique = hist_g_gen_loss_unique + [d_gen_loss_unique]

                print(d_accuracy)

                # Train discriminator
                if d_accuracy < d_thresh:
                    sess.run([optimizer_op_d],feed_dict={z_vector:z, x_vector:x})
                    print('Discriminator Training ', "epoch: ", epoch,', d_loss:', discriminator_loss,'g_loss:', generator_loss, "d_acc: ", d_accuracy)

                # Train generator
                if d_accuracy > g_thresh:
                    sess.run([optimizer_op_g],feed_dict={z_vector:z})
                    print('Generator Training ', "epoch: ",epoch,', d_loss:', discriminator_loss,'g_loss:', generator_loss, "d_acc: ", d_accuracy)

                # Visualize and safe
                if epoch % 200 == 0 and epoch > 0:
                    #merge_result  = sess.run([merge], feed_dict={z_vector:z, x_vector:x})
                    #writer.add_summary(merge_result, epoch)
                    saver.save(sess, 'output/' + cur_model_name + '/model/' + cur_model_name + '_' + str(epoch))

                    plotLineGraph({'D_Loss': hist_d_loss,
                                  'G_Loss': hist_g_loss}, 'output/' + cur_model_name + '/loss/' + '01_loss' + str(epoch) + '.png')

                    plotLineGraph({'D_Acc': hist_d_acc}, 'output/' + cur_model_name + '/loss/' + '01_acc' + str(epoch) + '.png')
                    plotLineGraph({'D_Var_Acc': hist_d_acc_var}, 'output/' + cur_model_name + '/loss/' + '01_var_acc' + str(epoch) + '.png')
                    plotLineGraph({'G_LOSS_UNIQUE': hist_g_gen_loss_unique}, 'output/' + cur_model_name + '/loss/' + '01_g_loss_unique' + str(epoch) + '.png')

                    z = np.random.normal(0, 0.33, size=[batch_size, 200]).astype(np.float32)
                    output = sess.run([gen_out_test], feed_dict={z_vector:z})
                    for i in range(5):
                        visImage(output[0][i], 'output/' + cur_model_name + '/vis/' + cur_model_name + '_' + str(epoch) + '_' + str(i) + '_')
                        visImage(output[0][i]>0.9, 'output/' + cur_model_name + '/vis/' + cur_model_name + '_' + str(epoch) + '_' + str(i) + '_treshhold09')

                    output = sess.run([gen_out], feed_dict={z_vector:z})
                    for i in range(5):
                        visImage(output[0][i], 'output/' + cur_model_name + '/vis/GEN_OUT_' + cur_model_name + '_' + str(epoch) + '_' + str(i) + '_')
                        visImage(output[0][i]>0.9, 'output/' + cur_model_name + '/vis/GEN_OUT_' + cur_model_name + '_' + str(epoch) + '_' + str(i) + '_treshhold09')
    return (hist_d_loss, hist_g_loss, hist_d_acc, hist_d_acc_var, hist_g_gen_loss_unique)
