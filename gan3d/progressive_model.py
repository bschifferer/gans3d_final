import tensorflow as tf

def upBlock(x, no_filters, padding, strides, bl_training, idx, reuse=False):
    with tf.variable_scope("UP_BLOCK_" + str(idx), reuse=reuse):
        x = tf.layers.conv3d_transpose(x,
                                       filters = no_filters,
                                       kernel_size = [4,4,4],
                                       padding = padding,
                                       strides = strides,
                                       kernel_initializer = tf.contrib.layers.xavier_initializer(),
                                       bias_initializer = tf.contrib.layers.xavier_initializer(),
                                       name = "UP_3DCONV_" + str(idx)
                                      )
        x = tf.layers.batch_normalization(x, name="UP_BATCHNORM_" + str(idx), training = bl_training)
        x = tf.nn.relu(x, name="UP_RELU_" + str(idx))
    return(x)

def downBlock(x, no_filters, padding, bl_training, idx, reuse=False):
    with tf.variable_scope("DOWN_BLOCK_" + str(idx), reuse=reuse):
        x = tf.layers.conv3d(x,
                             filters = no_filters,
                             kernel_size = [4,4,4],
                             padding = padding,
                             strides = (2, 2, 2),
                             kernel_initializer = tf.contrib.layers.xavier_initializer(),
                             bias_initializer = tf.contrib.layers.xavier_initializer(),
                             name = "DOWN_3DCONV_" + str(idx)
                            )
        x = tf.layers.batch_normalization(x, training = bl_training, name="DOWN_BATCHNORM_" + str(idx))
        x = tf.nn.leaky_relu(x, name="DOWN_RELU_" + str(idx))
    return(x)

def generator(z_vector, layers, batch_size, pro_growth, training=False, reuse=False):    
    x = tf.reshape(z_vector, (batch_size, 1, 1, 1, 200))
    with tf.variable_scope("GAN_GEN_", reuse=reuse):
        for idx, layer in enumerate(layers):
            x = upBlock(x, layer['no_filters'], layer['padding'], layer['strides'], training, idx, reuse)
        x = tf.layers.conv3d_transpose(x,
                                       filters = 1,
                                       kernel_size = [4,4,4],
                                       padding = 'same',
                                       strides = (2, 2, 2),
                                       kernel_initializer = tf.contrib.layers.xavier_initializer(),
                                       bias_initializer = tf.contrib.layers.xavier_initializer(),
                                       name="UP_3DCONV_FINAL_" + str(pro_growth)
                                      )
        x = tf.nn.sigmoid(x, name="UP_SIGMOID_FINAL")
    return(x)

def dicriminator(x, layers, batch_size, pro_growth, training=False, reuse=False):
    with tf.variable_scope("GAN_DIS_",reuse=reuse):
        for idx, layer in enumerate(layers):
            x = downBlock(x, layer['no_filters'], layer['padding'], training, idx, reuse)
        x = tf.layers.conv3d(x,
                             filters = 1,
                             kernel_size = [4,4,4],
                             padding = 'valid',
                             strides = (1, 1, 1),
                             kernel_initializer = tf.contrib.layers.xavier_initializer(),
                             bias_initializer = tf.contrib.layers.xavier_initializer(),
                             name="DOWN_3DCONV_FINAL_" + str(pro_growth)
                            )
        x_no = x
        x = tf.nn.sigmoid(x_no, name="DOWN_SIGMOID_FINAL")
        x_no = tf.reshape(x_no, (batch_size, 1))
        x = tf.reshape(x, (batch_size, 1))  
    return(x, x_no)


