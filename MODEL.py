import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

kernel_size = 3
f = 64
n_res_blocks=3
n_res_groups=5

weights = []
w_init = tf.variance_scaling_initializer(scale=1., mode='fan_avg', distribution="uniform")
b_init = tf.zeros_initializer()




def conv2d(x, f_in, f_out, k, name):    
    conv_w = tf.get_variable(name + "_w" , [k,k,f_in,f_out], initializer=w_init)
    conv_b = tf.get_variable(name + "_b" , [f_out], initializer=b_init)
    weights.append(conv_w)
    weights.append(conv_b)
    return tf.nn.bias_add(tf.nn.conv2d(x, conv_w, strides=[1,1,1,1], padding='SAME'), conv_b)                           

            
            
def residual_block(x, name):
    skip_conn = x  
    conv_name = "conv2d-1" + "_" + name
    x = conv2d(x, f_in=f, f_out=f, k=kernel_size, name=conv_name)
    x = tf.nn.relu(x)
    conv_name = "conv2d-2" + "_" + name
    x = conv2d(x, f_in=f, f_out=f, k=kernel_size, name=conv_name)

    return tf.add(x , skip_conn)
    
    
    
    
def residual_group(x, name):
    x = conv2d(x, f_in=f, f_out=f, k=kernel_size, name="residual_group-head" + name)
    skip_conn = x

    for i in range(n_res_blocks):
        x = residual_block(x, name=name + "_" + str(i))
        
    conv_name = "rg-conv-" + name
    x = conv2d(x, f_in=f, f_out=f, k=kernel_size, name=conv_name)
    return tf.add(x , skip_conn) 
            
            
            
            
            
            
def residual_channel_attention_network(x):
    x = conv2d(x, f_in=1, f_out=f, k=kernel_size, name="conv2d-head_0")
    head = x

    x = head
    for i in range(n_res_groups):
        x = residual_group(x, name=str(i) )

    body = conv2d(x, f_in=f, f_out=f, k=kernel_size, name="conv2d-body")
    body = tf.add(body , head)
    tail = conv2d(body, f_in=f, f_out=1, k=kernel_size, name="conv2d-tail")  

    return tail
                      
def model(input_tensor):
    with tf.device("/gpu:0"):
        tensor = None
        tensor = residual_channel_attention_network(input_tensor)
        
        # Patch size should be multiple of 7, or used only at inference time
        # tensor = tf.space_to_depth(tensor, block_size=7)
        # inputTensor = tf.space_to_depth(input_tensor, block_size=7)
        # tensor = tf.concat( [tf.expand_dims(inputTensor[:,:,:,0],axis=3), tensor[:,:,:,1:3], tf.expand_dims(inputTensor[:,:,:,3],axis=3), tensor[:,:,:,4:6], tf.expand_dims(inputTensor[:,:,:,6],axis=3), tensor[:,:,:,7:21], tf.expand_dims(inputTensor[:,:,:,21],axis=3), tensor[:,:,:,22:24], tf.expand_dims(inputTensor[:,:,:,24],axis=3), tensor[:,:,:,25:27], tf.expand_dims(inputTensor[:,:,:,27],axis=3), tensor[:,:,:,28:42], tf.expand_dims(inputTensor[:,:,:,42],axis=3), tensor[:,:,:,43:45], tf.expand_dims(inputTensor[:,:,:,45],axis=3), tensor[:,:,:,46:48], tf.expand_dims(inputTensor[:,:,:,48],axis=3)], axis=3)
        # print(tensor.shape)
        # tensor = tf.depth_to_space(tensor, block_size=7)

		
        return tensor, weights

