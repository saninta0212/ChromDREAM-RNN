import sys
import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer, Concatenate, AveragePooling1D
from tensorflow.keras import Model
from tensorflow.keras.activations import sigmoid, linear
from tensorflow import transpose, matmul, expand_dims
from tensorflow import keras
from tensorflow.keras.layers import Input, Cropping1D, add, Conv1D, GlobalAvgPool1D, Dense, Flatten, MaxPooling1D, Dropout

import tensorflow as tf
from tensorflow.keras import layers as layers
from tensorflow.keras import Model
from tensorflow.keras.activations import swish
    
        
def seq_nn_model(seqsize, in_ch=4, stem_ch=256, stem_ks = 7, target_bins=64, num_tasks = 1) :
    inputs = Input(shape=(seqsize, in_ch), name='input')
    
    
    # First Layers Block
    dropout=0.2
    out_channels=320
    seqsize=seqsize
    kernel_sizes=[9, 15] 
    pool_size=1
    assert out_channels % len(kernel_sizes) == 0, "out_channels must be divisible by the number of kernel sizes"
    each_out_channels = out_channels // len(kernel_sizes)
    
    conv1 = Conv1D(filters=each_out_channels, kernel_size=kernel_sizes[0], padding='same', activation='relu')(inputs)
    conv1 = MaxPooling1D(pool_size=pool_size, strides=pool_size)(conv1)
    conv1 = Dropout(rate=dropout)(conv1)
    
    conv2 = Conv1D(filters=each_out_channels, kernel_size=kernel_sizes[1], padding='same', activation='relu')(inputs)
    conv2 = MaxPooling1D(pool_size=pool_size, strides=pool_size)(conv2)
    conv2 = Dropout(rate=dropout)(conv2)
    
    x = tf.concat([conv1, conv2], axis=-1)
    
    # Core Layers Block
    out_channels=320
    seqsize=seqsize
    lstm_hidden_channels=320
    kernel_sizes=[9, 15]
    pool_size=1
    dropout1=0.2
    dropout2=0.5
    assert out_channels % len(kernel_sizes) == 0, "out_channels must be divisible by the number of kernel sizes"
    each_conv_out_channels = out_channels // len(kernel_sizes)
    
    lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_hidden_channels, return_sequences=True))(x)
    
    conv1 = Conv1D(filters=each_conv_out_channels, kernel_size=kernel_sizes[0], padding='same', activation='relu')(lstm)
    conv1 = MaxPooling1D(pool_size=pool_size, strides=pool_size)(conv1)
    conv1 = Dropout(rate=dropout1)(conv1)
    
    conv2 = Conv1D(filters=each_conv_out_channels, kernel_size=kernel_sizes[1], padding='same', activation='relu')(lstm)
    conv2 = MaxPooling1D(pool_size=pool_size, strides=pool_size)(conv2)
    conv2 = Dropout(rate=dropout1)(conv2)
    
    x = tf.concat([conv1, conv2], axis=-1)
    core_block_output = Dropout(rate=dropout2)(x)
    
    
    # Cropping acc to target lengths
    prof_out_precrop = Conv1D(filters=num_tasks, kernel_size=75,padding='valid', name='wo_bias_bpnet_prof_out_precrop')(core_block_output)
    seq_len, target_len = prof_out_precrop.shape[-2], target_bins

    if target_len == -1:
        return prof_out_precrop

    if seq_len < target_len:
        raise ValueError(f'sequence length {seq_len} is less than target length {target_len}')

    trim = (target_len - seq_len) // 2

    if trim == 0:
        prof = prof_out_precrop
        
    # For profile

    prof = prof_out_precrop[:, -trim:trim, :]
    profile_out = Flatten(name="wo_bias_bpnet_logits_profile_predictions")(prof)
    
    # For counts
    gap_combined_conv = GlobalAvgPool1D(name='gap')(core_block_output)
    count_out = Dense(num_tasks, name="wo_bias_bpnet_logcount_predictions")(gap_combined_conv)

    model=Model(inputs=[inputs], outputs=[profile_out, count_out], name="seq_nn")
    print(model.summary)

    return model
    

    