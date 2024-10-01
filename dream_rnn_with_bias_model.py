import sys
import numpy as np
from tensorflow.keras.backend import int_shape
from tensorflow.keras.layers import Input, Cropping1D, add, Conv1D, GlobalAvgPool1D, Dense, Add, Concatenate, Lambda, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from chrombpnet.training.utils.losses import multinomial_nll
import tensorflow as tf
import random as rn
import os
from chrom_dreamnet.h5_mode.dream_rnn import *

os.environ['PYTHONHASHSEED'] = '0'

#### Took this code from chrombpnet #####
def load_pretrained_bias(model_hdf5):
    from tensorflow.keras.models import load_model
    from tensorflow.keras.utils import get_custom_objects
    custom_objects={"multinomial_nll":multinomial_nll, "tf":tf}
    get_custom_objects().update(custom_objects)
    pretrained_bias_model=load_model(model_hdf5)
    #freeze the model
    num_layers=len(pretrained_bias_model.layers)
    for i in range(num_layers):
        pretrained_bias_model.layers[i].trainable=False
    return pretrained_bias_model


def getModelGivenModelOptionsAndWeightInits(args, model_params):   
    
    assert("bias_model_path" in model_params.keys()) # bias model path not specfied for model
    filters=int(model_params['filters'])
    n_dil_layers=int(model_params['n_dil_layers'])
    counts_loss_weight=float(model_params['counts_loss_weight'])
    bias_model_path=model_params['bias_model_path']
    sequence_len=int(model_params['inputlen'])
    out_pred_len=int(model_params['outputlen'])


    bias_model = load_pretrained_bias(bias_model_path)
    dream_rnn_wo_bias = seq_nn_model(seqsize=sequence_len, target_bins = out_pred_len)

    #read in arguments
    seed=args.seed
    np.random.seed(seed)    
    tf.random.set_seed(seed)
    rn.seed(seed)
    
    inp = Input(shape=(sequence_len, 4),name='sequence')    

    ## get bias output
    bias_output=bias_model(inp)
    ## get wo bias output
    output_wo_bias=dream_rnn_wo_bias(inp)
    assert(len(bias_output[1].shape)==2) # bias model counts head is of incorrect shape (None,1) expected
    assert(len(bias_output[0].shape)==2) # bias model profile head is of incorrect shape (None,out_pred_len) expected
    assert(len(output_wo_bias[0].shape)==2)
    assert(len(output_wo_bias[1].shape)==2)
    assert(bias_output[1].shape[1]==1) #  bias model counts head is of incorrect shape (None,1) expected
    assert(bias_output[0].shape[1]==out_pred_len) # bias model profile head is of incorrect shape (None,out_pred_len) expected


    profile_out = Add(name="logits_profile_predictions")([output_wo_bias[0],bias_output[0]])
    concat_counts = Concatenate(axis=-1)([output_wo_bias[1], bias_output[1]])
    count_out = Lambda(lambda x: tf.math.reduce_logsumexp(x, axis=-1, keepdims=True),
                        name="logcount_predictions")(concat_counts)

    # instantiate keras Model with inputs and outputs
    model=Model(inputs=[inp],outputs=[profile_out, count_out])

    model.compile(optimizer=Adam(learning_rate=args.learning_rate),
                    loss=[multinomial_nll,'mse'],
                    loss_weights=[1,counts_loss_weight])
    print(model.summary())
    return model 


def save_model_without_bias(model, output_prefix):
   
    
    model_wo_bias = model.get_layer("seq_nn").output
    #counts_output_without_bias = model.get_layer("wo_bias_bpnet_logcount_predictions").output
    model_without_bias = Model(inputs=model.get_layer("seq_nn").inputs,outputs=[model_wo_bias[0], model_wo_bias[1]])
    print('save model without bias') 
    model_without_bias.save(output_prefix+"_nobias.h5")