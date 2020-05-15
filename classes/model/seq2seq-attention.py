from keras.models import Model, Sequential
from keras.layers import Bidirectional, Concatenate, Permute, dot, Input, LSTM, Multiply, Reshape
from keras.layers import RepeatVector, Dense, Activation, Lambda, Embedding
from keras import backend as K, callbacks
from keras.optimizers import Adam
import keras
import numpy as np

def softmax(x, axis=1):
    """
    Softmax activation function.
    """
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')


# Tx = max input seq len
def one_step_attention(a, s_prev, Tx=50):
    """
    Attention机制的实现，返回加权后的Context Vector
    
    @param a: BiRNN的隐层状态
    @param s_prev: Decoder端LSTM的上一轮隐层输出
    @param Tx:  max input seq len
    Returns:
    context: 加权后的Context Vector
    """
    repeator = RepeatVector(Tx)
    concatenator = Concatenate(axis=-1)
    densor_tanh = Dense(32, activation = "tanh")
    densor_relu = Dense(1, activation = "relu")
    activator = Activation(K.softmax, name='attention_weights')
    dotor = Dot(axes = 1)
    
    # 将s_prev复制Tx次
    s_prev = repeator(s_prev)
    # 拼接BiRNN隐层状态与s_prev
    concat = concatenator([a, s_prev])
    # 计算energies
    e = densor_tanh(concat)
    energies = densor_relu(e)
    # 计算weights
    alphas = activator(energies)
    # 加权得到Context Vector
    context = dotor([alphas, a])
    
    return context

def attention_biLSTM_train_model(Tx, Ty, n_a, n_s, num_input_token, num_target_token):
    """
    构造模型
    
    @param Tx: 输入序列的长度
    @param Ty: 输出序列的长度
    @param n_a: Encoder端Bi-LSTM隐层结点数
    @param n_s: Decoder端LSTM隐层结点数
    @param num_input_token: 
    @param num_target_token: 
    """

    encoder_inputs = Input(shape=(None, num_input_token), name="encoder_input")
    decoder_inputs = Input(shape=(None, num_target_token), name="decoder_input")


def biLSTM_attention_predit(model, seq):
    while not stop_condition:
    for i in range(1, OUTPUT_LENGTH):
        output = model.predict([encoder_input, decoder_input]).argmax(axis=2)
        decoder_input[:,i] = output[:,i]
    return decoder_input[:,1:]