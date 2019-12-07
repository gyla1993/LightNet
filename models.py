from keras.models import Model
from keras.layers import Input,ConvLSTM2D,TimeDistributed,concatenate, Add, Bidirectional,Concatenate, dot, add, multiply, Permute
from keras.layers.core import Activation, Reshape, Dense, RepeatVector, Dropout
from keras.layers.convolutional import Conv3D,Conv2D,SeparableConv2D,Cropping3D,Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers import Lambda
from keras.layers import regularizers
from generator import num_frames, fea_dim, wrf_fea_dim, num_frames_truth
import keras.backend as K

# LightNet
def LSTM_Conv2D_KDD():
    # encoder1: layers definition && data flow  --------------------------------------
    encoder1_inputs = Input(shape=(None, 159, 159, fea_dim), name='encoder1_inputs') # (bs, 6, 159, 159, fea_dim)
    encoder1_conv2d = TimeDistributed(Conv2D(filters=64,kernel_size=(7, 7),strides=(2, 2),padding='same'),name='en1_conv2d')(encoder1_inputs)
    encoder1_conv2d = TimeDistributed(Activation('relu'))(encoder1_conv2d)

    encoder1_convlstm,h1,c1 = ConvLSTM2D(filters=128, kernel_size=(5, 5),
                    return_state=True, padding='same', return_sequences=True,name='en1_convlstm')(encoder1_conv2d)
    # --------------------------------------------------------------------------------
    # encoder2: layers definition && data flow  --------------------------------------
    encoder2_inputs = Input(shape=(None, 159, 159, 1), name='encoder2_inputs')  # (bs, 3, 159, 159, 1)
    encoder2_conv2d = TimeDistributed(Conv2D(filters=4, kernel_size=(7, 7), strides=(2, 2), padding='same'),name='en2_conv2d')(encoder2_inputs)
    encoder2_conv2d = TimeDistributed(Activation('relu'))(encoder2_conv2d)

    encoder2_convlstm,h2,c2 = ConvLSTM2D(filters=8, kernel_size=(5, 5),
                    return_state=True, padding='same', return_sequences=True,name='en2_convlstm')(encoder2_conv2d)
    # --------------------------------------------------------------------------------
    # encoder to decoder: layers definition && data flow  --------------------

    h = concatenate([h1, h2],axis=-1)
    c = concatenate([c1, c2],axis=-1)
    h = Conv2D(filters=64, kernel_size=(1, 1), padding="same", name='h_conv2d', activation='relu')(h)
    c = Conv2D(filters=64, kernel_size=(1, 1), padding="same", name='c_conv2d', activation='relu')(c)
    # --------------------------------------------------------------------------------

    # decoder: layers definition -------------------------------
    _decoder_inputs = Input(shape=(None, 159, 159, 1), name='decoder_inputs')  # (bs, 1, 159,159,1)
    decoder_inputs = _decoder_inputs
    de_conv2d = TimeDistributed(Conv2D(filters=4, kernel_size=(7, 7), strides=(2, 2),padding='same'), name='de_conv2d')
    de_convlstm = ConvLSTM2D(filters=64, kernel_size=(5, 5), name='de_convlstm',
                            return_state=True, padding='same', return_sequences=True)
    de_conv2dT = TimeDistributed(Conv2DTranspose(filters=64, kernel_size=(7, 7), strides=(2, 2),padding='same'), name='de_conv2dT')
    de_out_conv2d = TimeDistributed(Conv2D(filters=1, kernel_size=(1, 1), padding="same"), name='de_out_conv2d')
    # ----------------------------------------------------------

    relu = Activation('relu')
    sigmoid = Activation('sigmoid')
    cropper = Cropping3D(cropping=((0, 0), (0, 1), (0, 1)))

    # decoder: data flow-----------------------------------------
    decoder_outputs = []
    for i in range(num_frames):
        decoder_conv2d = de_conv2d(decoder_inputs)
        decoder_conv2d = relu(decoder_conv2d)

        decoder_convlstm, h, c = de_convlstm([decoder_conv2d, h, c])

        decoder_conv2dT = de_conv2dT(decoder_convlstm)
        decoder_conv2dT = relu(decoder_conv2dT)

        decoder_out_conv2d = de_out_conv2d(decoder_conv2dT)  # (bs, 1, 160, 160, 1)
        decoder_output = cropper(decoder_out_conv2d)  # (bs, 1, 159, 159, 1)
        decoder_outputs.append(decoder_output)
        decoder_output = sigmoid(decoder_output)
        decoder_inputs = decoder_output

    decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(decoder_outputs)   # (bs, 6, 159, 159, 1)
    decoder_outputs = Reshape((-1, 159 * 159, 1), input_shape=(-1, 159, 159, 1))(decoder_outputs)
    # ----------------------------------------------------------
    return Model([encoder1_inputs, encoder2_inputs, _decoder_inputs], decoder_outputs, name='ConvLSTM-Conv2d-KDD')

# LightNet-W
def LSTM_Conv2D_KDD_t1():
    # encoder: layers definition && data flow  --------------------------------------
    encoder_inputs = Input(shape=(None, 159, 159, fea_dim), name='encoder_inputs') # (bs, 6, 159, 159, fea_dim)
    encoder_conv2d = TimeDistributed(Conv2D(filters=64,kernel_size=(7, 7),strides=(2, 2),padding='same'),name='en_conv2d')(encoder_inputs)
    encoder_conv2d = TimeDistributed(Activation('relu'))(encoder_conv2d)

    _,h,c = ConvLSTM2D(filters=128, kernel_size=(5, 5), return_sequences=True,
                    return_state=True, padding='same', name='en_convlstm')(encoder_conv2d)
    # --------------------------------------------------------------------------------
    # encoder to decoder: layers definition && data flow  ----------------------------
    h = Conv2D(filters=64, kernel_size=(1, 1), padding="same", name='h_conv2d', activation='relu')(h)
    c = Conv2D(filters=64, kernel_size=(1, 1), padding="same", name='c_conv2d', activation='relu')(c)
    # --------------------------------------------------------------------------------

    # decoder: layers definition -------------------------------
    _decoder_inputs = Input(shape=(None, 159, 159, 1), name='decoder_inputs')  # (bs, 1, 159,159,1)
    decoder_inputs = _decoder_inputs
    de_conv2d = TimeDistributed(Conv2D(filters=4, kernel_size=(7, 7), strides=(2, 2),padding='same'), name='de_conv2d')
    de_convlstm = ConvLSTM2D(filters=64, kernel_size=(5, 5), name='de_convlstm',
                            return_state=True, padding='same', return_sequences=True)
    de_conv2dT = TimeDistributed(Conv2DTranspose(filters=64, kernel_size=(7, 7), strides=(2, 2),padding='same'), name='de_conv2dT')
    de_out_conv2d = TimeDistributed(Conv2D(filters=1, kernel_size=(1, 1), padding="same"), name='de_out_conv2d')
    # ----------------------------------------------------------

    relu = Activation('relu')
    sigmoid = Activation('sigmoid')
    cropper = Cropping3D(cropping=((0, 0), (0, 1), (0, 1)))

    # decoder: data flow-----------------------------------------
    decoder_outputs = []
    for i in range(num_frames):
        decoder_conv2d = de_conv2d(decoder_inputs)
        decoder_conv2d = relu(decoder_conv2d)

        decoder_convlstm, h, c = de_convlstm([decoder_conv2d, h, c])

        decoder_conv2dT = de_conv2dT(decoder_convlstm)
        decoder_conv2dT = relu(decoder_conv2dT)

        decoder_out_conv2d = de_out_conv2d(decoder_conv2dT)  # (bs, 1, 160, 160, 1)
        decoder_output = cropper(decoder_out_conv2d)  # (bs, 1, 159, 159, 1)
        decoder_outputs.append(decoder_output)
        decoder_output = sigmoid(decoder_output)
        decoder_inputs = decoder_output

    decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(decoder_outputs)   # (bs, 6, 159, 159, 1)
    decoder_outputs = Reshape((-1, 159 * 159, 1), input_shape=(-1, 159, 159, 1))(decoder_outputs)
    # ----------------------------------------------------------
    return Model([encoder_inputs, _decoder_inputs], decoder_outputs, name='ConvLSTM-Conv2d-KDD-t1')

# LightNet-O
def LSTM_Conv2D_KDD_t2():
    # encoder: layers definition && data flow  --------------------------------------
    encoder_inputs = Input(shape=(None, 159, 159, 1), name='encoder_inputs')  # (bs, 3, 159, 159, 1)
    encoder_conv2d = TimeDistributed(Conv2D(filters=4, kernel_size=(7, 7), strides=(2, 2), padding='same'),
                                      name='en_conv2d')(encoder_inputs)
    encoder_conv2d = TimeDistributed(Activation('relu'))(encoder_conv2d)
    _, h, c = ConvLSTM2D(filters=8, kernel_size=(5, 5), return_sequences=True,
                                           return_state=True, padding='same', name='en_convlstm')(encoder_conv2d)
    # --------------------------------------------------------------------------------
    # encoder to decoder: layers definition && data flow  ----------------------------
    h = Conv2D(filters=64, kernel_size=(1, 1), padding="same", name='h_conv2d', activation='relu')(h)
    c = Conv2D(filters=64, kernel_size=(1, 1), padding="same", name='c_conv2d', activation='relu')(c)
    # --------------------------------------------------------------------------------

    # decoder: layers definition -------------------------------
    _decoder_inputs = Input(shape=(None, 159, 159, 1), name='decoder_inputs')  # (bs, 1, 159,159,1)
    decoder_inputs = _decoder_inputs
    de_conv2d = TimeDistributed(Conv2D(filters=4, kernel_size=(7, 7), strides=(2, 2),padding='same'), name='de_conv2d')
    de_convlstm = ConvLSTM2D(filters=64, kernel_size=(5, 5), name='de_convlstm',
                            return_state=True, padding='same', return_sequences=True)
    de_conv2dT = TimeDistributed(Conv2DTranspose(filters=64, kernel_size=(7, 7), strides=(2, 2),padding='same'), name='de_conv2dT')
    de_out_conv2d = TimeDistributed(Conv2D(filters=1, kernel_size=(1, 1), padding="same"), name='de_out_conv2d')
    # ----------------------------------------------------------

    relu = Activation('relu')
    sigmoid = Activation('sigmoid')
    cropper = Cropping3D(cropping=((0, 0), (0, 1), (0, 1)))

    # decoder: data flow-----------------------------------------
    decoder_outputs = []
    for i in range(num_frames):
        decoder_conv2d = de_conv2d(decoder_inputs)
        decoder_conv2d = relu(decoder_conv2d)

        decoder_convlstm, h, c = de_convlstm([decoder_conv2d, h, c])

        decoder_conv2dT = de_conv2dT(decoder_convlstm)
        decoder_conv2dT = relu(decoder_conv2dT)

        decoder_out_conv2d = de_out_conv2d(decoder_conv2dT)  # (bs, 1, 160, 160, 1)
        decoder_output = cropper(decoder_out_conv2d)  # (bs, 1, 159, 159, 1)
        decoder_outputs.append(decoder_output)
        decoder_output = sigmoid(decoder_output)
        decoder_inputs = decoder_output

    decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(decoder_outputs)   # (bs, 6, 159, 159, 1)
    decoder_outputs = Reshape((-1, 159 * 159, 1), input_shape=(-1, 159, 159, 1))(decoder_outputs)
    # ----------------------------------------------------------
    return Model([encoder_inputs, _decoder_inputs], decoder_outputs, name='ConvLSTM-Conv2d-KDD-t2')

# StepDeep
def Conv3D_KDD():
    WRF_inputs = Input(shape=(num_frames, 159, 159, fea_dim))   # (bs, 6, 159, 159, fea_dim)
    _history_inputs = Input(shape=(num_frames_truth, 159, 159, 1))  # (bs,3,159,159,1)
    # history_inputs = Lambda(lambda x: K.squeeze(x, axis=-1))(_history_inputs)   # (bs, 3, 159, 159)
    history_inputs = Permute((4, 2, 3, 1))(_history_inputs)              #  (bs, 1, 159, 159, 3)
    conv_1 = Conv3D(filters=128, kernel_size=(2, 1, 1), padding='same', name='conv3d_1')(WRF_inputs)
    conv_1 = Activation('relu')(conv_1)
    conv_2 = Conv3D(filters=128, kernel_size=(1, 3, 3), padding='same', name='conv3d_2')(conv_1)
    conv_2 = Activation('relu')(conv_2)
    conv_3 = Conv3D(filters=256, kernel_size=(2, 3, 3), padding='same', name='conv3d_3')(conv_2)
    conv_3 = Activation('relu')(conv_3)
    conv_4 = Conv3D(filters=128, kernel_size=(3, 1, 1), padding='same', name='conv3d_4')(conv_3)
    conv_4 = Activation('relu')(conv_4)
    conv_5 = Conv3D(filters=128, kernel_size=(1, 3, 3), padding='same', name='conv3d_5')(conv_4)
    conv_5 = Activation('relu')(conv_5)
    conv_6 = Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same', name='conv3d_6')(conv_5)
    conv_6 = Activation('relu')(conv_6)
    steps = []
    for i in range(num_frames):
        conv_6_i = Cropping3D(cropping=((i, num_frames - i - 1), (0, 0), (0, 0)))(conv_6)   # (bs, 1, 159, 159, 64)
        conv2d_in = concatenate([history_inputs, conv_6_i], axis=-1)                        # (bs, 1, 159, 159, 64+3)
        conv2d_in = Lambda(lambda x: K.squeeze(x, axis=1))(conv2d_in)  # (bs, 159, 159, 67)
        conv2d_1_i = Conv2D(filters=64, kernel_size=(7, 7), padding='same', name='conv2d_1_%d' % i)(conv2d_in)
        conv2d_1_i = Activation('relu')(conv2d_1_i)
        conv2d_2_i = Conv2D(filters=1, kernel_size=(7, 7), padding='same', name='conv2d_2_%d' % i)(conv2d_1_i)
        steps.append(conv2d_2_i)
    conv_out = concatenate(steps, axis=1)  # (bs, 6, 159, 159, 1)
    outputs = Reshape((-1, 159 * 159, 1), input_shape=(-1, 159, 159, 1))(conv_out)
    return Model([WRF_inputs, _history_inputs], outputs, name='Conv3D-KDD')
