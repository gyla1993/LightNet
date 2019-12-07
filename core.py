# -*- coding: utf-8 -*-
from keras.models import Model, load_model
from keras import optimizers
from genetator import DataGenerator, PredictDataGenerator, \
          getTimePeriod,ncFileDir_2016,ncFileDir_2017,M,npyWRFFileDir, \
         getHoursGridFromNPY,  num_frames, getHoursNCLGridFromTXT, getHoursNCLGridFromNPY, ncl_layers, \
         param_list,use_zscore, apply_zscore, fea_dim \
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score
from scores import cal_7_scores_0_6h, cal_7_scores_0_6h_neighborhood
import os
import numpy as np
import datetime
from keras import backend as K
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from keras.callbacks import Callback
from keras import losses
from plot import fig_single_timeperoid, Heatmap_single_timeperoid
from keras.callbacks import TensorBoard, ModelCheckpoint,LearningRateScheduler
import time as TI
import models

modelfileDir = 'models/'

def POD(y_true, y_pred):
    ytrue = y_true
    ypred = K.sigmoid(y_pred)
    ypred = K.round(ypred)
    true_positives = K.sum(ytrue * ypred)
    possible_positives = K.sum(ytrue)
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def FAR(y_true, y_pred):
    ytrue = y_true
    ypred = K.sigmoid(y_pred)
    ypred = K.round(ypred)
    true_positives = K.sum(ytrue * ypred)
    predicted_positives = K.sum(ypred)
    precision = true_positives / (predicted_positives + K.epsilon())
    return 1 - precision

def weight_loss(y_true,y_pred):  # binary classification
    pw = 16
    ytrue = K.flatten(y_true)
    ypred = K.flatten(y_pred)
    return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=ypred,targets=ytrue,pos_weight=pw))

def MSE(y_true,y_pred):
    y_pred = K.sigmoid(y_pred)
    return K.mean(K.square(y_pred - y_true), axis=-1)

def binary_acc(y_true,y_pred):
    ypred = K.sigmoid(y_pred)
    return K.mean(K.equal(y_true, K.round(ypred)), axis=-1)

class RecordMetricsAfterEpoch(Callback):
    def on_epoch_end(self, epoch, logs={}):
        filename = modelrecordname
        with open('records/' + filename + '.txt','a') as f:
            f.write('epoch %d:\r\n' % (epoch+1))
            for key in ['loss','POD','FAR','binary_acc','val_loss','val_POD','val_FAR','val_binary_acc']:
                f.write('%s: %f   ' % (key, logs[key]))
            f.write('\r\n')

class PrintTimeElapsedperBatch(Callback):
    t=0
    def on_batch_begin(self, batch, logs=None):
        self.t = TI.time()
    def on_batch_end(self, batch, logs=None):
        print('\n')
        print(TI.time()-self.t, end='')

def DoTrain(train_list, val_list):
    # parameters
    train_batchsize = 4
    val_batchsize = 4
    class_num = 2
    epochs_num = 50
    initial_epoch_num = 0

    train_gen = DataGenerator(train_list, train_batchsize, class_num, generator_type='train')
    val_gen = DataGenerator(val_list, val_batchsize, class_num, generator_type='val')

    # when train a new model --------------------------------------------

    # model = models.LSTM_Conv2D_KDD()
    # model = models.LSTM_Conv2D_KDD_t1()
    # model = models.LSTM_Conv2D_KDD_t2()
    model = models.Conv3D_KDD()
    # model = models.LSTM_Conv2D_KDD_v2()
    # model = models.LSTM_Conv2D_KDD_t1_v2()
    # model = models.LSTM_Conv2D_KDD_t2_v2()


    dt_now = datetime.datetime.now().strftime('%Y%m%d%H%M')
    print(dt_now)
    adam = optimizers.adam(lr=0.0001)

    model.compile(
                   # loss='categorical_crossentropy',
                   loss = weight_loss,
                   optimizer = adam,
                   #  optimizer=rmsprop,
                   metrics=[POD,FAR,binary_acc])
        #            metrics = [MSE])
    modelfilename = "%s-%s-{epoch:02d}.hdf5" % (dt_now, model.name)


    global modelrecordname
    modelrecordname = dt_now + '_' + model.name

    checkpoint = ModelCheckpoint(modelfileDir + modelfilename, monitor='val_loss', verbose=1,
                                 save_best_only=False, mode='min')
    RMAE = RecordMetricsAfterEpoch()
    hist = model.fit_generator(train_gen,
                                validation_data=val_gen,
                                epochs=epochs_num,
                                initial_epoch=initial_epoch_num,
                                # use_multiprocessing=True,
                                workers=3,
                                # max_queue_size=20,
                                callbacks=[checkpoint,RMAE]
                                # callbacks = [RMAE]
                              )
    print(hist.history)

def DoTest_step_seq(test_list, model, modelfilepath, testset_disp):
    # --------------------------------------------------
    test_batchsize = 1
    M = 1
    test_gen = PredictDataGenerator(test_list, test_batchsize)
    print('generating test data and predicting...')
    ypred = model.predict_generator(test_gen, workers=5, verbose=1)  # [len(test_list),num_frames,159*159,1]
    # ypred = 1.0 / (1.0 + np.exp(-ypred))  # only for Conv3d models, in which No Sigmoid layer is contained.
    # plot (the prediction for 6/12 timesteps)  ------------------------------------
    with tf.device('/cpu:0'):
        for id, ddt_item in enumerate(test_list):
            ddt = datetime.datetime.strptime(ddt_item, '%Y%m%d%H%M')
            utc = ddt + datetime.timedelta(hours=-8)  # convert Beijing time into UTC time
            ft = utc + datetime.timedelta(hours=(-6) * M)
            nchour, delta_hour = getTimePeriod(ft)
            delta_hour += M * 6
            y_pred = ypred[id]     # [num_frames,159*159,1]
            for hour_plus in range(num_frames):
                y_pred_i = y_pred[hour_plus]
                dt = ddt + datetime.timedelta(hours=hour_plus)
                dt_item = dt.strftime('%Y%m%d%H%M')
                resDir = 'results/%s_set%s/' % (modelfilepath, testset_disp)
                if not os.path.isdir(resDir):
                    os.makedirs(resDir)
                with open(resDir + '%s_h%d' % (dt_item, hour_plus), 'w') as rfile:
                    for i in range(159 * 159):
                        rfile.write('%f\r\n' % y_pred_i[i])  # the probability value
                # print(dt_item)

if __name__ == "__main__":

    # mode = 'TRAIN'
    mode = 'TEST'

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True


    sess = tf.Session(config=config)
    KTF.set_session(sess)


    TrainSetFilePath = 'train_lite_new.txt'
    ValSetFilePath = 'July.txt'
    TestSetFilePath = '20170809_6n.txt'

    testset_disp = '20170809_6n'

    if mode == 'TRAIN':
        train_list = []
        with open(TrainSetFilePath, 'r') as file:
            for line in file:
                train_list.append(line.rstrip('\n'))
        val_list = []
        with open(ValSetFilePath, 'r') as file:
            for line in file:
                val_list.append(line.rstrip('\n'))
        DoTrain(train_list, val_list)

    elif mode == 'TEST':
        test_list = []
        with open(TestSetFilePath, 'r') as file:
            for line in file:
                test_list.append(line.rstrip('\n'))


        for i in [16]:
            # modelfilepath = '201901131536-Conv3D-KDD-%s.hdf5' % str(i).zfill(2)
            modelfilepath = '201901202105-ConvLSTM-Conv2d-KDD-%s.hdf5' % str(i).zfill(2)

            trained_model = load_model(modelfileDir + modelfilepath, {'weight_loss': weight_loss, 'POD': POD, 'FAR': FAR,
                            'binary_acc': binary_acc, 'num_frames': num_frames})
            model = models.PredModel_LSTM_Conv2D_KDD(trained_model)

            DoTest_step_seq(test_list, trained_model, modelfilepath, testset_disp)
            resultfolderpath = modelfilepath + '_set%s' % testset_disp
            scores.eva(resultfolderpath, 0.5)
    sess.close()