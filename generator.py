# -*- coding: utf-8 -*-
import numpy as np
import datetime
import os
import keras
import math
import time as T
import tensorflow as tf
import random

ncFileDir_2017 = '/data/media/wrf-2017/'
ncFileDir_2016 = '/data/media-2016/'
npyWRFFileDir = '/data/wrf_npy_6-24/'
GuiTruthOriGridDir = '/data/DATA/guishan_grid_4x4/'
npyNCLFileDir = '/data/ncl_npy/'

variables3d = ['U', 'V', 'W', 'T', 'P','QVAPOR','QCLOUD','QRAIN','QICE','QHAIL',
               'QGRAUP','QSNOW','QEI','QEG','QEC','QES','QER','QEH','QESUM','REFL_10CM',
               'QNICE', 'QNSNOW', 'QNGRAUPEL']   # param with 27 layers
variables2d = ['Q2', 'T2', 'TH2', 'PSFC', 'U10', 'V10', 'OLR', 'PBLH', 'W_max']  # param with 1 layer
sumVariables2d = ['RAINC','RAINNC','HAILNC','FN']  # accumulated param with 1 layer
variables3d_ave3 = ['QICE_ave3','QSNOW_ave3','QGRAUP_ave3']  # param with 9 layers

# used parameters
param_list = ['QICE_ave3','QSNOW_ave3','QGRAUP_ave3','W_max']

# to set whether NCL Radar data is used
# use_ncl = True
use_ncl = False

ncl_ave3 = True
ncl_layers = 27
ncl_disp = 'dbz'


label_type = 'ori'
m, n = 159, 159
M = 1
num_frames = 12
num_frames_truth = 3
highlevels = [i for i in range(27)]
mode_3d = 'select'
# mode_3d = 'ave'
# data_source = 'nc'
data_source = 'npy'
if mode_3d == 'ave':
    wrf_fea_dim = len(param_list)
elif mode_3d == 'select':
    wrf_fea_dim = 0
    for param in param_list:
        if param in variables3d:
            wrf_fea_dim += len(highlevels)
        elif param in variables2d or param in sumVariables2d:
            wrf_fea_dim += 1
        elif param in variables3d_ave3:
            wrf_fea_dim += 9

if use_ncl:
    if not ncl_ave3:
        fea_dim = wrf_fea_dim + 1 * ncl_layers
    elif ncl_ave3:
        fea_dim = wrf_fea_dim + 9
else:
    fea_dim = wrf_fea_dim

if label_type == 'ori':
    GuiTruthGridDir = GuiTruthOriGridDir

def getTimePeriod(dt):
    time = dt.strftime("%H:%M:%S")
    hour = int(time[0:2])
    if 0 <= hour < 6: nchour = '00'
    elif 6 <= hour < 12: nchour = '06'
    elif 12 <= hour < 18: nchour = '12'
    elif 18 <= hour <= 23: nchour = '18'
    else: print('error')
    delta_hour = hour - int(nchour)
    return nchour, delta_hour

# load WRF data from pre-stored .npy files
def getHoursGridFromNPY(npyFileDir,delta_hour, param_list,dim):
    grid = np.zeros(shape=[num_frames,m,n,dim],dtype=np.float32)
    s_idx = 0
    delta_hour -= 6   #
    for s in param_list:
        npy_grid = np.load(npyFileDir + '%s.npy' % s)
        if s in variables3d_ave3:
            temp = npy_grid[delta_hour:delta_hour+num_frames,0:9, 0:159, 0:159]
            temp = np.transpose(temp, (0, 2, 3, 1))
            if s in E_3d:
                temp = np.abs(temp)
            if s in ['QICE_ave3','QSNOW_ave3','QGRAUP_ave3']:   # no negative values
                temp[temp < 0] = 0
                # temp = temp * 100000
            if use_zscore:
                temp = apply_zscore(temp, s)
            if use_minmax:
                temp = apply_zscore(temp, s)
            grid[:, :, :, s_idx:s_idx+9] = temp
            s_idx += 9
        elif s in variables3d:
            if mode_3d == 'ave':
                temp = np.zeros((num_frames,m,n))
                for t in range(27):
                    temp += npy_grid[delta_hour:delta_hour+num_frames,t,0:159,0:159]
                if use_zscore:
                    temp = apply_zscore(temp, s)
                if use_minmax:
                    temp = apply_zscore(temp, s)
                grid[:,:,:,s_idx] = temp / 27.0
                s_idx += 1
            elif mode_3d == 'select':
                temp = npy_grid[delta_hour:delta_hour+num_frames,highlevels[0]:highlevels[-1]+1, 0:159, 0:159]
                temp = np.transpose(temp, (0, 2, 3, 1))
                if s in E_3d:
                    temp = np.abs(temp)
                if s in ['QICE','QGRAUP','QSNOW','QICE_ave3','QSNOW_ave3','QGRAUP_ave3']:   # no negative values
                    temp[temp < 0] = 0
                    # temp = temp * 100000
                if use_zscore:
                    temp = apply_zscore(temp, s)
                if use_minmax:
                    temp = apply_zscore(temp, s)
                grid[:, :, :, s_idx:s_idx + len(highlevels)] = temp
                s_idx += len(highlevels)
        elif s in variables2d or s in sumVariables2d:
            grid[:,:,:,s_idx] = npy_grid[delta_hour:delta_hour+num_frames,0:159, 0:159]
            if use_zscore:
                grid[:, :, :, s_idx] = apply_zscore(grid[:, :, :, s_idx], s)
            if use_minmax:
                grid[:, :, :, s_idx] = apply_minmax(grid[:, :, :, s_idx], s)
            s_idx += 1
    return grid

# load NCL Radar data from pre-stored .npy files
def getHoursNCLGridFromNPY(ft, nchour, delta_hour):
    if ncl_ave3 and ncl_layers == 27:
        grid = np.zeros(shape=[num_frames, m, n, 9], dtype=np.float32)
    else:
        grid = np.zeros(shape=[num_frames, m, n, ncl_layers], dtype=np.float32)
    for t in range(num_frames):
        if ncl_ave3 and ncl_layers == 27:
            nclfilepath = npyNCLFileDir + ft.strftime('%Y%m%d') + '/' + nchour + '/' + '%d_dbz_ave3.npy' % (delta_hour + t)
        else:
            nclfilepath = npyNCLFileDir + ft.strftime('%Y%m%d') + '/' + nchour + '/' + '_%d_%s.npy' % (delta_hour + t, ncl_disp)
        tmp = np.load(nclfilepath)    # (27, 159, 159) or (9, 159, 159)
        grid[t, :, :, :] = np.transpose(tmp, (1, 2, 0))
    return grid

class DataGenerator(keras.utils.Sequence):
    def __init__(self, lists, batch_size, n_labels, generator_type, shuffle=True):
        self.batch_size = batch_size
        self.lists = lists
        self.n_classes = n_labels
        self.shuffle = shuffle
        self.type = generator_type
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.lists) / self.batch_size)

    def __getitem__(self, index):
        list_batch = self.lists[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        X, y = self.__data_generation(list_batch)
        return X, y

    def on_epoch_end(self):
        if self.shuffle and self.type != 'val':
            random.shuffle(self.lists)

    def __data_generation(self, list_batch):
        batchsize = len(list_batch)
        imgs_batch = np.zeros(shape=[batchsize, num_frames, m, n, fea_dim], dtype=np.float32)
        labels_batch = np.zeros(shape=[batchsize, num_frames, m * n, 1], dtype=np.float32)
        history_batch = np.zeros(shape=[batchsize, num_frames_truth, m, n, 1], dtype=np.float32)
        for i, datetime_peroid in enumerate(list_batch):
            ddt = datetime.datetime.strptime(datetime_peroid, '%Y%m%d%H%M')
            # read WRF data
            utc = ddt + datetime.timedelta(hours=-8)
            ft = utc + datetime.timedelta(hours=(-6) * M)
            nchour, delta_hour = getTimePeriod(ft)
            delta_hour += M * 6
            if data_source == 'npy':
                date_str = ft.date().strftime("%Y%m%d")
                npyFileDir = npyWRFFileDir + '%s/' % date_str + '%s/' % nchour
                npy_grid = getHoursGridFromNPY(npyFileDir, delta_hour, param_list,wrf_fea_dim)
                imgs_batch[i,:,:,:,0:wrf_fea_dim] = npy_grid
            if use_ncl:
                nclgrid = getHoursNCLGridFromNPY(ft,nchour, delta_hour)
                if not ncl_ave3:
                    imgs_batch[i,:,:,:,-ncl_layers:] = nclgrid
                elif ncl_ave3:
                    imgs_batch[i, :, :, :, -9:] = nclgrid
            # read labels
            for hour_plus in range(num_frames):
                dt = ddt +datetime.timedelta(hours = hour_plus)
                tFilePath = GuiTruthGridDir + dt.strftime('%Y%m%d%H%M') + '_truth'
                with open(tFilePath) as tfile:
                    truth_grid = np.array(tfile.readlines(), dtype=np.float32)
                # convert to binary
                if label_type == 'ori':
                    truth_grid[truth_grid > 1] = 1
                labels_batch[i,hour_plus,:,:] = truth_grid[:,np.newaxis]
            # read history observations
            for hour_plus in range(num_frames_truth):
                dt = ddt + datetime.timedelta(hours = hour_plus - num_frames_truth)
                tFilePath = GuiTruthGridDir + dt.strftime('%Y%m%d%H%M') + '_truth'
                with open(tFilePath) as tfile:
                    truth_grid = np.array(tfile.readlines(), dtype=np.float32)
                if label_type == 'ori':
                    truth_grid[truth_grid > 1] = 1
                history_batch[i,hour_plus,:,:,:] = (truth_grid.reshape(m,n))[:,:,np.newaxis]
        truth_m1_batch = np.zeros(shape=[batchsize, 1, m, n, 1], dtype=np.float32)
        truth_m1_batch[:,0,:,:,:] = history_batch[:,-1,:,:,:]
        # full model
        return [imgs_batch, history_batch, truth_m1_batch], labels_batch
        # for WRF-Only encoder
        # return [imgs_batch, truth_m1_batch], labels_batch
        # for Observation-Only encoder
        # return [history_batch, truth_m1_batch], labels_batch
        # for full model, but -1 truth all zeros
        # return [imgs_batch, history_batch, truth_m1_batch], labels_batch
        # for conv-3d model
        # return [imgs_batch, history_batch], labels_batch

class PredictDataGenerator(keras.utils.Sequence):

    def __init__(self, lists, batch_size, shuffle=False):
        self.batch_size = batch_size
        self.lists = lists
        self.shuffle = shuffle

    def __len__(self):
        return math.ceil(len(self.lists) / self.batch_size)

    def __getitem__(self, index):
        list_batch = self.lists[index*self.batch_size:(index+1)*self.batch_size]
        X = self.__data_generation(list_batch)
        return X

    def __data_generation(self, list_batch):
        batchsize = len(list_batch)
        imgs_batch = np.zeros(shape=[batchsize, num_frames, m, n, fea_dim], dtype=np.float32)
        history_batch = np.zeros(shape=[batchsize, num_frames_truth, m, n, 1], dtype=np.float32)
        for i, datetime_peroid in enumerate(list_batch):
            ddt = datetime.datetime.strptime(datetime_peroid, '%Y%m%d%H%M')
            utc = ddt + datetime.timedelta(hours=-8)
            ft = utc + datetime.timedelta(hours=(-6) * M)
            nchour, delta_hour = getTimePeriod(ft)
            delta_hour += M * 6
            date_str = ft.date().strftime("%Y%m%d")
            npyFileDir = npyWRFFileDir + '%s/' % date_str + '%s/' % nchour
            imgs_batch[i,:,:,:,0:wrf_fea_dim] = getHoursGridFromNPY(npyFileDir, delta_hour, param_list, wrf_fea_dim)
            if use_ncl:
                nclgrid = getHoursNCLGridFromNPY(ft, nchour, delta_hour)
                if not ncl_ave3:
                    imgs_batch[i, :, :, :, -ncl_layers:] = nclgrid
                elif ncl_ave3:
                    imgs_batch[i, :, :, :, -9:] = nclgrid
            for hour_plus in range(num_frames_truth):
                dt = ddt + datetime.timedelta(hours=hour_plus - num_frames_truth)
                tFilePath = GuiTruthGridDir + dt.strftime('%Y%m%d%H%M') + '_truth'
                with open(tFilePath) as tfile:
                    truth_grid = np.array(tfile.readlines(), dtype=np.float32)
                if label_type == 'ori':
                    truth_grid[truth_grid > 1] = 1
                history_batch[i, hour_plus, :, :, :] = (truth_grid.reshape(m, n))[:, :, np.newaxis]
        truth_m1_batch = np.zeros(shape=[batchsize, 1, m, n, 1], dtype=np.float32)
        truth_m1_batch[:, 0, :, :, :] = history_batch[:, -1, :, :, :]
        return [imgs_batch, history_batch, truth_m1_batch]
        # for WRF-Only encoder
        # return [imgs_batch, truth_m1_batch]
        # for Observation-Only encoder
        # return [history_batch, truth_m1_batch]
        # for full model, but -1 truth all zeros
        # return [imgs_batch, history_batch, truth_m1_batch]
        # for conv-3d model
        # return [imgs_batch, history_batch]

