import datetime
import os
import numpy as np
import csv
# from generator import num_frames

def convert_to_binary(data, threshold):
    if data.ndim == 1:
        for i in range(len(data)):
            if data[i] >= threshold:
                data[i] = 1
            else:
                data[i] = 0
    elif data.ndim == 2:
        mm,nn = data.shape
        for i in range(mm):
            for j in range(nn):
                if data[i][j] >= threshold:
                    data[i][j] = 1
                else:
                    data[i][j] = 0
def calc_four_situation(tN, y_pred, y_test):
    y_pred = np.array(y_pred)
    y_test = np.array(y_test)
    tN[0] = np.sum((y_pred > 0) & (y_test > 0))
    tN[1] = np.sum((y_pred > 0) & (y_test < 1))
    tN[2] = np.sum((y_pred < 1) & (y_test > 0))
    tN[3] = np.sum((y_pred < 1) & (y_test < 1))
    print("N1=%d,  N2=%d,  N3=%d,  N4=%d" % (tN[0], tN[1], tN[2], tN[3]))
def calc_four_situation_neighborhood(tN, y_pred, y_test, r, side):  # side:  side-length of one grid, km
    y_pred = y_pred.reshape(159, 159)
    y_test = y_test.reshape(159, 159)
    # 0: unmarked, 1: hit, 2: miss, 3: false alarm, 4: correct negative
    y_sign = np.zeros((159, 159))
    for i in range(159):
        for j in range(159):
            if y_sign[i][j] != 0:
                continue
            elif y_test[i][j] == 0 and y_pred[i][j] == 0:
                y_sign[i][j] = 4
            elif y_test[i][j] > 0 and y_pred[i][j] > 0:
                y_sign[i][j] = 1
            elif y_test[i][j] > 0 and y_pred[i][j] == 0:
                flag = 0
                for di in range(-(r//side),(r//side)+1):
                    if flag == 1: break
                    for dj in range(-(r//side),(r//side)+1):
                        ii = i + di
                        jj = j + dj
                        if ii < 0 or ii >= 159 or jj < 0 or jj >= 159:
                            continue
                        if ((di*side)**2 + (dj*side)**2) > r**2:
                            continue
                        if y_pred[ii][jj] > 0:
                            y_sign[i][j] = 1
                            flag = 1
                            break
                if flag == 0:
                    y_sign[i][j] = 2
            elif y_pred[i][j] > 0 and y_test[i][j] == 0:
                flag = 0
                for di in range(-(r//side), (r//side)+1):
                    if flag == 1: break
                    for dj in range(-(r//side), (r//side)+1):
                        ii = i + di
                        jj = j + dj
                        if ii < 0 or ii >= 159 or jj < 0 or jj >= 159:
                            continue
                        if ((di*side)**2 + (dj*side)**2) > r**2:
                            continue
                        if y_test[ii][jj] > 0:
                            y_sign[i][j] = 1
                            flag = 1
                            break
                if flag == 0:
                    y_sign[i][j] = 3

    tN[0] = np.sum(y_sign == 1)
    tN[1] = np.sum(y_sign == 3)
    tN[2] = np.sum(y_sign == 2)
    tN[3] = np.sum(y_sign == 4)
    print("N1=%d,  N2=%d,  N3=%d,  N4=%d" % (tN[0], tN[1], tN[2], tN[3]))

def calc_TS(tN):
    if tN[0] + tN[1] + tN[2] == 0:
        return -1
    return tN[0]/(tN[0]+tN[1]+tN[2])
def calc_ETS(tN):
    if tN[0] + tN[1] + tN[2] == 0:
        return -1
    R = (tN[0]+tN[1])*(tN[0]+tN[2])/(tN[0]+tN[1]+tN[2]+tN[3])
    return (tN[0]-R)/((tN[0]+tN[1]+tN[2])-R)
def calc_POD(tN):
    if tN[0] == 0:
        return 0
    return tN[0]/(tN[0]+tN[2])
def calc_FAR(tN):
    if tN[1] == 0:
        return 0
    return tN[1]/(tN[0]+tN[1])
def calc_MAR(tN):
    if tN[2] == 0:
        return 0
    return tN[2]/(tN[0]+tN[2])
def calc_BS(tN):
    if tN[0] + tN[1] + tN[2] == 0:
        return -1
    elif tN[1] != 0 and tN[0]+tN[2] == 0:
        return 0
    return (tN[0]+tN[1])/((tN[0]+tN[2]))
def calc_AC(tN):
    return (tN[0]+tN[3])/sum(tN)
def calc_evaluation(tN, Eval):
    Eval[0] = calc_TS(tN)
    Eval[1] = calc_ETS(tN)
    Eval[2] = calc_POD(tN)
    Eval[3] = calc_FAR(tN)
    Eval[4] = calc_MAR(tN)
    Eval[5] = calc_BS(tN)
    Eval[6] = calc_AC(tN)

def cal_scores(ypred,ytest):
    tN = np.zeros(4,dtype=int)
    Eval = np.zeros(7,dtype=float)
    calc_four_situation(tN,ypred,ytest)
    calc_evaluation(tN, Eval)
    return tN,Eval
def cal_scores_neighborhood(ypred,ytest,r,side):
    tN = np.zeros(4,dtype=int)
    Eval = np.zeros(7,dtype=float)
    calc_four_situation_neighborhood(tN,ypred,ytest,r,side)
    calc_evaluation(tN, Eval)
    return tN,Eval
def cal_7_scores():
    st = datetime.datetime(2015, 1, 1, 0, 0, 0, 0)
    et = datetime.datetime(2018, 1, 1, 0, 0, 0, 0)
    tt = st
    DateTimeList = []
    while (tt < et):
        DateTimeList.append(tt)
        tt += datetime.timedelta(hours=1)
    tot1 = np.zeros(4,dtype=int)
    tot2 = np.zeros(4,dtype=int)
    sum_list_1 = []
    sum_list_2 = []
    ave_list_1 = []
    ave_list_2 = []
    for hour_plus in range(num_frames):
        ifile = open(ResultDir + '7_scores_h%d.txt' % hour_plus, 'w')
        eve_s1 = np.zeros(7, dtype=float)
        eve_s2 = np.zeros(7, dtype=float)
        n1, n2 = 0, 0
        for dt in DateTimeList:
            dt = dt + datetime.timedelta(hours=hour_plus)
            dt_str = dt.strftime('%Y%m%d%H%M')
            truthfilepath = TruthGridDir + dt_str + '_truth'
            predfilepath = ResultDir + dt_str + '_h%d' % hour_plus
            fnfilepath = WRFGridDir + dt.strftime("%Y%m%d") + '/FN/' + dt_str + '_WRF_FN.txt'
            if not os.path.exists(predfilepath)\
                    or not os.path.exists(truthfilepath)\
                    or not os.path.exists(fnfilepath):
                continue
            with open(truthfilepath) as tfile:
                truthgrid = np.array(tfile.readlines(), dtype=float)
            with open(predfilepath) as pfile:
                predgrid = np.array(pfile.readlines(), dtype=float)
            with open(fnfilepath) as wfile:
                fngrid = np.array(wfile.readlines(), dtype=float)

            print('Calculating scores for peroid %s, h%d' % (dt_str,hour_plus))
            tN_1, Eval_1 = cal_scores(predgrid,truthgrid)
            tN_2, Eval_2 = cal_scores(fngrid, truthgrid)

            ifile.write('Time Peroid %s\n' % dt_str)
            ifile.write('MODEL PRED:\n')
            ifile.write('N1:%d\tN2:%d\tN3:%d\tN4:%d\n' % (tN_1[0],tN_1[1],tN_1[2],tN_1[3]))
            ifile.write('TS:%f\tETS:%f\tPOD:%f\tFAR:%f\tMAR:%f\tBS:%f\tAC:%f\n' \
                        % (Eval_1[0], Eval_1[1], Eval_1[2], Eval_1[3], Eval_1[4], Eval_1[5], Eval_1[6] ))
            ifile.write('WRF-FN(6-12h):\n')
            ifile.write('N1:%d\tN2:%d\tN3:%d\tN4:%d\n' % (tN_2[0], tN_2[1], tN_2[2], tN_2[3]))
            ifile.write('TS:%f\tETS:%f\tPOD:%f\tFAR:%f\tMAR:%f\tBS:%f\tAC:%f\n' \
                        % (Eval_2[0], Eval_2[1], Eval_2[2], Eval_2[3], Eval_2[4], Eval_2[5], Eval_2[6]))
            ifile.write('\n')
            tot1 += tN_1
            tot2 += tN_2
            eve_1 = np.zeros(7, dtype=float)
            eve_2 = np.zeros(7, dtype=float)
            calc_evaluation(tN_1, eve_1)
            calc_evaluation(tN_2, eve_2)
            if -1 not in eve_1:      # -1 means the denominators of one or more scores are zero
                eve_s1 += eve_1
                n1 += 1
            if -1 not in eve_2:
                eve_s2 += eve_2
                n2 += 1

        eve1 = np.zeros(7,dtype=float)
        eve2 = np.zeros(7,dtype=float)
        calc_evaluation(tot1,eve1)
        calc_evaluation(tot2,eve2)

        # accumulate N1-N4 of all test samples and calculate scores one time
        ifile.write('Total(sum):\n')
        ifile.write('MODEL PRED:\n')
        ifile.write('TS:%f\tETS:%f\tPOD:%f\tFAR:%f\tMAR:%f\tBS:%f\tAC:%f\n' % tuple(eve1))
        ifile.write('WRF-FN(6-12h):\n')
        ifile.write('TS:%f\tETS:%f\tPOD:%f\tFAR:%f\tMAR:%f\tBS:%f\tAC:%f\n' % tuple(eve2))
        ifile.write('\n')

        # calculate scores for each time periods
        ifile.write('Total(average):\n')
        ifile.write('MODEL PRED:\n')
        ifile.write('TS:%f\tETS:%f\tPOD:%f\tFAR:%f\tMAR:%f\tBS:%f\tAC:%f\n' % tuple(eve_s1 / n1))
        ifile.write('WRF-FN(6-12h):\n')
        ifile.write('TS:%f\tETS:%f\tPOD:%f\tFAR:%f\tMAR:%f\tBS:%f\tAC:%f\n' % tuple(eve_s2 / n2))
        ifile.close()

        sum_list_1.append(eve1)
        sum_list_2.append(eve2)
        ave_list_1.append(eve_s1 / n1)
        ave_list_2.append(eve_s2 / n2)
    # average scores on all timesteps (no weights)
    file = open(ResultDir + '7_scores_all.txt', 'w')
    file.write('Average scores on %d timesteps:\n' % num_frames)
    file.write('sum:\n')
    file.write('MODEL PRED:\n')
    file.write('TS:%f\tETS:%f\tPOD:%f\tFAR:%f\tMAR:%f\tBS:%f\tAC:%f\n' % tuple(sum(sum_list_1) / num_frames) )
    file.write('WRF-FN(6-12h):\n')
    file.write('TS:%f\tETS:%f\tPOD:%f\tFAR:%f\tMAR:%f\tBS:%f\tAC:%f\n' % tuple(sum(sum_list_2) / num_frames))
    file.write('\n')
    file.write('ave:\n')
    file.write('MODEL PRED:\n')
    file.write('TS:%f\tETS:%f\tPOD:%f\tFAR:%f\tMAR:%f\tBS:%f\tAC:%f\n' % tuple(sum(ave_list_1) / num_frames))
    file.write('WRF-FN(6-12h):\n')
    file.write('TS:%f\tETS:%f\tPOD:%f\tFAR:%f\tMAR:%f\tBS:%f\tAC:%f\n' % tuple(sum(ave_list_2) / num_frames))
    file.close()

def cal_7_scores_0_nh(st, et, ResultDir,ScoreDir,threshold, frame_start, frame_end):
    tt = st
    DateTimeList = []
    while (tt < et):
        DateTimeList.append(tt)
        tt += datetime.timedelta(hours=1)
    tot1 = np.zeros(4,dtype=np.int64)
    tot2 = np.zeros(4,dtype=np.int64)
    ifile = open(ScoreDir + '7_scores_%d-%dh_t%.2f.txt' % (frame_start,frame_end,threshold), 'w')

    eve_s1 = np.zeros(7, dtype=float)
    eve_s2 = np.zeros(7, dtype=float)
    n1, n2 = 0, 0

    for ddt in DateTimeList:
        ddt_str = ddt.strftime('%Y%m%d%H%M')
        flag = 0
        truthgrid = np.zeros(159*159,dtype=int)
        predgrid = np.zeros(159*159,dtype=float)
        fngrid = np.zeros(159*159,dtype=int)
        for hour_plus in range(frame_start,frame_end):
            dt = ddt + datetime.timedelta(hours=hour_plus)
            dt_str = dt.strftime('%Y%m%d%H%M')
            truthfilepath = TruthGridDir + dt_str + '_truth'
            predfilepath = ResultDir + dt_str + '_h%d' % hour_plus
            fnfilepath = WRFGridDir + dt.strftime("%Y%m%d") + '/FN/' + dt_str + '_WRF_FN.txt'
            if not os.path.exists(predfilepath)\
                    or not os.path.exists(truthfilepath)\
                    or not os.path.exists(fnfilepath):
                flag = 1
                break
            with open(truthfilepath) as tfile:
                truthgrid += np.array(tfile.readlines(), dtype=int)
            with open(predfilepath) as pfile:
                tmp = np.array(pfile.readlines(), dtype=float)
                predgrid += np.round(tmp - (threshold - 0.5))
            with open(fnfilepath) as wfile:
                fngrid += np.array(wfile.readlines(), dtype=int)
        if flag == 1: continue

        print('Calculating scores for datetime peroid %s' % (ddt_str))
        tN_1, Eval_1 = cal_scores(predgrid, truthgrid)
        tN_2, Eval_2 = cal_scores(fngrid, truthgrid)

        tot1 += tN_1
        tot2 += tN_2
        eve_1 = np.zeros(7, dtype=float)
        eve_2 = np.zeros(7, dtype=float)
        calc_evaluation(tN_1, eve_1)
        calc_evaluation(tN_2, eve_2)
        if -1 not in eve_1:      # -1 means the denominators of one or more scores are zero
            eve_s1 += eve_1
            n1 += 1
        if -1 not in eve_2:
            eve_s2 += eve_2
            n2 += 1

    eve1 = np.zeros(7,dtype=float)
    eve2 = np.zeros(7,dtype=float)
    calc_evaluation(tot1,eve1)
    calc_evaluation(tot2,eve2)

    # accumulate N1-N4 of all test samples and calculate scores one time
    ifile.write('Total(sum):\n')
    ifile.write('MODEL PRED:\n')
    ifile.write('N1:%d\tN2:%d\tN3:%d\tN4:%d\n' % tuple(tot1))
    ifile.write('TS:%f\tETS:%f\tPOD:%f\tFAR:%f\tMAR:%f\tBS:%f\tAC:%f\n' % tuple(eve1))
    ifile.write('WRF-FN(6-12h):\n')
    ifile.write('N1:%d\tN2:%d\tN3:%d\tN4:%d\n' % tuple(tot2))
    ifile.write('TS:%f\tETS:%f\tPOD:%f\tFAR:%f\tMAR:%f\tBS:%f\tAC:%f\n' % tuple(eve2))
    ifile.write('\n')

    # calculate scores for each time periods
    ifile.write('Total(average):\n')
    ifile.write('MODEL PRED:\n')
    ifile.write('TS:%f\tETS:%f\tPOD:%f\tFAR:%f\tMAR:%f\tBS:%f\tAC:%f\n' % tuple(eve_s1 / n1))
    ifile.write('WRF-FN(6-12h):\n')
    ifile.write('TS:%f\tETS:%f\tPOD:%f\tFAR:%f\tMAR:%f\tBS:%f\tAC:%f\n' % tuple(eve_s2 / n2))
    ifile.close()

    return eve1

def cal_7_scores_0_nh_neighborhood(st, et, radius, side_length, ResultDir,ScoreDir , threshold, frame_start, frame_end):
    tt = st
    DateTimeList = []
    while (tt < et):
        DateTimeList.append(tt)
        tt += datetime.timedelta(hours=1)
    tot1 = np.zeros(4,dtype=np.int64)
    tot2 = np.zeros(4,dtype=np.int64)

    ifile = open(ScoreDir + '7_scores_%d-%dh_nbh_r%d_s%d_t%.2f.txt' % (frame_start,frame_end,radius,side_length,threshold), 'w')
    eve_s1 = np.zeros(7, dtype=float)
    eve_s2 = np.zeros(7, dtype=float)
    n1, n2 = 0, 0
    for ddt in DateTimeList:
        ddt_str = ddt.strftime('%Y%m%d%H%M')
        flag = 0
        truthgrid = np.zeros(159*159,dtype=int)
        predgrid = np.zeros(159*159,dtype=float)
        fngrid = np.zeros(159*159,dtype=int)
        for hour_plus in range(frame_start, frame_end):
            dt = ddt + datetime.timedelta(hours=hour_plus)
            dt_str = dt.strftime('%Y%m%d%H%M')
            truthfilepath = TruthGridDir + dt_str + '_truth'
            predfilepath = ResultDir + dt_str + '_h%d' % hour_plus
            fnfilepath = WRFGridDir + dt.strftime("%Y%m%d") + '/FN/' + dt_str + '_WRF_FN.txt'
            if not os.path.exists(predfilepath)\
                    or not os.path.exists(truthfilepath)\
                    or not os.path.exists(fnfilepath):
                flag = 1
                break
            with open(truthfilepath) as tfile:
                truthgrid += np.array(tfile.readlines(), dtype=int)
            with open(predfilepath) as pfile:
                tmp = np.array(pfile.readlines(), dtype=float)
                predgrid += np.round(tmp - (threshold - 0.5))
            with open(fnfilepath) as wfile:
                fngrid += np.array(wfile.readlines(), dtype=int)
        if flag == 1: continue
        # print(predgrid)
        print('Calculating scores for datetime peroid %s ' % (ddt_str))
        tN_1, Eval_1 = cal_scores_neighborhood(predgrid,truthgrid, radius, side_length )
        tN_2, Eval_2 = cal_scores_neighborhood(fngrid, truthgrid, radius, side_length )

        tot1 += tN_1
        tot2 += tN_2
        eve_1 = np.zeros(7, dtype=float)
        eve_2 = np.zeros(7, dtype=float)
        calc_evaluation(tN_1, eve_1)
        calc_evaluation(tN_2, eve_2)
        if -1 not in eve_1:      # -1 means the denominators of one or more scores are zero
            eve_s1 += eve_1
            n1 += 1
        if -1 not in eve_2:
            eve_s2 += eve_2
            n2 += 1

    eve1 = np.zeros(7,dtype=float)
    eve2 = np.zeros(7,dtype=float)
    calc_evaluation(tot1,eve1)
    calc_evaluation(tot2,eve2)

    # accumulate N1-N4 of all test samples and calculate scores one time
    ifile.write('Total(sum):\n')
    ifile.write('MODEL PRED:\n')
    ifile.write('N1:%d\tN2:%d\tN3:%d\tN4:%d\n' % tuple(tot1))
    ifile.write('TS:%f\tETS:%f\tPOD:%f\tFAR:%f\tMAR:%f\tBS:%f\tAC:%f\n' % tuple(eve1))
    ifile.write('WRF-FN(6-12h):\n')
    ifile.write('N1:%d\tN2:%d\tN3:%d\tN4:%d\n' % tuple(tot2))
    ifile.write('TS:%f\tETS:%f\tPOD:%f\tFAR:%f\tMAR:%f\tBS:%f\tAC:%f\n' % tuple(eve2))
    ifile.write('\n')

    # calculate scores for each time periods
    ifile.write('Total(average):\n')
    ifile.write('MODEL PRED:\n')
    ifile.write('TS:%f\tETS:%f\tPOD:%f\tFAR:%f\tMAR:%f\tBS:%f\tAC:%f\n' % tuple(eve_s1 / n1))
    ifile.write('WRF-FN(6-12h):\n')
    ifile.write('TS:%f\tETS:%f\tPOD:%f\tFAR:%f\tMAR:%f\tBS:%f\tAC:%f\n' % tuple(eve_s2 / n2))
    ifile.close()
    return eve1

def eva(resultfolderpath, threshold):

    global TruthGridDir
    global WRFGridDir
    global FigureDir
    global ResultDir
    global num_frames

    TruthGridDir = 'data/guishan_grid_4x4/'
    WRFGridDir = 'data/WRF6-24_FN/'
    ResultDir = 'results/%s/' % (resultfolderpath)

    num_frames = 12

    st = datetime.datetime(2017, 8, 1, 0, 0, 0, 0)
    et = datetime.datetime(2017, 10, 1, 0, 0, 0, 0)
    testset_disp = '2017_8_9'

    ScoreDir = 'scores/%s/%s/' % (resultfolderpath, testset_disp)

    if not os.path.isdir(ScoreDir):
        os.makedirs(ScoreDir)

    # _ = cal_7_scores_0_nh(st, et, ResultDir, ScoreDir, threshold, 0, 6)
    _ = cal_7_scores_0_nh(st, et, ResultDir, ScoreDir, threshold, 0, 12)
    # _ = cal_7_scores_0_nh(st, et, ResultDir, ScoreDir, threshold, 6, 12)
    # _ = cal_7_scores_0_nh_neighborhood(st, et, radius, 4, ResultDir, ScoreDir, threshold, 0, 6)
    # _ = cal_7_scores_0_nh_neighborhood(st, et, radius, 4, ResultDir, ScoreDir, threshold, 0, 12)
    # _ = cal_7_scores_0_nh_neighborhood(st, et, radius, 4, ResultDir, ScoreDir, threshold, 6, 12)
