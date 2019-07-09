import numpy as np
import pandas as pd
import math
import xlsxwriter
import time
import psutil
from random import randint

def arrayData(input_file):
    #
    # get typeFile
    typeFile = (((input_file).split("\\")[-1]).split('.'))[-1]

    # get array type numpy
    if typeFile == 'csv':
        array = pd.read_csv(input_file)
    elif typeFile == 'xlsx' or typeFile == 'xls' or typeFile == 'xlsm':
        array = pd.read_excel(input_file)
    else:
        array = ''
    return array

def EncryptionArray(array):
    d = 1
    for i in range(0, np.size(array[0], axis=0)):
        for j in np.unique(array[:, i]):
            for k in np.where(array[:, i] == j)[0]:
                array[k][i] = d
            d = d + 1
    return array


def CaculateGain(WP, WN, _WP, _WN):
    gain = _WP * (math.log(_WP / (_WP + (_WP + _WN))) - math.log(WP / (WP + (WP + WN))))
    return gain


def CaculateA(ATT, PN, WP, WN):
    A = np.array([[0, 0]])
    for i in range(0, np.size(ATT)):

        # handle caculate
        _WP = PN[i][0]
        _WN = PN[i][1]

        if _WP == 0:
            gain = 0
        else:
            gain = CaculateGain(WP, WN, _WP, _WN)

        # add result into A
        A = np.append(A, [[0, gain]], axis=0)
    A = np.delete(A, 0, axis=0)
    return A


def ReduceWeightP(P, item):
    for i in range(0, np.size(P, axis=0)):
        if np.in1d(item, P[i])[0] == True:
            # P[i][-1] = P[i][-1].astype(np.float)*(1/3)
            P[i][-1] = P[i][-1] * (1 / 3)
    return P


def ReduceWeightN(N, item):
    for i in range(0, np.size(N, axis=0)):
        if np.in1d(item, N[i])[0] == True:
            # N[i][-1] = N[i][-1].astype(np.float)*(1/3)
            N[i][-1] = N[i][-1] * (1 / 3)
    return N

def CaculateMissingValue(pathFile):
    df = arrayData(pathFile)
    countNan = df.isna().sum()
    sumCountNan = countNan.values.sum()
    return sumCountNan

def GenneratePN(ATT, P, N):
    PN = np.array([[0, 0]])
    for att in ATT:
        p_location = np.where(P[:, 0:-1] == att)
        n_location = np.where(N[:, 0:-1] == att)
        wp = 0
        wn = 0
        for i in p_location[0]:
            # wp = wp + P[:,-1][i].astype(np.float)
            wp = wp + P[:, -1][i]

        for j in n_location[0]:
            # wn = wn + N[:,-1][j].astype(np.float)
            wn = wn + N[:, -1][j]

        PN = np.append(PN, [[wp, wn]], axis=0)
    PN = np.delete(PN, (0), axis=0)
    return PN


def RemoveRecords_P(_P, i):
    p_location = np.where(_P[:, 0:-1] == i)
    _P_out = np.array([_P[0]])

    for j in p_location[0]:
        _P_out = np.append(_P_out, [_P[j]], axis=0)

    return np.delete(_P_out, 0, axis=0)


def RemoveRecords_N(_N, i):
    n_location = np.where(_N[:, 0:-1] == i)
    _N_out = np.array([_N[0]])

    for j in n_location[0]:
        _N_out = np.append(_N_out, [_N[j]], axis=0)

    return np.delete(_N_out, 0, axis=0)


def CaculateLapace(n_c, n_total, k):
    return ((n_c + 1) / (n_total + k)) * 100

def handleIgrpne(array_core):
    idx = list(np.where(array_core.isnull()))
    return array_core.drop(idx[0]).values

def handleConvert(array_core, percent):
    # present = 50
    countNan = array_core.isna().sum()
    sumCountNan = countNan.values.sum()
    row = array_core.count()[0]
    col = array_core.count(axis = 1)[0]
    pcent = int(percent)
    kq = ((row)*(col-1)*pcent/100 - sumCountNan)
    n = 0
    while n < kq:
        a = randint(0, row - 1)
        b = randint(0, col - 2)
        if array_core.isnull().ix[a,b] == False:
            n = n + 1
            array_core.ix[a, b] = np.nan
    return array_core

def handleReplaceValue0(array_core):
    columnsNamesArr = list(array_core.columns.values)
    idx = list(np.where(array_core.isnull()))
    for i in range (0, len(idx[0]) ) :
        array_core.ix[idx[0][i],idx[1][i]] = 0
        typeDf = array_core.dtypes.value_counts().idxmax()
        if(typeDf == 'object'):
        	array_core = array_core.astype('str')
    return array_core.values

def handleReplaceMuch(array_core):
    columnsNamesArr = list(array_core.columns.values)
    idx =  list(np.where(array_core.isnull()))
    for i in range (0, len(idx[0]) ) :
        countValueColumn = array_core[columnsNamesArr[idx[1][i]]].value_counts()
        array_core.ix[idx[0][i],idx[1][i]] = countValueColumn.dropna().idxmax()
    return array_core.values

def handleReplaceNear(array_core):
    idx = list(np.where(array_core.isnull()))
    for i in range(0, len(idx[0])):
        if idx[0][i] > 0:
            array_core.ix[idx[0][i], idx[1][i]] = array_core.iloc[idx[0][i]-1][idx[1][i]]
        if idx[0][i] == 0:
            array_core.ix[idx[0][i], idx[1][i]] = DeQuy(array_core, idx[0][i] + 1, idx[1][i])
    return array_core.values

def DeQuy(array_core,x,y):
    if array_core.isnull().ix[x,y] == True:
        kq = DeQuy(array_core, x + 1, y)
    else:
        return array_core.ix[x, y]
    return kq


def handleSubClass(kq, subclass):
    kq = kq.values
    arrDem = np.array([[0, 0, 0]])
    for x in subclass.items():
        for y in kq:
            if x[0] == y[0]:
                try:
                    if float(x[1]) == float(y[1]):
                        if (np.where(arrDem[:,0] == float(y[2]))[0].size) > 0:
                            for z in arrDem:
                                if float(z[0]) == float(y[2]):
                                    idx1 = np.where(arrDem[:,0] == z[0])
                                    arrDem[idx1[0][0]][1] = z[1] + float(y[3])
                                    arrDem[idx1[0][0]][2] = z[2] + 1
                            break
                        else:
                            arrDem = np.append(arrDem, [[float(y[2]),float(y[3]),1]], axis = 0)
                            break
                except:
                    if x[1] == y[1]:
                        if (np.where(arrDem[:, 0] == y[2])[0].size) > 0:
                            for z in arrDem:
                                if z[0] == y[2]:
                                    idx1 = np.where(arrDem[:, 0] == z[0])
                                    arrDem[idx1[0][0]][1] = float(z[1]) + float(y[3])
                                    arrDem[idx1[0][0]][2] = float(z[2]) + 1
                            break
                        else:
                            arrDem = np.append(arrDem, [[y[2], float(y[3]), 1]], axis=0)
                            break


    arrDem = np.delete(arrDem, (0), axis=0)
    arrResult = np.array([[float(0),float(0),float(0)]])
    for x in arrDem:
        arrResult = np.append(arrResult, [[x[0],float(x[1])/float(x[2]), round((float(x[2])/float(len(subclass)))*100, 2)]], axis = 0)
    arrResult = np.delete(arrResult, (0), axis=0)
    return arrResult

def caseForClassification(list, subClass):
    rs = []
    case = ['Xóa dòng bị thiếu dữ liệu', 'Thay thế giá trị bị thiếu bằng số 0', 'Thay thế giá trị bị thiếu giá trị xuất hiện nhiều nhất', 'Thay thế giá trị bị thiếu bằng giá trị gần nhất', 'Dữ liệu gốc']
    for i in range(4):
        ar = handleSubClass(list[i], subClass)
        ar = ar.astype(str)
        ar = np.insert(ar, 0, case[i], axis=1)
        rs.append(ar[0].tolist())
    ar = handleSubClass(list[17], subClass)
    ar = ar.astype(str)
    ar = np.insert(ar, 0, case[4], axis=1)
    rs.append(ar[0].tolist())
    return rs

def handleMissingValue(input_file, chose, percentMValue):
    df = arrayData(input_file)
    name_cols = df.columns.values

    time5 = time.time()
    df_perfect = specifyPN(df.copy().values, name_cols)
    ram5 = round(psutil.virtual_memory()[2], 2)
    cpu5 = round(psutil.cpu_percent(), 2)
    time6 = time.time()
    t_case5 = round((time6 - time5), 2)

    # handle Percent Missing Value
    countNan = df.isna().sum()
    sumCountNan = countNan.values.sum()
    if int(percentMValue) >= int(sumCountNan):
        percentMValue = int(percentMValue) - int(sumCountNan)
    if int(percentMValue) != 0:
        df = handleConvert(df, percentMValue)

    startTime = time.time()

    emptyData = np.array([['Row', 'Name', 'Value', 'Class', 'Laplace'],
                     ['', '', '', '', '']])
    emptyDF = pd.DataFrame(data=emptyData[1:,1:],
                  index=emptyData[1:,0],
                  columns=emptyData[0,1:])

    #Handle missing value
    #four case
    if 1 in chose:
        time1    = time.time()
        ar_case1 = handleIgrpne(df.copy())
        df_case1 = specifyPN(ar_case1, name_cols)
        ram1     = round(psutil.virtual_memory()[2], 2)
        cpu1     = round(psutil.cpu_percent(),2)
        time2    = time.time()
    else:
        df_case1 = emptyDF
        ram1 = ''
        cpu1 = ''

    if 2 in chose:
        time2    = time.time()
        ar_case2 = handleReplaceValue0(df.copy())
        df_case2 = specifyPN(ar_case2, name_cols)
        ram2 = round(psutil.virtual_memory()[2], 2)
        cpu2 = round(psutil.cpu_percent(), 2)
        time3 = time.time()
    else:
        df_case2 = emptyDF
        ram2 = ''
        cpu2 = ''

    if 3 in chose:
        time3    = time.time()
        ar_case3 = handleReplaceMuch(df.copy())
        df_case3 = specifyPN(ar_case3, name_cols)
        ram3 = round(psutil.virtual_memory()[2], 2)
        cpu3 = round(psutil.cpu_percent(), 2)
        time4 = time.time()
    else:
        df_case3 = emptyDF
        ram3 = ''
        cpu3 = ''

    if 4 in chose:
        time4    = time.time()
        ar_case4 = handleReplaceNear(df.copy())
        df_case4 = specifyPN(ar_case4, name_cols)
        ram4 = round(psutil.virtual_memory()[2], 2)
        cpu4 = round(psutil.cpu_percent(), 2)
        time5    = time.time()
    else:
        df_case4 = emptyDF
        ram4 = ''
        cpu4 = ''

    if 1 in chose:
        t_case1  = round((time2 - time1), 2)
    else:
        t_case1 = ' '

    if 2 in chose:
        t_case2  = round((time3 - time2), 2)
    else:
        t_case2 = ' '

    if 3 in chose:
        t_case3  = round((time4 - time3), 2)
    else:
        t_case3 = ' '

    if 4 in chose:
        t_case4  = round((time5 - time4), 2)
    else:
        t_case4 = ' '

    list = [df_case1, df_case2, df_case3, df_case4, t_case1, t_case2, t_case3, t_case4, df, ram1, ram2, ram3,
            ram4, cpu1, cpu2, cpu3, cpu4, df_perfect, t_case5, ram5, cpu5]

    return list

def specifyPN(array, name_cols):
    # init
    L = np.array([[0, 0, 0]])
    R = np.array([[0, 0, 0, 0]])
    MIN_BEST_GAIN = 0.7

    array_core = EncryptionArray(array.copy())

    # Attribute
    ATT = np.unique(np.delete(array_core, -1, axis=1))

    C = np.unique(array_core[:, -1])

    # count C
    k = np.size(C)

    # generate an empty global attributes array A
    A = np.array([[0, 0]])

    for c in C:
        # postitives
        P = array_core[np.where(array_core[:, -1] == c)]

        # negatives
        N = array_core[np.where(array_core[:, -1] != c)]

        # size
        size_ar_core = np.size(array_core, 0)  # 15
        size_P = np.size(P, 0)  # 6
        size_N = np.size(N, 0)  # 9

        # set total weight P & N
        P = np.c_[P, np.ones(size_P)]
        N = np.c_[N, np.ones(size_N)]
        # WP		     = np.sum(P[:,-1].astype(np.float))
        # WN		     = np.sum(N[:,-1].astype(np.float))
        WP = np.sum(P[:, -1])
        WN = np.sum(N[:, -1])

        # Total Weight Threshold (TWT)
        TOTAL_WEIGHT_FACTOR = 0.05
        TWT = WP * TOTAL_WEIGHT_FACTOR

        # PN array
        PN = GenneratePN(ATT, P, N)

        A = CaculateA(ATT, PN, WP, WN)

        while WP > TWT:
            # copy  P, N, A and PN: P', N', A' and PN'
            _P = P
            _N = N
            _A = A
            _PN = PN

            r = np.array([[0, 0]])

            _NSizePre = 0
            while True:

                # get location best gain
                best_gain = np.where(_A[:, -1] == np.amax(_A[:, -1]))

                # COMPARE MINGAIN
                if (_A[best_gain[0][0]][1] <= MIN_BEST_GAIN):
                    break

                r = np.append(r, [[ATT[best_gain[0][0]], c]], axis=0)

                # Adjust PN
                _P = RemoveRecords_P(_P, ATT[best_gain[0][0]])
                _N = RemoveRecords_N(_N, ATT[best_gain[0][0]])
                _PN = GenneratePN(ATT, _P, _N)

                if (_N.size == 0):
                    break

                # _WP = np.sum(_P[:,-1].astype(np.float))
                # _WN = np.sum(_N[:,-1].astype(np.float))
                _WP = np.sum(_P[:, -1])
                _WN = np.sum(_N[:, -1])

                _A = CaculateA(ATT, _PN, _WP, _WN)

            r = np.delete(r, 0, axis=0)

            if np.size(r) == 0:
                break

            # reduce weighting by decay factor and adjust PN array accordingly
            for i in r:
                P = ReduceWeightP(P, i[0])
                # N  = ReduceWeightN(N, i[0])
                n_total = np.count_nonzero(array_core == i[0])
                n_c = np.count_nonzero(P == i[0])
                lapace = CaculateLapace(n_c, n_total, k)

                # get location rule
                loc = np.where(array_core == i[0])
                row = loc[0][0]
                col = loc[1][0]
                L = np.append(L, [[row, col, lapace]], axis=0)

            PN = GenneratePN(ATT, P, N)
            A = CaculateA(ATT, PN, WP, WN)
            # WP = np.sum(P[:,-1].astype(np.float))
            # WN = np.sum(N[:,-1].astype(np.float))
            WP = np.sum(P[:, -1])
            WN = np.sum(N[:, -1])

    L = np.delete(L, 0, axis=0)
    L = np.unique(L, axis=0)

    name_cols = name_cols
    names = []
    val = []
    clas = []
    Lap = []

    for l in L:
        if l[2] > 50:
            names.append(name_cols[l[1].astype(int)])
            val.append(array[l[0].astype(int), l[1].astype(int)])
            clas.append(array[l[0].astype(int), -1])
            Lap.append(round(l[2], 2))
    # create dataframe
    d = {
        'Name': names,
        'Value': val,
        'Class': clas,
        'Laplace': Lap
    }

    # df = pd.DataFrame(d,columns=['Name','Value','Class','Lapace'])
    df = pd.DataFrame(d)
    return df


def main_cli(input_file):
    # Create a workbook
    workbook = xlsxwriter.Workbook('E:\out.xlsx')
    worksheet = workbook.add_worksheet()
    workbook.close()

    # write to new file excel above
    df = specifyPN(input_file)
    writer = pd.ExcelWriter('E:\out.xlsx')
    df.to_excel(writer, 'Sheet1', index=False)
    writer.save()
