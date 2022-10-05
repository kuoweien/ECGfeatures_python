#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 14:23:50 2022

@author: weien
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import def_dataDecode as dataDecode
import def_getRpeak_main as getRpeak
import def_measureSQI as measureSQI
import datetime
from scipy.interpolate import interp1d # 導入 scipy 中的一維插值工具 interp1d
import math

# 讀Raw檔
def openRawFile(filename):
    
    with open(filename,'rb') as f:
        rawtxt=f.read()
    rawlist=dataDecode.dataDecode.rawdataDecode(rawtxt)
    
    rawdata=rawlist[0]
    ecg_rawdata=rawdata[0]#取原始資料channel1 (234是加速度)
    
    frquency=rawlist[1]
    ecg_fq=frquency[0]#頻率 狗狗的是250Hz
    
    updatetime_str=rawlist[2].split(' ')[1]#抓上傳時間
    update_datetime = datetime.datetime.strptime(updatetime_str, '%H:%M:%S')#上傳時間字串轉datetime type

    return ecg_rawdata,ecg_fq,update_datetime

#%%

filename_raw = '1-220808b.241'
# read ecg data from rawfile
ecg_raw, fs, update_datetime = openRawFile('Data/1-220808b.241')
fs = int(fs)

# Signal quality indices threshold in two seconds epoch (Patch=0.419 LTA3=2.210)
checknoiseThreshold = 20  

rri_epoch = 30 # Epoch

# Device transform to mv
patch_baseline = 0
patch_magnification = 120
lta3_baseline = 0.9
lta3_magnification = 250

# get R peak parameters
medianfilter_size = 61
gaussian_filter_sigma =  0.03*fs #20
moving_average_ms = 2.5
final_shift = 0 #Hibert轉換找到交零點後需位移回來 0.1*fs (int(0.05*fs))
detectR_maxvalue_range = (0.32*fs)*2  #草哥使用(0.3*fs)*2 #Patch=0.4*fs*2 LTA3=0.35*fs*2
rpeak_close_range = 0.15*fs #0.1*fs
lowpass_fq = 20
highpass_fq = 10

# EMG parameters
qrs_range = int(0.32*fs)    
tpeak_range = int(0.2*fs)   

# output excel url
df_output_url = 'Data/221005_TimeDomain_Features.xlsx'

df_timedomain = pd.DataFrame()

#%%讀取Data

print('ECG data record time: {} seconds'.format(len(ecg_raw)/30))

for i in range(0, int(len(ecg_raw)/rri_epoch)): 
    print('Epoch {}'.format(i))
    
    ecg = ecg_raw[i*fs*rri_epoch : (i+1)*rri_epoch*fs]
    if len(ecg) < (rri_epoch*fs):
        break
    ecg_nonoise = measureSQI.splitEpochandisCleanSignal(ecg, fs, checknoiseThreshold) #兩秒為Epoch，將雜訊的Y值改為0
    
#%%抓R peak位置

#單位換算
    ecg_nonoise_mV = (((np.array(ecg_nonoise))*1.8/65535-lta3_baseline)/lta3_magnification)*1000
    ecg_mV = (((np.array(ecg))*1.8/65535-lta3_baseline)/lta3_magnification)*1000

#抓Rpeak
    median_ecg, rpeakindex = getRpeak.getRpeak_shannon(ecg_nonoise_mV, fs, medianfilter_size, gaussian_filter_sigma, moving_average_ms, final_shift ,detectR_maxvalue_range,rpeak_close_range)
    x_lin = np.linspace(0, 30, len(median_ecg))
    
    
#%%計算RRI
    if len(rpeakindex)<=2: #若只抓到小於等於2點的Rpeak，會無法算HRV參數，因此將參數設為0
        rri_mean =0
        rri_sd = 0
        rri_rmssd = 0
        rri_nn50 = 0
        rri_pnn50 = 0
        rri_skew = 0
        rri_kurt = 0
        
    else: #若Rpeak有多於2個點，進行HRV參數計算
        rrinterval = np.diff(rpeakindex)
        rrinterval = rrinterval/(fs/1000) #RRI index點數要換算回ms (%fs，1000是因為要換算成毫秒)
    
        re_rrinterval = getRpeak.interpolate_rri(rrinterval, fs) # 補點
        
        #RRI 相關參數
        rri_mean = np.mean(re_rrinterval)
        rri_sd = np.std(re_rrinterval)
        
        outlier_upper = rri_mean+(3*rri_sd) 
        outlier_lower = rri_mean-(3*rri_sd)
        
        re_rrinterval = re_rrinterval[re_rrinterval<outlier_upper]
        re_rrinterval = re_rrinterval[re_rrinterval>outlier_lower]  #刪除outlier的rrinterval
        
        #因有刪除outlier，所以重新計算平均與SD
        rri_mean = np.mean(re_rrinterval)
        rri_sd = np.std(re_rrinterval)
        [niu, sigma, rri_skew, rri_kurt] = getRpeak.calc_stat(re_rrinterval) #峰值與偏度
        rri_rmssd = math.sqrt(np.mean((np.diff(re_rrinterval)**2))) #RMSSD
        rri_nn50 = len(np.where(np.abs(np.diff(re_rrinterval))>50)[0]) #NN50 心跳間距超過50ms的個數，藉此評估交感
        rri_pnn50 = rri_nn50/len(re_rrinterval)

#%%取EMG        
    emg_mV_linearwithzero, _ = getRpeak.deleteRTpeak(median_ecg,rpeakindex, qrs_range, tpeak_range) #刪除rtpeak並補0
    emg_mV_withoutZero = getRpeak.deleteZero(emg_mV_linearwithzero) 
    

    # EMG相關參數計算
    emg_rms = np.sqrt(np.mean(emg_mV_withoutZero**2))
    emg_var = np.var(emg_mV_withoutZero)
    emg_mav = np.sqrt(np.mean(np.abs(emg_mV_withoutZero)))
    emg_energy = np.sum((np.abs(emg_mV_withoutZero))**2)
    emg_zc = 0 #交0的次數 公式：{xi >0andxi+1 <0}or{xi <0andxi+1 >0}
    emg_mV_withoutZero = emg_mV_withoutZero.reset_index(drop=True)
    for x in range(len(emg_mV_withoutZero)-1):
        if (emg_mV_withoutZero[x]>0 and emg_mV_withoutZero[x+1]<0) or (emg_mV_withoutZero[x]<0 and emg_mV_withoutZero[x+1]>0):
            emg_zc += 1
    
    # 存入Data
    df_timedomain = df_timedomain.append({'Epoch':i+1, 'Mean':rri_mean , 'SD':rri_sd, 'RMSSD':rri_rmssd, 'NN50':rri_nn50, 'pNN50':rri_pnn50, 'Skewness':rri_skew, 'Kurtosis':rri_kurt , 'EMG_RMS':emg_rms, 'EMG_VAR':emg_var, 'EMG_MAV':emg_mav, 'EMG_ENERGY': emg_energy, 'EMG_ZC': emg_zc} ,ignore_index=True)
  
# 排序欄位位置
df_timedomain = df_timedomain[['Epoch', 'Mean','SD', 'RMSSD','NN50', 'pNN50','Skewness','Kurtosis', 'EMG_RMS', 'EMG_ENERGY', 'EMG_MAV', 'EMG_VAR', 'EMG_ZC']]

df_timedomain.to_excel(df_output_url, index=False)
        
        
        
        
        
    
