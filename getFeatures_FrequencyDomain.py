#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 14:01:57 2022

@author: weien
"""

import pandas as pd
import numpy as np
import def_getRpeak_main as getRpeak
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d # 導入 scipy 中的一維插值工具 interp1d
import def_measureSQI as measureSQI
import def_dataDecode as dataDecode
import datetime

def interpolate(raw_signal,n):   #signal為原始訊號 n為要插入產生為多少長度之訊號

    x = np.linspace(0, len(raw_signal)-1, num=len(raw_signal), endpoint=True)
    f = interp1d(x, raw_signal, kind='cubic')
    xnew = np.linspace(0, len(raw_signal)-1, num=n, endpoint=True)  
    
    return f(xnew)

def window_function(window_len,window_type='hanning'):
    if window_type=='hanning':
        return np.hanning(window_len)
    elif window_type=='hamming':
        return np.hamming(window_len)

def fft_power(input_signal,sampling_rate,window_type):
    w=window_function(len(input_signal))
    window_coherent_amplification=sum(w)/len(w)
    y_f = np.fft.fft(input_signal*w)
    y_f_Real= 2.0/len(input_signal) * np.abs(y_f[:len(input_signal)//2])/window_coherent_amplification
    x_f = np.linspace(0.0, 1.0/(2.0*1/sampling_rate), len(input_signal)//2)        
    return y_f_Real,x_f

def medfilt (x, k): #基線飄移 x是訊號 k是摺照大小
    """Apply a length-k median filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    """
    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    k2 = (k - 1) // 2
    y = np.zeros ((len (x), k), dtype=x.dtype)
    y[:,k2] = x
    for i in range (k2):
        j = k2 - i
        y[j:,i] = x[:-j]
        y[:j,i] = x[0]
        y[:-j,-(i+1)] = x[j:]
        y[-j:,-(i+1)] = x[-1]
    return np.median (y, axis=1)  #做完之後還要再用原始訊號減此值

def slidingMeanSDtofilterRRI(rri_list, epoch):
    
    clean_rri = rri_list[0:epoch]
    
    rri_standard_epoch=rri_list[0*epoch : 1*epoch]
    
    for i in range(int(len(rri_list)/epoch)-2):
        
        # rri_standard_epoch=rri_list[i*epoch : (i+1)*epoch]
        rri_input_epoch = rri_list[(i+1)*epoch : (i+2)*epoch]
        
        standdard_mean = np.mean(rri_standard_epoch)
        standdard_sd = np.std(rri_standard_epoch)
        
        input_mean = np.mean(rri_input_epoch)
        input_sd = np.std(rri_input_epoch)
        
        if standdard_sd*5 < input_sd and standdard_mean+50<input_mean and standdard_mean+50<input_mean:
            # print('Noisy')
            continue
        
        else  : #正常
            clean_rri = list(clean_rri) + list(rri_input_epoch)
            rri_standard_epoch=rri_list[(i+1)*epoch : (i+2)*epoch]
    
    return clean_rri

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


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
filename_raw = '221006a.241'
# read ecg data from rawfile
ecg_raw, fs, update_datetime = openRawFile('Data/{}'.format(filename_raw))
fs = int(fs)

# Read data parameters
lta3_baseline = 0.9
lta3_magnification = 250


# Sliding window parameters
epoch_len = 150 # seconds
rr_resample_rate = 7
slidingwidow_len = 30 #seconds
epoch = 2.5 # minutes
minute_to_second = 60

# Noise threshold
checknoiseThreshold = 20

# 抓Rpeak的參數
medianfilter_size = 61
gaussian_filter_sigma =  0.03*fs #20
moving_average_ms = 2.5
final_shift = 0 #Hibert轉換找到交零點後需位移回來 0.1*fs (int(0.05*fs))
detectR_maxvalue_range = (0.32*fs)*2  #草哥使用(0.3*fs)*2 #Patch=0.4*fs*2 LTA3=0.35*fs*2
rpeak_close_range = 0.4*fs #0.1*fs
lowpass_fq = 20
highpass_fq = 10

# EMG參數
qrs_range = int(0.32*fs)    # Human: int(0.32*fs)
tpeak_range = int(0.2*fs)   # Human: int(0.2*fs)


df_output_url = 'Data/221005_FrequencyDomain.xlsx'


df_HRV_fqdomain = pd.DataFrame()
tp_HRV = []
hf_HRV = []
lf_HRV = []
vlf_HRV = []
nLF_HRV = []
nHF_HRV = []
lfhf_ratio_hrv = []


# Rebuild protocal data
ecg_nonoise = measureSQI.splitEpochandisCleanSignal(ecg_raw, fs, checknoiseThreshold) #兩秒為Epoch，將雜訊的Y值改為0
ecg_mV = (((np.array(ecg_nonoise))*1.8/65535-lta3_baseline)/lta3_magnification)*1000


#%%計算頻域

    
for i in range(0, int(len(ecg_raw)/slidingwidow_len-1)): # one stress situation have 10 data

    print('Epoch:{}'.format(i))
    

    input_ecg = ecg_mV[(i*slidingwidow_len*fs) : int((epoch*minute_to_second*fs) + (i * slidingwidow_len*fs))] 
    
    if len(input_ecg) < (epoch*60*fs):
        break     
    # Get R peak from ECG by using shannon algorithm
    median_ecg, rpeakindex = getRpeak.getRpeak_shannon(input_ecg, fs, medianfilter_size, gaussian_filter_sigma, moving_average_ms, final_shift ,detectR_maxvalue_range,rpeak_close_range)
    
    if len(rpeakindex)<=2: #若只抓到小於等於2點的Rpeak，會無法算HRV參數，因此將參數設為0
        tp_log =0
        hf_log = 0
        vlf_log = 0
        nLF = 0
        nHF = 0
        lfhf_ratio_log = 0
        mnf = 0
        mdf = 0
       
    else: #若Rpeak有多於2個點，進行HRV參數計算
    
        # RRI計算
        rrinterval = np.diff(rpeakindex)
        rrinterval = rrinterval/(fs/1000) #RRI index點數要換算回ms (%fs，1000是因為要換算成毫秒)
        
        re_rrinterval = getRpeak.interpolate_rri(rrinterval, fs) #對因雜訊刪除的RRI進行補點
        #RRI 相關參數
        re_rri_mean = np.mean(re_rrinterval)
        re_rri_sd = np.std(re_rrinterval)
        
        outlier_upper = re_rri_mean+(3*re_rri_sd) 
        outlier_lower = re_rri_mean-(3*re_rri_sd)
        
        re_rrinterval = re_rrinterval[re_rrinterval<outlier_upper]
        re_rrinterval = re_rrinterval[re_rrinterval>outlier_lower]  #刪除outlier的rrinterval
        

        rrinterval_resample = interpolate(re_rrinterval, rr_resample_rate*epoch_len) #補點為rr_resample_rate HZ
        x_rrinterval_resample = np.linspace(0, epoch_len, len(rrinterval_resample))
        rrinterval_resample_zeromean=rrinterval_resample-np.mean(rrinterval_resample)
        
                
        # EMG計算
        emg_mV_linearwithzero, _ = getRpeak.deleteRTpeak(median_ecg,rpeakindex, qrs_range, tpeak_range) #刪除rtpeak並補0       
           
        # FFT轉頻域
        y_f_ECG, x_f_ECG = fft_power(rrinterval_resample_zeromean, rr_resample_rate, 'hanning')
        y_f_EMG, x_f_EMG = fft_power(emg_mV_linearwithzero, fs, 'hanning')
        
        
        # Calculate HRV frequency domain parameters
        tp_index = []
        hf_index = []
        lf_index = []
        vlf_index = []
        ulf_index = []
            
        
        tp_index.append(np.where( (x_f_ECG<=0.4)))  
        hf_index.append(np.where( (x_f_ECG>=0.15) & (x_f_ECG<=0.4)))  
        lf_index.append(np.where( (x_f_ECG>=0.04) & (x_f_ECG<=0.15)))  
        vlf_index.append(np.where( (x_f_ECG>=0.003) & (x_f_ECG<=0.04)))   
        ulf_index.append(np.where( (x_f_ECG<=0.003)))   
        
        
        tp_index = tp_index[0][0]
        hf_index = hf_index[0][0]
        lf_index = lf_index[0][0]
        vlf_index = vlf_index[0][0]
        ulf_index = ulf_index[0][0]
        
        
        tp = np.sum(y_f_ECG[tp_index[0]:tp_index[-1]])
        hf = np.sum(y_f_ECG[hf_index[0]:hf_index[-1]])
        lf = np.sum(y_f_ECG[lf_index[0]:lf_index[-1]])
        vlf = np.sum(y_f_ECG[vlf_index[0]:vlf_index[-1]])
        # ulf = np.log(np.sum(y_f_ECG[ulf_index[0]:ulf_index[-1]]))
        nLF = (lf/(tp-vlf))*100
        nHF = (hf/(tp-vlf))*100
        lfhf_ratio_log = np.log(lf/hf)
        
        tp_log = np.log(tp)
        hf_log = np.log(hf)
        lf_log = np.log(lf)
        vlf_log = np.log(vlf)
        
        
        ## Calculate EMG frequency domain parameters
        mnf = np.sum(y_f_EMG*x_f_EMG)/np.sum(y_f_EMG)
        

        y_f_EMG_integral = np.cumsum(y_f_EMG)
        mdf_median_index = (np.where(y_f_EMG_integral>np.max(y_f_EMG_integral)/2))[0][0] # Array is bigger than (under area)/2
        mdf = x_f_EMG[mdf_median_index]

        
        
        df_HRV_fqdomain = df_HRV_fqdomain.append({'Epoch':i+1, 
                                                  'TP':tp_log , 'HF':hf_log, 'LF':lf_log, 'VLF':vlf_log,
                                                  'nLF':nLF, 'nHF':nHF , 'LF/HF':lfhf_ratio_log, 
                                                  'MNF': mnf, 'MDF': mdf, 
                                                 } ,ignore_index=True)

        
        
df_HRV_fqdomain = df_HRV_fqdomain[['Epoch', 'TP', 'HF', 'LF', 'VLF', 'nLF', 'nHF', 'LF/HF', 'MNF', 'MDF']]
df_HRV_fqdomain.to_excel(df_output_url)


