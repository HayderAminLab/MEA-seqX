# -*- coding: utf-8 -*-
"""
Created on August 10 2018
@author:  Alessandro Maccione
@company: 3Brain
"""

# Data input settings
#mainPath            = '/Users/xinhu' #'C:\\_TEST\\LFPTest' #'C:\\Users\\alessandro.maccione\\Desktop' # main folder containing the experimental folders (one for each experiment)
clusters            = ['DG_Infra','DG_Supra','Hilus','CA3','CA1','EC','PC']
clusters_OB         = ['GCL','PL','GL','ONL','OCx'] #####################['GCL','PL','GL','ONL','OCx']
clusters_BS         = ['DG_Infra','DG_Supra','Hilus','CA3','CA1','EC','PC'] ########### ['DG','Hilus','CA3','CA1','EC','PC']
useUniqueClusterDef = True # if True it will use the same clusters found in the first file to all the other file in a exp folder
#filesWithCluster    = ['Exp1_CTR_4AP_Phase_00_4AP_5 Hz_10 seconds.bxr'] #['Slice_3sec.bxr'] #['Ph_03_long.bxr']#['Phase_03.bxr']#['Phase01.bxr', 'Phase01.bxr'] #['AteTest_10_10_2018.bxr'] # ['Ph_03_ALE.bxr', 'Phase01.bxr'] # if useUniqueClusterDef = True the user has to indicate which file for each folder contains the cluster definition

# global statistics settings
isNormalization     = True # if True it will normalize all files - Added 18_10_14
idFileNormalization = 0 # for each exp folder indicates which file will be used to normalize data
mainLabel           = 'CTRL'
normActiveChsLim    = (0, 6) # set the min and max value of the normalized active channels
normLFPRateLim      = (0, 6) # set the min and max value of the normalized LFP rate
normLFPAmpLim       = (0, 6) # set the min and max value of the normalized LFP amplitude
normLFPDurLim       = (0, 6) # set the min and max value of the normalized LFP duration
normLFPFirstLim     = (0, 6) # set the min and max value of the normalized LFP time to first synch LFP event
normLFPEnLim        = (0, 6) # set the min and max value of the normalized LFP energy

# thresholding settings
isWaveFormAllPositive          = True # if True it will average the LFP channel by channels by taking the abs value of the signal - Added 18_10_14
eventRateMinThr                = 1 # minimum event rate to consider active a channel [event/min]
eventRateMaxThr                = 100 # max event rate to consider active a channel [event/min]
firstEventPercSimChs           = 40 # minimum % of active channels belonging to a cluster that has to have a LFP simultaneously to calculate the first LFP event
firstEventBinSize              = 0.3 # time window of inspection in sec to identify the first simultaneous LFP occuring on a certain cluster
bufferSizeWaveForm             = 1 # buffer size in sec to average all the vaweform along a single channel (used to calculate energy and duration)
percOverlappingWaves           = 50 # percentual of overlapping waves to determine the average duration of an LFP for each cluster
isEnergyCalculatedOnTimeWindow = True # if True the energy parameter extracted from the waveform is multiplied by the event rate (for all the experiments) - Added 18_10_16

# graph settings
isHighQualitySaving      = False # if True it will store .eps high quality images - Added 18_10_16
activityMapScale         = (0, 60)  # set the min and max value of the activity map color scale [event/min]
lfpRateDistributionScale = (0, 20) # set the min and max value of the LFP rate in the boxplot distributions [event/min]
lfpAmpDistributionScale  = (0, 500) # set the min and max value of the LFP amplitudes in the boxplot distributions [uVolt]
lfpDurDistributionScale  = (0, 1.2) # set the min and max value of the LFP duration in the boxplot distributions [sec]
lfpEnDistributionScale   = (0, 50) # set the min and max value of the LFP Energy in the boxplot distributions [mVolt * msec]
averageEventRateBinSize  = 0.15 # bin size in sec to calculate the average event rate cluster by cluster
plotSingleChannels       = False # if True it stores the average of the waveform detected channel by channel
invertSignalDenoise      = True # if True it will plot the average LFP rate in the raster of the denoise in negative values in order not to overlap with the raster itself

# clean procedure settings (parameters used by the algorithm to clean false positive detection)
cleanHARDThresholding = True # it will clean at all chip level strong synchronous noise
cleanHARDBinSize      = 0.01 # (one for each expFolder) time window of inspection in sec to identify peak of activity (used to generate the bin in the histogram) and remove noise
cleanHARDStdFactor    = 5.0 # (one for each expFolder) num times the histogram value of the lfp has to overcome the std + average to be considered noise

cleanIsGlobalDenoise  = False # (one for each expFolder) if True the cleaning will be performed on the entire dataset, if False it will be performed for each single cluster

cleanBinSize          = 0.01 # (one for each expFolder) time window of inspection in sec to identify peak of activity (used to generate the bin in the histogram) and remove noise
cleanIsExtractCalib   = False # (one for each expFolder) if True it will extract the calibration signals to perform a cleaning of the false positive LFP synchronized with the calib signals
cleanIdxCalibCh       = 0 # (one for each expFolder) idx of the channel with the calibration signal (in general is 1,1-> idx=0)

cleanIsDenoise        = True # (one for each expFolder) if True it will clean from denoised LFPs
cleanStdFactor        = 1 # (one for each expFolder) num times the histogram value of the lfp has to overcome the std + average to be considered active

cleanIsRemovePeak     = True # (one for each expFolder) if True it will clean from fast co-activation
cleanMinimumNumBin    = 1 # (one for each expFolder) minimum number of bins in the histograms that has to be consecutive to consider the LFPs as activity and not a noisy oscillation