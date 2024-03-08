# -*- coding: utf-8 -*-
import h5py
import numpy as np
import os
import help_functions.LFPAnalysis_Parameters    as LFPp
"""need change these parameters based on difference datasets"""
threshold_low = 100   #uV
threshold_high = 4000  #uV
LFPMax= 100               # (Hz) - maximum acceptable LFP Rate(LFP events/min)
LFPMin= 1                # (Hz) - minumum acceptable LFP Rate(LFP events/min)
Accept_active_channel_numbers_at_same_time = 7000
Duration_threshold = 10 #min duration thrshold(ms)
Hist_bin_low_threshold = 0.1 #Those averageEventRateHists less than Hist_bin_low_threshold of all the events count in hist are regarded as random Events

class LFPAnalysis_Function:
    def __init__(self,srcfilepath,condition_choose ='OB'):
        self.srcfilepath = srcfilepath  # main path
        self.condition = condition_choose
        if self.condition == 'OB':
            self.clusters = LFPp.clusters_OB
        else:
            self.clusters = LFPp.clusters_BS

    def get_filename(self, filepath, filetype):
        """get the reference file name and path"""
        filename = []
        for root, dirs, files in os.walk(filepath):
            for i in files:
                if filetype in i:
                    filename.append(i)
        return filename


    def denosing(self,lfpChId_raw=None, lfpTimes_raw=None, LfpForms=None,ChsGroups = None,stepVolt=0,signalInversion =0,threshold_low = 210,threshold_high = 2000,samplingRate=None):
        """1.denosing based on wavefroms amplitudes threshold
           2.denosing based on duration
        """
        numLFPs = len(lfpChId_raw)  # extract the total number of detected LFPs
        LFPLength = len(LfpForms) // numLFPs  # extract the LFP length
        tempWaveForms = LfpForms
        tempWaveForms = np.array(tempWaveForms)
        tempWaveForms = tempWaveForms.reshape(numLFPs, LFPLength)[:]
        cluster_ID = []
        for i in range(len(ChsGroups['Chs'])):
            for j in range(len(ChsGroups['Chs'][i])):
                k = (ChsGroups['Chs'][i][j][0] - 1) * 64 + (ChsGroups['Chs'][i][j][1] - 1)
                cluster_ID.append(k)
        Id = [i for i in range(len(lfpChId_raw)) if lfpChId_raw[i] in cluster_ID]
        lfpChId = [lfpChId_raw[i] for i in Id]
        lfpTimes = [lfpTimes_raw[i] for i in Id]
        tempWaveForms = [tempWaveForms[i] for i in Id]
        ##########################################Duration filter
        #all the duration
        Duration_all = [np.nonzero(el)[0][-1]*1000/samplingRate for el in tempWaveForms]
        #duration threshold(The duration for each event>= mean of all the duration)
        Normal_ID_duration = [j for j in range(len(lfpTimes)) if Duration_all[j] >= Duration_threshold]
        lfpChId_duration_filter = [lfpChId[i] for i in Normal_ID_duration]
        lfpTimes_duration_filter = [lfpTimes[i] for i in Normal_ID_duration]
        LfpForms_filter = [tempWaveForms[i] for i in Normal_ID_duration]
        ##############denoise low amplitudes channels
        Normal_ID = [j for j in range(len(lfpChId_duration_filter)) if (max(list(abs((tempWaveForms[Normal_ID_duration[j]][:np.nonzero(tempWaveForms[Normal_ID_duration[j]])[0][-1]] - (4096.0 / 2)) * stepVolt * signalInversion)))>=threshold_low and max(list(abs((tempWaveForms[Normal_ID_duration[j]][:np.nonzero(tempWaveForms[Normal_ID_duration[j]])[0][-1]] - (4096.0 / 2)) * stepVolt * signalInversion)))<=threshold_high)]
        lfpChId_new = [lfpChId_duration_filter[i] for i in Normal_ID]
        lfpTimes_new = [lfpTimes_duration_filter[i] for i in Normal_ID]
        LfpForms_new = [LfpForms_filter[i] for i in Normal_ID]
        return lfpChId_new,lfpTimes_new,LfpForms_new

    def MFR_denoise(self,lfpChId_raw=None, lfpTimes_raw=None,recordingLength = 0):
        """delete the event that actve channels appear at the same time over Accept_active_channel_numbers_at_same_time"""
        lfpChId_unique = np.unique(lfpChId_raw)
        filter_ID = [i for i in lfpChId_unique if list(lfpChId_raw).count(i)*60/recordingLength >= LFPMin and list(lfpChId_raw).count(i)*60/recordingLength <= LFPMax]
        lfpChId_filter_index = [i for i in range(len(lfpChId_raw)) if lfpChId_raw[i] in filter_ID]
        lfpChId_filter = [lfpChId_raw[i] for i in lfpChId_filter_index]
        lfpTimes_filter = [lfpTimes_raw[i] for i in lfpChId_filter_index]
        ################deleterd all channels active at the same time
        lfpTimes_filter_uni = np.unique(lfpTimes_filter)
        noise_time = [i for i in lfpTimes_filter_uni if list(lfpTimes_filter).count(i) > Accept_active_channel_numbers_at_same_time]
        lfpTimes_filter_remain_index = [i for i in range(len(lfpTimes_filter)) if lfpTimes_filter[i] not in noise_time]
        lfpChId_filter = [lfpChId_filter[i] for i in lfpTimes_filter_remain_index]
        lfpTimes_filter = [lfpTimes_filter[i] for i in lfpTimes_filter_remain_index]
        return lfpChId_filter, lfpTimes_filter

    def duplicate_channels(self,ChsGroups = None):
        """Find the wrong culster name and duplicate channels in difference clusters"""
        for clu in range(len(ChsGroups['Name'])):
            if ChsGroups['Name'][clu] in self.clusters:
                channels  = ChsGroups['Chs'][clu]
                if len(channels)>0:
                    continue
                elif len(channels)< 5:
                    print(ChsGroups['Name'][clu] + ' only have '+str(len(channels)) + ' channels!')
                else:
                    print('Error! '+ ChsGroups['Name'][clu]+' is empty!')
            else:
                print('Error! '+ ChsGroups['Name'][clu]+' not match the clusters. Please check the name of cluster.')

        for i in range(len(ChsGroups['Chs'])):
            for j in range(i+1,len(ChsGroups['Chs'])):
                com = [i for i in ChsGroups['Chs'][i] if i in ChsGroups['Chs'][j]]
                if len(com)>0:
                    for id in com:
                        print('Error! '+'Channel: '+str(id) + 'belong to cluster:' + ChsGroups['Name'][i] + ' and '+ChsGroups['Name'][j])
                else:
                    continue



    def histogram_filter(self,recordingLength=0,LFPTime=None,lfpChId =None,LfpForms=None):
        binsDistr = np.arange(0, recordingLength, LFPp.averageEventRateBinSize)  # fixed bin size
        averageEventRateHist, averageEventRateBinsEdge = np.histogram(LFPTime, bins=binsDistr,normed=False, weights=None, density=None)
        Hist_filter = [i for i in range(len(averageEventRateHist)) if averageEventRateHist[i] >= Hist_bin_low_threshold*np.max([count for count in averageEventRateHist if count < Accept_active_channel_numbers_at_same_time])]
        Hist_filter = np.unique(Hist_filter)
        Remain_Time = []
        for ID in Hist_filter:
            time_bin = [time for time in LFPTime if time>= averageEventRateBinsEdge[ID] and time <= averageEventRateBinsEdge[ID+1]]
            Remain_Time.extend(time_bin)
        Remain_ID = [i for i in range(len(LFPTime)) if LFPTime[i] in Remain_Time]
        lfpChId_remain = [lfpChId[i] for i in Remain_ID]
        LFPTime_remain = [LFPTime[i] for i in Remain_ID]
        LfpForms_remian = [LfpForms[i] for i in Remain_ID]
        return lfpChId_remain,LFPTime_remain,LfpForms_remian


    def AnalyzeExp(self,expFile = None):
        """the denosing codes main function"""
        if os.path.exists(self.srcfilepath + expFile[:-4] + '_denoised_LfpChIDs' + '.npy') and os.path.exists(
                self.srcfilepath + expFile[:-4] + '_denoised_LfpTimes' + '.npy') and os.path.exists(self.srcfilepath + expFile[:-4] + '_denoised_LfpForms' + '.npy'):
            lfpChId_last = np.load(self.srcfilepath + expFile[:-4] + '_denoised_LfpChIDs' + '.npy')
            lfpTimes_last = np.load(self.srcfilepath + expFile[:-4] + '_denoised_LfpTimes' + '.npy')
            LfpForms_last = np.load(self.srcfilepath + expFile[:-4] + '_denoised_LfpForms' + '.npy')
        else:
            filehdf5_bxr = h5py.File(self.srcfilepath + expFile, 'r')  # read LFPs bxr files
            NRecFrames = np.asarray(filehdf5_bxr["3BRecInfo"]["3BRecVars"]["NRecFrames"])[0]
            samplingRate = np.asarray(filehdf5_bxr["3BRecInfo"]["3BRecVars"]["SamplingRate"])[0]
            recordingLength = NRecFrames / samplingRate  # recording duraton in [s]
            ChsGroups = np.asarray(filehdf5_bxr["3BUserInfo"]["ChsGroups"])
            LfpForms = None
            lfpChId = np.asarray(filehdf5_bxr["3BResults"]["3BChEvents"]["LfpChIDs"])
            lfpTimes = np.asarray(filehdf5_bxr["3BResults"]["3BChEvents"]["LfpTimes"]) / samplingRate
            maxVolt = np.asarray(filehdf5_bxr["3BRecInfo"]["3BRecVars"]['MaxVolt'])[0]
            minVolt = np.asarray(filehdf5_bxr["3BRecInfo"]["3BRecVars"]['MinVolt'])[0]
            stepVolt = (maxVolt - minVolt) / 4096
            signalInversion = np.asarray(filehdf5_bxr["3BRecInfo"]["3BRecVars"]["SignalInversion"])[0]
            ##################First highlight the duplicate cahnnels,empty cluster and Wrong name of cluster
            self.duplicate_channels(ChsGroups =ChsGroups)
            ##################Second lfpChId and lfpTimes denoise
            lfpChId_denosing, lfpTimes_denosing,tempWaveForms = self.denosing(lfpChId_raw=lfpChId, lfpTimes_raw=lfpTimes,
                                                                LfpForms=LfpForms, ChsGroups=ChsGroups,
                                                                stepVolt=stepVolt, signalInversion=signalInversion,threshold_low = threshold_low,threshold_high = threshold_high,samplingRate=samplingRate)
            #threshold: for the waveforms in the range(-threshold,+threshold) detected as noise or unactive channels
            ##################Third:Mean LFP rate threshold
            lfpChId_filter, lfpTimes_filter = self.MFR_denoise(lfpChId_raw=lfpChId_denosing, lfpTimes_raw=lfpTimes_denosing,recordingLength = recordingLength)
            #######################get the filter waveforms
            ID_filter = [i for i in range(len(lfpChId_denosing)) if lfpChId_denosing[i] in lfpChId_filter]
            LfpForms_filter = np.asarray([tempWaveForms[i] for i in ID_filter])
            ######################################hist filter
            lfpChId_last, lfpTimes_last ,LfpForms_last = self.histogram_filter(recordingLength=recordingLength, LFPTime=lfpTimes_filter,lfpChId=lfpChId_filter,LfpForms=LfpForms_filter)
            LfpForms_last = np.asarray(LfpForms_last).reshape(-1)
            ##############save the new results in .npy file as dic
            # Dic = {"LfpChIDs":lfpChId_last,"LfpTimes":lfpTimes_last,"LfpForms":LfpForms_last}
            np.save(self.srcfilepath+expFile[:-4]+'_denoised_LfpChIDs',lfpChId_last)
            np.save(self.srcfilepath + expFile[:-4] + '_denoised_LfpTimes', lfpTimes_last)
            np.save(self.srcfilepath + expFile[:-4] + '_denoised_LfpForms', LfpForms_last)
        print('Denoise is Done')
        return np.asarray(lfpChId_last), np.asarray(lfpTimes_last),np.asarray(LfpForms_last)