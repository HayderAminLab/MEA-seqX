# -*- coding: utf-8 -*-
"""
Created on Dec 12 2021
@author:  BIONICS_LAB
@company: DZNE
"""
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, hilbert
from scipy import signal,stats
import networkx as nx
import h5py
import numpy as np
import pandas as pd
import threading
import json
from ast import literal_eval
import matplotlib.markers as markers
import os
import seaborn as sns
import scipy.sparse as sp_sparse
import matplotlib.image as mpimg
import help_functions.LFP_denoising as LFP_denosing
import help_functions.LFPAnalysis_Parameters as LFPp
from matplotlib.axes._axes import _log as matplotlib_axes_logger
import connectivipy
matplotlib_axes_logger.setLevel('ERROR')

"""
The following input parameters are used for calculating network topology metrics from SRT and nEphys datasets.
To compare conditions put the path for datasets in the input parameters and label condition name i.e. SD and ENR and assign desired color

"""

rows = 64
cols = 64

Start_time = 0
Stop_time = 30

column_list = ["IEGs"]  ### "Hipoo Signaling Pathway","Synaptic Vescicles_Adhesion","Receptors and channels","Synaptic plasticity","Hippoocampal Neurogenesis","IEGs"

conditions = ['SD', 'ENR']
condition1_path = r'Z:/ANALYSES/SPATIOSCALES- 10X genomics/Data/SD/'
condition2_path = r'Z:/ANALYSES/SPATIOSCALES- 10X genomics/Data/ENR/'

color = ['silver', 'dodgerblue']  # color for pooled plotting of conditions
color_choose = ['silver', 'dodgerblue', 'green', 'orange', 'purple', 'black'] # color for multiple gene plots

network_topology_metric = ['Clustering Coefficient',"Centrality","Degree",'Betweenness']

Region = ['Cortex', 'Hippo']
Cortex = ['EC', 'PC']
Hippo = ['DG','Hilus','CA3','CA1','EC','PC']

class MEASeqX_Project:

    def __init__(self, srcfilepath):
        self.srcfilepath = srcfilepath  # main path
        self.clusters = ['DG', 'Hilus', 'CA3', 'CA1', 'EC', 'PC']

    def get_filename_path(self, filepath, filetype):
        """
        Search the provided path for all files that match the filetype specified.

            Parameters
            ----------
            filepath : string
                The folder path.
            filetype: string
                The file type(e.g. .bxr, .xlsx).
            Returns
            -------
            Returns the paths for all files math the filetype.
        """
        filename = []
        Root = []
        for root, dirs, files in os.walk(filepath):
            for i in files:
                if filetype in i:
                    filename.append(i)
                    Root.append(root)
        return filename, Root

    def getRawdata(self, start, stop, expFile, channel, thr=2000):
        '''
        Get network topology metric raw data
        '''
        filehdf5 = h5py.File(expFile, 'r')
        # CONSTANT
        MaxValue = 4096.
        MaxVolt = np.asarray(filehdf5["3BRecInfo"]["3BRecVars"]["MaxVolt"])[0]
        MinVolt = np.asarray(filehdf5["3BRecInfo"]["3BRecVars"]["MinVolt"])[0]
        NRecFrames = np.asarray(filehdf5["3BRecInfo"]["3BRecVars"]["NRecFrames"])[0]
        SignInversion = np.asarray(filehdf5["3BRecInfo"]["3BRecVars"]["SignalInversion"])[0]
        stepVolt = (MaxVolt - MinVolt) / MaxValue
        version = int(filehdf5["3BData"].attrs['Version'])
        rawData = filehdf5["3BData"]["Raw"]
        if start < 0:
            start = 0
        if stop >= NRecFrames:
            stop = NRecFrames - 1
        if isinstance(channel, int) or isinstance(channel, float):
            # get one Single channel
            if version == 100:
                raw = ((rawData[int(start):int(stop), channel] - (
                        4096.0 / 2)) * stepVolt * SignInversion)
            else:
                raw = rawData[int(start) * 4096:int(stop) * 4096]
                raw = raw.reshape((len(raw) // 4096.0, 4096.0))
                raw = (raw[:, channel] - (4096.0 / 2)) * stepVolt * SignInversion
        elif isinstance(channel, str):
            # Get all channels
            if version == 100:
                raw = ((rawData[int(start):int(stop), :] - (4096.0 / 2))) * SignInversion
            else:
                raw = rawData[int(start) * 4096:int(stop) * 4096]
                raw = raw.reshape((len(raw) // 4096, 4096))
                raw = (raw - (4096.0 / 2)) * SignInversion
            #           Put to 0 saturation sample
            index = np.where(raw > thr)
            raw[index[0], index[1]] = 0
            raw = raw * float(stepVolt)
        elif isinstance(channel, np.ndarray):
            # Get an array of channels
            if version == 100:
                raw = ((rawData[int(start):int(stop), channel] - (
                        4096.0 / 2))) * stepVolt * SignInversion
            else:
                raw = rawData[int(start) * 4096:int(stop) * 4096]
                raw = raw.reshape((len(raw) // 4096, 4096))
                raw = (raw[:, channel] - (4096.0 / 2)) * stepVolt * SignInversion

        return raw

    def filter_data(self, data, SamplingRate, low, high, order=2):
        # Determine Nyquist frequency
        nyq = SamplingRate / 2
        # Set bands
        low = low / nyq
        high = high / nyq
        # Calculate coefficients
        b, a = butter(order, [low, high], btype='band')
        # Filter signal
        filtered_data = lfilter(b, a, data)
        return filtered_data

    def downsample_data(self,data, sf, target_sf):
        """Because the LFP signal does not contain high frequency components anymore the original sampling rate can be reduced.
        This will lower the size of the data and speed up calculations. Therefore we define a short down-sampling function."""
        factor = sf / target_sf
        if factor <= 10:
            data_down = signal.decimate(data, factor)
        else:
            factor = 10
            data_down = data
            while factor > 1:
                data_down = signal.decimate(data_down, factor)
                sf = sf / factor
                factor = int(min([10, sf / target_sf]))

        return data_down, sf

    def threshold_cluster(self,Data_set, threshold):
        # change all to one list
        stand_array = np.asarray(Data_set).ravel('C')
        stand_Data = pd.Series(stand_array)
        index_list, class_k = [], []
        while stand_Data.any():
            if len(stand_Data) == 1:
                index_list.append(list(stand_Data.index))
                class_k.append(list(stand_Data))
                stand_Data = stand_Data.drop(stand_Data.index)
            else:
                class_data_index = stand_Data.index[0]
                class_data = stand_Data[class_data_index]
                stand_Data = stand_Data.drop(class_data_index)
                if (abs(stand_Data - class_data) <= threshold).any():
                    args_data = stand_Data[abs(stand_Data - class_data) <= threshold]
                    stand_Data = stand_Data.drop(args_data.index)
                    index_list.append([class_data_index] + list(args_data.index))
                    class_k.append([class_data] + list(args_data))
                else:
                    index_list.append([class_data_index])
                    class_k.append([class_data])
        return index_list, class_k

    def read_related_files(self):
        """
        Read the related files.

            File input needed:
            -------
                - 'filtered_feature_bc_matrix.h5' (spaceranger_count pipeline output)
                - 'scalefactors_json.json' (spaceranger_count pipeline output)
                - 'tissue_positions_list.csv' (spaceranger_count pipeline output)
                - 'tissue_lowres_image.png' (spaceranger_count pipeline output)
                - 'Loupe Clusters.csv' (independently generated tissue structural clusters using Loupe Browser)

            Returns
            -------
            csv_file: pandas.DataFrame tissue_positions_list.xlsx
            'filtered_feature_bc_matrix.h5': parameters as followed
                -tissue_lowres_scalef.
                -features_name.
                -matr_raw
                -barcodes
            img: png 'tissue_lowres_image.png'
            csv_file_cluster:pandas.DataFrame 'Loupe Clusters.csv'

        """
        ##########################
        h5_file_name = 'filtered_feature_bc_matrix.h5'
        print(self.srcfilepath)
        h5_file, json_Root = self.get_filename_path(self.srcfilepath, h5_file_name)
        print(json_Root)
        for i in range(len(h5_file)):
            if h5_file[i][0] != '.':
                h5_root = json_Root[i] + '/' + h5_file[i]

        #############################################
        filehdf5_10x = h5py.File(h5_root, 'r')
        matrix = np.asarray(filehdf5_10x["matrix"])
        shape = np.asarray(filehdf5_10x["matrix"]['shape'])
        barcodes = np.asarray(filehdf5_10x["matrix"]["barcodes"])

        # print(len(barcodes))
        indices = np.asarray(filehdf5_10x["matrix"]["indices"])
        indptr = np.asarray(filehdf5_10x["matrix"]["indptr"])
        data = np.asarray(filehdf5_10x["matrix"]["data"])
        features_name = np.asarray(filehdf5_10x["matrix"]["features"]['name'])
        matr_raw = sp_sparse.csc_matrix((data, indices, indptr), shape=shape).toarray()
        # Read json file to get the tissue_hires_scalef values to transfor the dots in csv to images
        json_file_name = 'scalefactors_json.json'
        json_file, json_Root = self.get_filename_path(self.srcfilepath, json_file_name)
        for i in range(len(json_file)):
            if json_file[i][0] != '.':
                json_root = json_Root[i] + '/' + json_file[i]

        with open(json_root) as json_file:
            data = json.load(json_file)
        spot_diameter_fullres = data['spot_diameter_fullres']
        tissue_hires_scalef = data['tissue_hires_scalef']
        fiducial_diameter_fullres = data['fiducial_diameter_fullres']
        tissue_lowres_scalef = data['tissue_lowres_scalef']

        column_list = ["barcode", "selection", "y", "x", "pixel_y", "pixel_x"]
        ######################
        csv_file_name = 'tissue_positions_list.csv'
        csv_file, csv_Root = self.get_filename_path(self.srcfilepath, csv_file_name)
        for i in range(len(csv_file)):
            if csv_file[i][0] != '.':
                csv_root = csv_Root[i] + '/' + csv_file[i]

        csv_file = pd.read_csv(csv_root, names=column_list)
        csv_file.to_excel(self.srcfilepath + "tissue_positions_list.xlsx", index=False)
        ##################################
        img_file_name = 'tissue_lowres_image.png'
        img_file, img_Root = self.get_filename_path(self.srcfilepath, img_file_name)
        for i in range(len(img_file)):
            if img_file[i][0] != '.':
                img_root = img_Root[i] + '/' + img_file[i]
        img = mpimg.imread(img_root)

        # color_map,Cluster_list = self.get_cluster_for_SRT(self, id = ix_filter, csv_file=csv_file)
        csv_file_cluster_name = 'Loupe Clusters.csv'
        csv_file_cluster_file, csv_file_cluster_Root = self.get_filename_path(self.srcfilepath, csv_file_cluster_name)
        for i in range(len(csv_file_cluster_file)):
            if csv_file_cluster_file[i][0] != '.':
                csv_file_cluster_root = csv_file_cluster_Root[i] + '/' + csv_file_cluster_file[i]
        #############################################
        csv_file_cluster = pd.read_csv(csv_file_cluster_root)

        return csv_file,tissue_lowres_scalef,features_name,matr_raw,barcodes,img,csv_file_cluster

    def get_Groupid_SRT(self,Lfps_ID,barcode_cluster =None,nEphys_cluster =None,data = None):

        Channel_ID = [data['Channel ID'][i] for i in range(len(data))]
        Channel_Barcodes = [str(data['Barcodes_Channel_ID'][i])[2:-1] for i in range(len(data))]
        Corr_id = [data['Corr_id'][i] for i in range(len(data))]
        Corr_Barcodes = [str(data['Barcodes_Corr_id'][i])[2:-1] for i in range(len(data))]
        id_all,indices = np.unique(Channel_ID+Corr_id,return_index=True)
        Barcodes_uni = [(Channel_Barcodes+Corr_Barcodes)[i] for i in indices]
        # ########################################################
        cluster_ids = []
        clusters_ID = [] #group by clusters
        for i in range(len(self.clusters)):
            cluster_ID = []
            for k in range(len(id_all)):
                if nEphys_cluster[list(barcode_cluster).index(Barcodes_uni[k])] == self.clusters[i]:
                    cluster_ids.append(id_all[k])
                    cluster_ID.append(id_all[k])
            clusters_ID.append(cluster_ID)
        colorMap = []
        for id in Lfps_ID:
            for i in range(len(self.clusters)):
                if id in clusters_ID[i]:
                    colorMap.append(i)

        return cluster_ids,clusters_ID,colorMap

    def get_Groupid_nEphys(self,ChsGroups=None):
        # ########################################################
        cluster_ids = []
        clusters_ID = []  # group by clusters
        for i in range(len(self.clusters)):
            for j in range(len(ChsGroups['Name'])):
                if ChsGroups['Name'][j] == self.clusters[i]:
                    chsLabel = ChsGroups['Chs'][j]
                    chsIdx = [(el[0] - 1) * 64 + (el[1] - 1)  for el in chsLabel]
                    cluster_ids.extend(chsIdx)
                    clusters_ID.append(chsIdx)

        return cluster_ids,clusters_ID

    def get_Groupid_nEphys_hub(self,Lfps_ID_1,ChsGroups=None,MeaStreams =None):
        dictChsId = {}
        clustersList = ChsGroups
        for clusterToBeAnalyzed in ChsGroups['Name']:
            for idx in range(clustersList.shape[0]):
                if str(clustersList[idx][0]) == clusterToBeAnalyzed:
                    chsLabel = clustersList[idx]['Chs']
                    chsIdx = []
                    for el in chsLabel:
                        chsIdx.append((el[0] - 1)* 64 + (el[1] - 1))
                    dictChsId[clusterToBeAnalyzed] = chsIdx
        colorMap = np.array(np.zeros(len(MeaStreams[:]), dtype=int))
        for tempKey in dictChsId.keys():
            for i in range(len(ChsGroups['Name'])):
                if tempKey == ChsGroups['Name'][i]:
                    colorMap[dictChsId[tempKey]] = i+1
        colorMap_Temp = []
        Lfps_ID,colormap,Clusters,cluster_name = self.Class_ID_detect(Lfps_ID_1,ChsGroups=ChsGroups)
        for i in Lfps_ID:
            colorMap_Temp.append(colorMap[i])
        return colorMap_Temp

    def Class_ID_detect(self, Lfps_ID_1, ChsGroups=None):
        Class_LFPs_ID = Lfps_ID_1
        colormap = []
        Clusters = []
        cluster_name = []
        clusters_names = np.unique(ChsGroups['Name'])
        a = [0] * len(ChsGroups['Chs'])
        cluster_ids = []
        for i in range(len(ChsGroups['Chs'])):
            cluster_id = []
            Clusters.append(ChsGroups['Name'][i])
            ################################################################################sort the cluster_name
            for k in range(len(clusters_names)):
                if ChsGroups['Name'][i] == clusters_names[k]:
                    l = k
            ################################################################################
            for j in range(len(ChsGroups['Chs'][i])):
                cluster_id.append((ChsGroups['Chs'][i][j][0] - 1) * 64 + (ChsGroups['Chs'][i][j][1] - 1))
            cluster_ids.append(cluster_id)
        for k in Class_LFPs_ID:
            for i in range(len(cluster_ids)):
                if k in cluster_ids[i]:
                    colormap.append(i)
                    cluster_name.append(ChsGroups['Name'][i])
        return Class_LFPs_ID, colormap, Clusters, cluster_name

    def DTF(self, A, sigma=None, n_fft=None):
        """Direct Transfer Function (DTF)
            Parameters
            ----------
            A : ndarray, shape (p, N, N)
                The AR coefficients where N is the number of signals
                and p the order of the model.
            sigma : array, shape (N, )
                The noise for each time series
            n_fft : int
                The length of the FFT
            Returns
            -------
            D : ndarray, shape (n_fft, N, N)
                The estimated DTF
            """
        p, N, N = A.shape
        import math
        if n_fft is None:
            n_fft = max(int(2 ** np.math.ceil(np.log2(p))), 512)
        H, freqs = self.spectral_density(A, n_fft)
        D = np.zeros((n_fft, N, N))
        if sigma is None:
            sigma = np.ones(N)
        for i in range(n_fft):
            S = H[i]
            V = (S * sigma[None, :]).dot(S.T.conj())
            V = np.abs(np.diag(V))
            D[i] = np.abs(S * np.sqrt(sigma[None, :])) / np.sqrt(V)[:, None]
        return D, freqs

    def spectral_density(self, A, n_fft=None):
        """Estimate PSD from AR coefficients
            Parameters
            ----------
            A : ndarray, shape (p, N, N)
                The AR coefficients where N is the number of signals
                and p the order of the model.
            n_fft : int
                The length of the FFT
            Returns
            -------
            fA : ndarray, shape (n_fft, N, N)
                The estimated spectral density.
            """
        import scipy
        p, N, N = A.shape
        if n_fft is None:
            n_fft = max(int(2 ** np.math.ceil(np.log2(p))), 512)
        A2 = np.zeros((n_fft, N, N))
        A2[1:p + 1, :, :] = A  # start at 1 !
        fA = scipy.fft.fft(A2, axis=0)
        freqs = np.fft.fftfreq(n_fft)
        I = np.eye(N)
        for i in range(n_fft):
            fA[i] = scipy.linalg.inv(I - fA[i])
        return fA, freqs

    def node_strength_detect(self, G, percent=0.05):
        '''degree 20% of all the metrics
        '''
        degree_centrality = G.degree()
        #######################
        degree_sort = sorted(degree_centrality, key=lambda k: k[1], reverse=True)
        n = int(round(len(degree_sort) * percent))
        indices = []
        for i in degree_sort[:n]:
            indices.append(i[0])
        #######################
        # print('degree_centrality',indices)
        return indices

    def clustering_coefficient_detect(self, G, percent=0.05):
        clustering_coefficient = nx.clustering(G)
        degree_centrality_sort = sorted(clustering_coefficient.items(), key=lambda k: k[1], reverse=True)
        n = int(round(len(degree_centrality_sort) * percent))
        betweenness_centrality_indices = []
        for i in degree_centrality_sort[:n]:
            betweenness_centrality_indices.append(i[0])
        return betweenness_centrality_indices

    def efficiency_detect(self, G, Class_LFPs_ID, Corr_ID, percent=0.05):
        Class_LFPs_ID = list(Class_LFPs_ID)
        Corr_ID = list(Corr_ID)
        efficiency_list = [nx.algorithms.efficiency(G, Class_LFPs_ID[i], Corr_ID[i]) for i in range(len(Class_LFPs_ID))]
        conbin = []
        for i in range(len(Class_LFPs_ID)):
            conbin.append((Class_LFPs_ID[i], Corr_ID[i], efficiency_list[i]))

        def takeOne(elem):
            return elem[2]

        conbin.sort(key=takeOne)
        Class_LFPs_ID_temp = []
        Corr_ID_temp = []
        efficiency_list_temp = []
        for j in range(len(conbin)):
            Class_LFPs_ID_temp.append(conbin[j][0])
            Corr_ID_temp.append(conbin[j][1])
            efficiency_list_temp.append(conbin[j][2])
        value = int(percent * len(Class_LFPs_ID_temp))
        return Class_LFPs_ID_temp[0:value]

    def degree_detect(self, raw_id, Corr_id, Degree_raw):
        '''degree 20% of all the metrics
        '''
        degree_id = []
        corr_id = []
        degree_centrality_sort = sorted(Degree_raw, reverse=True)
        n = int(round(len(degree_centrality_sort) * 0.05))
        for i in range(len(Degree_raw)):
            if Degree_raw[i] >= degree_centrality_sort[n]:
                degree_id.append(raw_id[i])
                corr_id.append(Corr_id[i])
        return degree_id, corr_id

    def nEphys_functional_connectivity(self, low=5, high=100, denosing=False, Analysis_Item = 'LFPs',Start_Time = 0,Stop_Time = 1000):
        """
        Calculate functional connectivity from nEphys data

            File input needed:
            -------
                - '[file].bxr'
                - denoising files

            Parameters
            -------

            Returns
            -------

            File output:
            -------
                - '[file]_nEphys_functional_connectivity.xlsx'
        """
        filetype_bxr = '.bxr'
        filename_bxr, Root = self.get_filename_path(self.srcfilepath, filetype_bxr)

        def to_excel(file_list, i):
            expFile = file_list[i]
            if expFile[0] != '.':
                print(expFile)
                file_brw = self.srcfilepath + expFile[:-4] + '.brw'
                filehdf5_bxr = h5py.File(self.srcfilepath + expFile, 'r')  # read LFPs bxr files
                samplingRate = np.asarray(filehdf5_bxr["3BRecInfo"]["3BRecVars"]["SamplingRate"])[0]
                ChsGroups = np.asarray(filehdf5_bxr["3BUserInfo"]["ChsGroups"])
                MeaChs2ChIDsVector = np.asarray(filehdf5_bxr["3BResults"]["3BInfo"]["MeaChs2ChIDsVector"])

                #######################################################only get the cluster id
                if os.path.exists(self.srcfilepath + expFile[:-4] + '_denosed_LfpChIDs' + '.npy') and os.path.exists(
                        self.srcfilepath + expFile[:-4] + '_denosed_LfpTimes' + '.npy'):
                    lfpChId_raw = np.load(self.srcfilepath + expFile[:-4] + '_denosed_LfpChIDs' + '.npy')
                    lfpTimes_raw = np.load(self.srcfilepath + expFile[:-4] + '_denosed_LfpTimes' + '.npy')
                else:
                    Analysis = LFP_denosing.LFPAnalysis_Function(self.srcfilepath, condition_choose='BS')  # condition_choose ='OB' or 'BS'
                    lfpChId_raw, lfpTimes_raw, LfpForms = Analysis.AnalyzeExp(expFile=expFile)

                ########################
                start = (Start_Time / 1000)
                stop = (Stop_Time / 1000)
                start_time = min(lfpTimes_raw, key=lambda x: abs(x - start))
                start_ID = min(i for i, v in enumerate(lfpTimes_raw) if v == start_time)
                stop_time = min(lfpTimes_raw, key=lambda x: abs(x - stop))
                stop_ID = max(i for i, v in enumerate(lfpTimes_raw) if v == stop_time)
                lfpChId_raw = list(lfpChId_raw[int(start_ID):int(stop_ID)])
                lfpTimes_raw = list(lfpTimes_raw[int(start_ID):int(stop_ID)])  # change to ms

                index_list, Event_group = self.threshold_cluster(np.unique(lfpTimes_raw), 0.3) ###bin size for correlation
                lfpTimes_raw = [i * samplingRate for i in lfpTimes_raw]
                Lfps_ID_new_all = []
                Corr_id_all = []
                coor_Value_all = []
                start_time_all = []
                stop_time_all = []
                count = 0
                for event in Event_group:
                    print('Event: ' + str(count))
                    Start_time = np.min(event)
                    Stop_time = np.max(event)
                    begin = int(Start_time * samplingRate)
                    end = int(Stop_time * samplingRate)
                    dataChannels = self.getRawdata(begin, end, file_brw, 'all')
                    ################################################
                    start_time = min(lfpTimes_raw, key=lambda x: abs(x - begin))
                    start_ID = min(i for i, v in enumerate(lfpTimes_raw) if v == start_time)
                    stop_time = min(lfpTimes_raw, key=lambda x: abs(x - end))
                    stop_ID = max(i for i, v in enumerate(lfpTimes_raw) if v == stop_time)
                    LfpsChIDs = lfpChId_raw[int(start_ID):int(stop_ID)]
                    LfpsTimes = lfpTimes_raw[int(start_ID):int(stop_ID)]
                    #################################################
                    Lfps_ID, indices = np.unique(LfpsChIDs, return_index=True)
                    cluster_ids = [(ChsGroups['Chs'][i][j][0] - 1) * 64 + (ChsGroups['Chs'][i][j][1] - 1) for i in
                                   range(len(ChsGroups['Chs'])) for j in range(len(ChsGroups['Chs'][i]))]
                    Lfps_ID = [i for i in Lfps_ID if i in cluster_ids]
                    ###########################################
                    time_series_raw = [
                        np.abs(
                            hilbert(self.filter_data(dataChannels[:, i], samplingRate, low=low, high=high))) - np.mean(
                            np.abs(hilbert(self.filter_data(dataChannels[:, i], samplingRate, low=low, high=high)))) for
                        i in Lfps_ID if len(self.filter_data(dataChannels[:, i], samplingRate, low=low, high=high)) > 0]
                    Lfps_ID = [i for i in Lfps_ID if
                               len(self.filter_data(dataChannels[:, i], samplingRate, low=low, high=high)) > 0]
                    if len(Lfps_ID) > 2:
                        Corr_matrix = np.corrcoef(np.asarray(time_series_raw))
                        time_series = np.asarray(time_series_raw).T
                        ##################################
                        MVAR = connectivipy.mvarmodel.Mvar
                        result = MVAR.fit(time_series.T)[0]

                        D, freqs = self.DTF(result)
                        value = np.ndarray.mean(D[freqs > 0, :, :]) + 2 * np.std(D[freqs > 0, :, :])
                        Lfps_ID_new = [Lfps_ID[i] for i in range(len(Lfps_ID)) for j in range(i + 1, len(Lfps_ID)) if
                                       np.mean(D[freqs > 0, i, j]) >= value]
                        Corr_id = [Lfps_ID[j] for i in range(len(Lfps_ID)) for j in range(i + 1, len(Lfps_ID)) if
                                   np.mean(D[freqs > 0, i, j]) >= value]
                        coor_Value = [Corr_matrix[i, j] for i in range(len(Lfps_ID)) for j in range(i + 1, len(Lfps_ID))
                                      if np.mean(D[freqs > 0, i, j]) >= value]

                        # Lfps_ID_new, Corr_id, coor_Value = self.Spatio_temporal_filter(Lfps_ID_new, Corr_id, LfpsTimes,
                        #                                                                coor_Value, Lfps_ID, indices,
                        #                                                                samplingRate, MeaChs2ChIDsVector)
                        print('Count of Active_ID after filtering:', len(Lfps_ID_new))
                        if len(Lfps_ID_new) == len(Corr_id) and len(Lfps_ID_new) == len(coor_Value):
                            Lfps_ID_new_all.extend(Lfps_ID_new)
                            Corr_id_all.extend(Corr_id)
                            coor_Value_all.extend(coor_Value)
                            start_time_all.extend([Start_time] * len(coor_Value))
                            stop_time_all.extend([Stop_time] * len(coor_Value))
                        else:
                            Lfps_ID_new_all.extend(
                                [Lfps_ID_new[i] for i in range(min(len(Lfps_ID_new), len(Corr_id), len(coor_Value)))])
                            Corr_id_all.extend(
                                [Corr_id[i] for i in range(min(len(Lfps_ID_new), len(Corr_id), len(coor_Value)))])
                            coor_Value_all.extend(
                                [coor_Value[i] for i in range(min(len(Lfps_ID_new), len(Corr_id), len(coor_Value)))])
                            start_time_all.extend([Start_time] * len(
                                [coor_Value[i] for i in range(min(len(Lfps_ID_new), len(Corr_id), len(coor_Value)))]))
                            stop_time_all.extend([Stop_time] * len(
                                [coor_Value[i] for i in range(min(len(Lfps_ID_new), len(Corr_id), len(coor_Value)))]))
                    count += 1
                New_ID = [i for i in range(len(coor_Value_all)) if coor_Value_all[i] > 0]
                Lfps_ID_new_all = [Lfps_ID_new_all[i] for i in New_ID]
                Corr_id_all = [Corr_id_all[i] for i in New_ID]
                coor_Value_all = [coor_Value_all[i] for i in New_ID]
                start_time_all = [start_time_all[i] for i in New_ID]
                stop_time_all = [stop_time_all[i] for i in New_ID]
                max_time_all = [max(stop_time_all) for i in New_ID]
                a = {'Class_LFPs_ID_Per': Lfps_ID_new_all, 'Corr_ID_Per': Corr_id_all, 'coor_Per_data': coor_Value_all,
                     'start_time_all': start_time_all, 'stop_time_all': stop_time_all, 'max_time_all': max_time_all}
                dataframe = pd.DataFrame.from_dict(a, orient='index').T
                name = file_brw[file_brw.rfind('/') + 1:]
                desfilepath = self.srcfilepath
                if not os.path.exists(desfilepath):
                    os.mkdir(desfilepath)
                dataframe.to_excel(desfilepath + name[:-4] + "_nEphys_functional_connectivity.xlsx", index=False)

        threads = []
        x = 0
        for t in range(len(filename_bxr)):
            t = threading.Thread(target=to_excel, args=(filename_bxr, x))
            threads.append(t)
            x += 1

        for thr in threads:
            thr.start()
            thr.join()
        print("all over")

    def nEphys_functional_connectivity_excel_to_gephi(self):
        """
        Obtain .gexf file from nEphys functional connectivity

            File input needed:
            -------
                - '[file].bxr'
                - '[file]_nEphys_functional_connectivity.xlsx'

            Parameters
            -------

            Returns
            -------

            File output:
            -------
                - '[file]_nEphys_functional_connectivity.gexf'
        """

        path = self.srcfilepath[:self.srcfilepath.rfind('/')]
        desfilepath = path + '/Network_Connectivity/'
        if not os.path.exists(desfilepath):
            os.mkdir(desfilepath)
        filetype_bxr = '.bxr'
        filename_bxr, Root = self.get_filename_path(self.srcfilepath, filetype_bxr)
        for expFile in filename_bxr:
            if expFile[0] != '.':
                print(expFile)
                if expFile[0] != '.':
                    filetype_xlsx = expFile[:-4] + '_nEphys_functional_connectivity.xlsx'
                    filehdf5_bxr = h5py.File(self.srcfilepath + expFile, 'r')  # read LFPs bxr files
                    ChsGroups = np.asarray(filehdf5_bxr["3BUserInfo"]["ChsGroups"])
                    if type(ChsGroups['Name'][0]) != str:
                        ChsGroups['Name'] = [i.decode("utf-8") for i in ChsGroups['Name']]
                    MeaChs2ChIDsVector = np.asarray(filehdf5_bxr["3BResults"]["3BInfo"]["MeaChs2ChIDsVector"])
                    filename_xlsx, Root = self.get_filename_path(self.srcfilepath, filetype_xlsx)
                    if len(filename_xlsx) > 0:
                        data = pd.read_excel(Root[0] + '/' + filename_xlsx[0])
                        data_new = data.copy()
                        data_new = data_new[data_new['coor_Per_data'] > 0]
                        df = nx.from_pandas_edgelist(data_new, source='Class_LFPs_ID_Per', target='Corr_ID_Per',
                                                     edge_attr=True)
                        cluster_ids = []
                        clusters_ID = []
                        ########################################################
                        for i in range(len(self.clusters)):
                            cluster_ID = []
                            for k in range(len(ChsGroups['Name'])):
                                print(ChsGroups['Name'][k])
                                if ChsGroups['Name'][k] == self.clusters[i]:
                                    for j in range(len(ChsGroups['Chs'][k])):
                                        cluster_ids.append(
                                            (ChsGroups['Chs'][k][j][0] - 1) * 64 + (ChsGroups['Chs'][k][j][1] - 1))
                                        cluster_ID.append(
                                            (ChsGroups['Chs'][k][j][0] - 1) * 64 + (ChsGroups['Chs'][k][j][1] - 1))
                            clusters_ID.append(cluster_ID)
                        ############all links

                        hub_rich_club_nodes_all = df.nodes()
                        data_filter = [[data['Class_LFPs_ID_Per'][i], data['Corr_ID_Per'][i]] for i in range(len(data))]
                        weight_fiter = [data['coor_Per_data'][i] for i in range(len(data))]

                        start_time = [data['start_time_all'][i] for i in range(len(data_filter)) if
                                      data_filter[i][0] in cluster_ids and data_filter[i][1] in cluster_ids]
                        print("Start Time:", start_time)
                        stop_time = [data['stop_time_all'][i] for i in range(len(data_filter)) if
                                     data_filter[i][0] in cluster_ids and data_filter[i][1] in cluster_ids]
                        max_time = [data['max_time_all'][i] for i in range(len(data_filter)) if
                                    data_filter[i][0] in cluster_ids and data_filter[i][1] in cluster_ids]
                        data_all_links = [data for data in data_filter if
                                          data[0] in hub_rich_club_nodes_all and data[1] in hub_rich_club_nodes_all]
                        weight_all_links = [weight_fiter[data] for data in range(len(data_filter)) if
                                            data_filter[data][0] in hub_rich_club_nodes_all and data_filter[data][
                                                1] in hub_rich_club_nodes_all]

                        ########################
                        All_id = []
                        Class_LFPs_nodes = [int(i[0]) for i in data_all_links]
                        Corr_ID_nodes = [int(i[1]) for i in data_all_links]
                        All_id.extend(Class_LFPs_nodes)
                        All_id.extend(Corr_ID_nodes)
                        # All_id = np.unique(All_id)
                        print("All ID:", All_id)
                        ######################
                        longitude = [MeaChs2ChIDsVector["Col"][i] - 1 for i in All_id]
                        latitude = [64 - (MeaChs2ChIDsVector["Row"][i] - 1) for i in All_id]
                        start_time_all = []
                        start_time_all.extend(start_time)
                        start_time_all.extend(start_time)

                        stop_time_all = []
                        stop_time_all.extend(stop_time)
                        stop_time_all.extend(stop_time)

                        max_time_all = []
                        max_time_all.extend(max_time)
                        max_time_all.extend(max_time)

                        All_nodes_clusters = [self.clusters[j] for i in range(len(All_id)) for j in
                                              range(len(clusters_ID)) if All_id[i] in clusters_ID[j]]
                        nodes = pd.DataFrame({'id': list(All_id), 'cluster': All_nodes_clusters, 'latitude': latitude,
                                              'longitude': longitude, 'start_time': start_time_all,
                                              'stop_time': stop_time_all, 'max_time': max_time_all})
                        edges = pd.DataFrame({'source': [int(i[0]) for i in data_all_links],
                                              'target': [int(i[1]) for i in data_all_links],
                                              'weight': weight_all_links})
                        G = nx.from_pandas_edgelist(edges, 'source', 'target', edge_attr='weight')
                        nx.set_node_attributes(G, pd.Series(nodes.latitude.values, index=nodes.id).to_dict(),
                                               'latitude')
                        nx.set_node_attributes(G, pd.Series(nodes.longitude.values, index=nodes.id).to_dict(),
                                               'longitude')
                        nx.set_node_attributes(G, pd.Series(nodes.cluster.values, index=nodes.id).to_dict(), 'cluster')

                        nx.set_node_attributes(G, pd.Series(nodes.start_time.values, index=nodes.id).to_dict(),
                                               'start_time')
                        nx.set_node_attributes(G, pd.Series(nodes.stop_time.values, index=nodes.id).to_dict(),
                                               'stop_time')
                        nx.set_node_attributes(G, pd.Series(nodes.max_time.values, index=nodes.id).to_dict(),
                                               'max_time')
                        print("Writing GEXF file to:", desfilepath + filename_xlsx[0][:-5] + '.gexf')
                        nx.write_gexf(G, desfilepath + filename_xlsx[0][:-5] + '.gexf')

    def SRT_mutual_information_connectivity_excel_to_gephi(self, gene_list_name=None, Given_gene_list=False):
        """
        Obtain .gexf file from SRT mutual information.

            File input needed:
            -------
                - '[gene_list]_SRT_mutual_information_connectivity.xlsx' (obtained from SRT Gene Expression.py)
                - 'SRT_nEphys_Multiscale_Coordinates.xlsx'

            Parameters
            -------

            Returns
            -------

            File output:
            -------
                - '[gene_list]_SRT_mutual_information_connectivity.gexf'
        """

        if Given_gene_list == True:
            gene_list_name = 'Selected_genes'
        filetype_xlsx = gene_list_name + '_SRT_mutual_information_connectivity' + ".xlsx"
        path = self.srcfilepath[:self.srcfilepath.rfind('/')]
        desfilepath = path + '/Network_Connectivity/'
        if not os.path.exists(desfilepath):
            os.mkdir(desfilepath)
        data = pd.read_excel(self.srcfilepath + filetype_xlsx)
        Barcodes_Channel_ID = [str(i)[2:-1] for i in data['Barcodes_Channel_ID']]
        Barcodes_Corr_id = [str(i)[2:-1] for i in data['Barcodes_Corr_id']]

        filetype_SRT_nEphys_Coordinates = 'SRT_nEphys_Multiscale_Coordinates.xlsx'
        data_SRT_nEphys_Coordinates = pd.read_excel(self.srcfilepath + filetype_SRT_nEphys_Coordinates)
        Barcodes = [str(i) for i in data_SRT_nEphys_Coordinates['Barcodes']]
        # SRT_coordinate = [literal_eval(i) for i in data_SRT_nEphys_Coordinates['Coordinates in SRT']]
        # x_SRT = [i[0] for i in SRT_coordinate]
        # y_SRT = [i[1] for i in SRT_coordinate]
        # Cluster = [i for i in data_SRT_nEphys_Coordinates['Cluster']]

        filter_id = [i for i in range(len(Barcodes_Channel_ID)) if
                     Barcodes_Channel_ID[i] in Barcodes and Barcodes_Corr_id[i] in Barcodes]
        Barcodes_Channel_ID = [Barcodes_Channel_ID[i] for i in filter_id]
        Barcodes_Corr_id = [Barcodes_Corr_id[i] for i in filter_id]
        Channel_ID = [data['Channel ID'][i] for i in filter_id]
        Corr_id = [data['Corr_id'][i] for i in filter_id]

        csv_file, tissue_lowres_scalef, features_name, matr_raw, barcodes, img, csv_file_cluster = self.read_related_files()
        barcode_cluster = np.asarray(csv_file_cluster["Barcode"])
        scatter_x = np.asarray(csv_file["pixel_x"] * tissue_lowres_scalef)
        scatter_y = np.asarray(csv_file["pixel_y"] * tissue_lowres_scalef)
        group = np.asarray(csv_file["selection"])
        barcode_CSV = np.asarray(csv_file["barcode"])
        g = 1
        ix = np.where(group == g)
        import re
        gene_name = [str(features_name[i])[2:-1] for i in range(len(features_name))]
        filter_gene_id = [i for i in range(len(gene_name)) if
                          len(re.findall(r'^SRS', gene_name[i], flags=re.IGNORECASE)) > 0 or len(
                              re.findall(r'^Mrp', gene_name[i], flags=re.IGNORECASE)) > 0 or len(
                              re.findall(r'^Rp', gene_name[i], flags=re.IGNORECASE)) > 0 or len(
                              re.findall(r'^mt', gene_name[i], flags=re.IGNORECASE)) > 0 or len(
                              re.findall(r'^Ptbp', gene_name[i], flags=re.IGNORECASE)) > 0]
        matr = np.delete(matr_raw, filter_gene_id, axis=0)
        genes_per_cell = np.asarray((matr > 0).sum(axis=0)).squeeze()
        ###############delete the nodes with less then 1000 gene count
        deleted_notes = [i for i in range(len(genes_per_cell)) if genes_per_cell[i] <= 1000]
        ###############delete the nodes not in clusters
        deleted_notes_cluster = [i for i in range(len(genes_per_cell)) if str(barcodes[i])[2:-1] not in barcode_cluster]
        deleted_notes.extend(deleted_notes_cluster)
        deleted_notes = list(np.unique(deleted_notes))
        ##########################################################
        new_id = [j for i in barcode_CSV for j in range(len(barcodes)) if str(barcodes[j])[2:-1] == i]
        x_filter = [scatter_x[ix][i] for i in range(len(scatter_x[ix])) if new_id[i] not in deleted_notes]
        y_filter = [scatter_y[ix][i] for i in range(len(scatter_y[ix])) if new_id[i] not in deleted_notes]
        barcode_CSV_filter = [barcode_CSV[ix][i] for i in range(len(scatter_y[ix])) if new_id[i] not in deleted_notes]
        mask_id = [i for i in range(len(group)) if group[i] == 1]
        extent = [min([scatter_x[i] for i in mask_id]), max([scatter_x[i] for i in mask_id]),
                  min([scatter_y[i] for i in mask_id]), max([scatter_y[i] for i in mask_id])]
        x_coordinate, y_coordinate = x_filter - extent[0], y_filter - extent[2]
        # print(len(x_coordinate),len(barcode_CSV_filter))
        ######################
        data_all_links = [[Channel_ID[i], Corr_id[i]] for i in range(len(Channel_ID))]
        weight_all_links = [data['coor_Per_data'][i] for i in filter_id]
        All_id, indices = np.unique(Channel_ID + Corr_id, return_index=True)
        All_Barcode = Barcodes_Channel_ID + Barcodes_Corr_id
        All_Barcode = [All_Barcode[i] for i in indices]
        ######################
        longitude = []
        latitude = []
        All_nodes_clusters = []

        for id in range(len(All_id)):
            # index_in_SRT = list(Barcodes).index(All_Barcode[id])
            # latitude.append(SRT_coordinate[index_in_SRT][1]-min(y_SRT)) #y
            # longitude.append(SRT_coordinate[index_in_SRT][0]-min(x_SRT)) #x
            # All_nodes_clusters.append(Cluster[index_in_SRT])  # x
            #########################
            index_in_Raw = list(barcode_CSV_filter).index(All_Barcode[id])
            latitude.append(y_coordinate[index_in_Raw])  # y
            longitude.append(x_coordinate[index_in_Raw])  # x
            All_nodes_clusters.append(
                csv_file_cluster["Loupe Clusters"][list(barcode_cluster).index(All_Barcode[id])])  # x
        nodes = pd.DataFrame(
            {'id': list(All_id), 'cluster': All_nodes_clusters, 'latitude': latitude, 'longitude': longitude})
        edges = pd.DataFrame(
            {'source': [int(i[0]) for i in data_all_links if int(i[0]) in list(All_id)],
             'target': [int(i[1]) for i in data_all_links if int(i[1]) in list(All_id)],
             'weight': weight_all_links})
        G = nx.from_pandas_edgelist(edges, 'source', 'target', edge_attr='weight')
        nx.set_node_attributes(G, pd.Series(nodes.latitude.values, index=nodes.id).to_dict(), 'latitude')
        nx.set_node_attributes(G, pd.Series(nodes.longitude.values, index=nodes.id).to_dict(), 'longitude')
        nx.set_node_attributes(G, pd.Series(nodes.cluster.values, index=nodes.id).to_dict(), 'cluster')
        nx.write_gexf(G, desfilepath + gene_list_name + '_SRT_mutual_information_connectivity' + '.gexf')

    def network_topological_metrics(self):  # run grapth_matrix firstly
        """
       Determine specified network topological metrics from nEphys data.

            File input needed:
            -------
                - '[file].bxr'
                - '[file]_nEphys_functional_connectivity.xlsx'
                - 'SRT_nEphys_Multiscale_Coordinates.xlsx'

            Parameters
            -------

            Returns
            -------

            File output:
            -------
                - '[file]_network_topological_metrics_per_cluster.xlsx'
        """

        filetype_bxr = '.bxr'
        filename_bxr, Root = self.get_filename_path(self.srcfilepath, filetype_bxr)
        for i in range(len(filename_bxr)):
            expFile = filename_bxr[i]
            if expFile[0]!='.':
                filehdf5_bxr = h5py.File(Root[i] + expFile, 'r')  # read LFPs bxr files
                ChsGroups = np.asarray(filehdf5_bxr["3BUserInfo"]["ChsGroups"])
                if type(ChsGroups['Name'][0]) != str:
                    ChsGroups['Name'] = [i.decode("utf-8") for i in ChsGroups['Name']]
                MeaChs2ChIDsVector = np.asarray(filehdf5_bxr["3BResults"]["3BInfo"]["MeaChs2ChIDsVector"])
                cluster_ids,clusters_ID = self.get_Groupid_nEphys(ChsGroups=ChsGroups)
                print (filename_bxr)
                filetype_xlsx = expFile[:-4] + '_nEphys_functional_connectivity.xlsx'
                filename_xlsx, Root = self.get_filename_path(self.srcfilepath, filetype_xlsx)
                print(filetype_xlsx)
                data = pd.read_excel(Root[0] + '/' + filename_xlsx[0])

                data_new = data.copy()
                Channel_ID_cluster = []
                for id in data_new['Class_LFPs_ID_Per']:
                    print (data_new)
                    for i in range(len(self.clusters)):
                        if i < len(clusters_ID) and id in clusters_ID[i]:
                            if i < len(self.clusters):
                                Channel_ID_cluster.append(self.clusters[i])
                # Channel_ID_cluster = [self.clusters[i] for id in data_new['Class_LFPs_ID_Per'] for i in range(len(self.clusters)) if id in clusters_ID[i]]
                Corr_id_cluster = []
                for id in data_new['Corr_ID_Per']:
                    for i in range(len(self.clusters)):
                        if i < len(clusters_ID) and i < len(self.clusters) and id in clusters_ID[i]:
                            Corr_id_cluster.append(self.clusters[i])
               #  Corr_id_cluster = [self.clusters[i] for id in data_new['Corr_ID_Per'] for i in range(len(self.clusters)) if id in clusters_ID[i]]
                df_add = pd.DataFrame({'Channel_ID_cluster': Channel_ID_cluster, 'Corr_id_cluster': Corr_id_cluster})
                new_df = pd.concat([data_new, df_add], axis=1)

                cluster = []
                channel_ID = []
                coordinates = []
                result_clustering_coefficient = []  # clustering coefficient
                result_average_node_connectivity = []

                result_degree = []
                result_betweenness = []
                df_new = nx.from_pandas_edgelist(new_df, source='Class_LFPs_ID_Per', target='Corr_ID_Per', edge_attr=True)
                clustering_coefficient = nx.clustering(df_new)
                degree_centrality = nx.algorithms.centrality.degree_centrality(df_new)
                betweenness_centrality = nx.algorithms.centrality.betweenness_centrality(df_new)
                for i in range(len(self.clusters)):
                    if i < len(clusters_ID):
                        for id in clusters_ID[i]:
                            if id in list(df_new.nodes):
                                channel_ID.append(id)
                                coordinates.append([MeaChs2ChIDsVector["Col"][id]-1,MeaChs2ChIDsVector["Row"][id]-1])
                                if id in clustering_coefficient.keys():
                                    result_clustering_coefficient.append(clustering_coefficient[id])
                                    result_degree.append(df_new.degree(id))
                                    result_average_node_connectivity.append(degree_centrality[id])
                                    result_betweenness.append(betweenness_centrality[id])
                                    cluster.append(self.clusters[i])
                                else:
                                    result_clustering_coefficient.append(0)
                                    result_degree.append(0)
                                    result_average_node_connectivity.append(0)
                                    result_betweenness.append(0)

                                    cluster.append(self.clusters[i])

                a = {'Channel ID': channel_ID, 'Clustering Coefficient':result_clustering_coefficient, "Centrality": result_average_node_connectivity,
                     "Degree": result_degree, 'Betweenness': result_betweenness,'Cluster':cluster, 'Coordinates': coordinates}
                df = pd.DataFrame.from_dict(a, orient='index').T
                df.to_excel(self.srcfilepath + expFile[:-4] +'_network_topological_metrics_per_cluster' + ".xlsx",index=False)

    def coordinates_for_network_topological_metrics(self):  # run grapth_matrix firstly
        """
        Provide the transcriptomic and electrophysiologic overlay coordinates for network topological metrics.

            File input needed:
            -------
                - related files
                - '[file].bxr'
                - 'SRT_nEphys_Multiscale_Coordinates.xlsx'
                - '[file]_network_topological_metrics_per_cluster.xlsx'

            Parameters
            -------

            Returns
            -------

            File output:
            -------
                - 'SRT_nEphys_Multiscale_Coordinates_for_network_topological_metrics.xlsx'
        """

        filetype_SRT_nEphys_Coordinates = 'SRT_nEphys_Multiscale_Coordinates.xlsx'
        filename_SRT_nEphys_Coordinates, Root = self.get_filename_path(self.srcfilepath, filetype_SRT_nEphys_Coordinates)
        for i in range(len(filename_SRT_nEphys_Coordinates)):
            if filename_SRT_nEphys_Coordinates[i][0] != '.':
                SRT_nEphys_Coordinates_root = Root[i] + '/' + filename_SRT_nEphys_Coordinates[i]

        # column_list = ["Synaptic plasticity", "Hippo Signaling Pathway", "Hippocampal Neurogenesis", "Synaptic Vescicles/Adhesion", "Receptors and channels", "IEGs"]
        data_SRT_nEphys_Coordinates = pd.read_excel(SRT_nEphys_Coordinates_root)

        filehdf5_bxr_name = '.bxr'
        filehdf5_bxr_file, filehdf5_bxr_Root = self.get_filename_path(self.srcfilepath, filehdf5_bxr_name)

        for i in range(len(filehdf5_bxr_file)):
            if filehdf5_bxr_file[i][0] != '.':
                filehdf5_bxr_root = filehdf5_bxr_Root[i] + '/' + filehdf5_bxr_file[i]
                expFile = filehdf5_bxr_file[i]
        data_nEphys = pd.read_excel(self.srcfilepath + expFile[:-4] + '_network_topological_metrics_per_cluster' + ".xlsx")
        filehdf5_bxr = h5py.File(filehdf5_bxr_root, 'r')
        MeaChs2ChIDsVector = np.asarray(filehdf5_bxr["3BResults"]["3BInfo"]["MeaChs2ChIDsVector"])
        coordinate_nEphys = [[MeaChs2ChIDsVector["Col"][id] - 1, MeaChs2ChIDsVector["Row"][id] - 1] for id in
                           data_nEphys['Channel ID']]

        cluster = []
        barcodes = []
        nEphys_coordinate = []

        result_clustering_coefficient_nEphys = []
        result_centrality_nEphys = []
        result_degree_nEphys = []
        result_betweenness_nEphys = []

        for cor in range(len(data_SRT_nEphys_Coordinates['Barcodes'])):
            related_LFP_cor = literal_eval(data_SRT_nEphys_Coordinates['Coordinates in nEphys'][cor])
            if len(related_LFP_cor) > 0:
                result_clustering_coefficient_nEphys_mean = []  # clustering coefficient
                result_centrality_nEphys_mean = []
                result_degree_nEphys_mean = []
                result_betweenness_nEphys_mean = []
                for cor_in_nEphys in related_LFP_cor:
                    if cor_in_nEphys in list(coordinate_nEphys):
                        index_in_nEphys = list(coordinate_nEphys).index(cor_in_nEphys)
                        result_clustering_coefficient_nEphys_mean.append(
                            float(data_nEphys['Clustering Coefficient'][index_in_nEphys]))
                        result_centrality_nEphys_mean.append(float(data_nEphys['Centrality'][index_in_nEphys]))
                        result_degree_nEphys_mean.append(float(data_nEphys['Degree'][index_in_nEphys]))
                        result_betweenness_nEphys_mean.append(float(data_nEphys['Betweenness'][index_in_nEphys]))
                    else:
                        result_clustering_coefficient_nEphys_mean.append(0)
                        result_centrality_nEphys_mean.append(0)
                        result_degree_nEphys_mean.append(0)
                        result_betweenness_nEphys_mean.append(0)

                cluster.append(data_SRT_nEphys_Coordinates['Cluster'][cor])
                barcodes.append(data_SRT_nEphys_Coordinates['Barcodes'][cor])
                nEphys_coordinate.append(related_LFP_cor)

                result_clustering_coefficient_nEphys.append(
                    np.mean([i for i in result_clustering_coefficient_nEphys_mean if i == i]))
                result_centrality_nEphys.append(np.mean([i for i in result_clustering_coefficient_nEphys_mean if i == i]))

                result_degree_nEphys.append(np.mean([i for i in result_degree_nEphys_mean if i == i]))
                result_betweenness_nEphys.append(np.mean([i for i in result_betweenness_nEphys_mean if i == i]))

        a = {'Barcodes': barcodes, 'Coordinates in nEphys': nEphys_coordinate,
             'Clustering Coefficient nEphys': result_clustering_coefficient_nEphys,
             "Centrality nEphys": result_centrality_nEphys, "Degree nEphys": result_degree_nEphys,
             'Betweenness nEphys': result_betweenness_nEphys, 'Cluster': cluster}
        df = pd.DataFrame.from_dict(a, orient='index').T
        df.to_excel(self.srcfilepath + 'SRT_nEphys_Multiscale_Coordinates_for_network_topological_metrics' + ".xlsx", index=False)

    def network_topological_feature_statistics_per_node(self, gene_list_name=None):  # run grapth_matrix firstly
        """
        Determine specified network topological metrics from nEphys data and the SRT data.

            File input needed:
            -------
                - related files
                - '[gene_list]_SRT_mutual_information_connectivity.xlsx'
                - 'SRT_nEphys_Multiscale_Coordinates_for_network_topological_metrics.xlsx'
                - 'SRT_nEphys_Multiscale_Coordinates.xlsx'
                - '[gene_list]_gene_expression_per_cluster.xlsx' (obtained from SRT Gene Expression.py)
                - '[file]_network_topological_metrics_per_cluster.xlsx'

            Parameters
            -------

            Returns
            -------

            File output:
            -------
                - '[gene_list]_SRT_nEphys_Multiscale_Coordinates_for_network_topological_metrics_per_cluster.xlsx'
        """

        csv_file, tissue_lowres_scalef, features_name, matr_raw, barcodes, img, csv_file_cluster = self.read_related_files()
        barcode_cluster = np.asarray(csv_file_cluster["Barcode"])
        nEphys_cluster = np.asarray(csv_file_cluster["Loupe Clusters"])

        data = pd.read_excel(self.srcfilepath + gene_list_name + '_SRT_mutual_information_connectivity' + ".xlsx")

        filename_network_topology_metric_statistics_from_nEphys = pd.read_excel(
            self.srcfilepath + 'SRT_nEphys_Multiscale_Coordinates_for_network_topological_metrics.xlsx')

        data_SRT_nEphys_Coordinates = pd.read_excel(self.srcfilepath + 'SRT_nEphys_Multiscale_Coordinates.xlsx')

        filetype_gene_expression_per_cluster = gene_list_name + '_gene_expression_per_cluster.xlsx'
        filename_gene_expression_per_cluster, Root = self.get_filename_path(self.srcfilepath,
                                                                         filetype_gene_expression_per_cluster)
        for i in range(len(filename_gene_expression_per_cluster)):
            if filename_gene_expression_per_cluster[i][0] != '.':
                gene_expression_per_cluster_root = Root[i] + '/' + filename_gene_expression_per_cluster[i]

        filehdf5_bxr_name = '.bxr'
        filehdf5_bxr_file, filehdf5_bxr_Root = self.get_filename_path(self.srcfilepath, filehdf5_bxr_name)

        for i in range(len(filehdf5_bxr_file)):
            if filehdf5_bxr_file[i][0] != '.':
                filehdf5_bxr_root = filehdf5_bxr_Root[i] + '/' + filehdf5_bxr_file[i]
                expFile = filehdf5_bxr_file[i]
        data_nEphys = pd.read_excel(self.srcfilepath + expFile[:-4] + '_network_topological_metrics_per_cluster' + ".xlsx")
        filehdf5_bxr = h5py.File(filehdf5_bxr_root, 'r')
        MeaChs2ChIDsVector = np.asarray(filehdf5_bxr["3BResults"]["3BInfo"]["MeaChs2ChIDsVector"])
        coordinate_nEphys = [[MeaChs2ChIDsVector["Col"][id] - 1, MeaChs2ChIDsVector["Row"][id] - 1] for id in
                           data_nEphys['Channel ID']]
        ##########################
        cluster_ids, clusters_ID, colorMap_hub_rich_club_nodes_all = self.get_Groupid_SRT([],
                                                                                          barcode_cluster=barcode_cluster,
                                                                                          nEphys_cluster=nEphys_cluster,
                                                                                          data=data)  # cluster_ID <-> self.clusters
        data_new = data.copy()
        Channel_ID_cluster = [self.clusters[i] for id in data_new['Channel ID'] for i in range(len(self.clusters)) if
                              id in clusters_ID[i]]
        Corr_id_cluster = [self.clusters[i] for id in data_new['Corr_id'] for i in range(len(self.clusters)) if
                           id in clusters_ID[i]]
        df_add = pd.DataFrame({'Channel_ID_cluster': Channel_ID_cluster, 'Corr_id_cluster': Corr_id_cluster})
        new_df = pd.concat([data_new, df_add], axis=1)

        cluster = []
        channel_ID = []
        barcodes = []
        nEphys_coordinate = []

        result_clustering_coefficient = []  # clustering coefficient
        result_centrality = []

        result_degree = []
        result_betweenness = []

        result_clustering_coefficient_nEphys = []  # clustering coefficient
        result_centrality_nEphys = []

        result_degree_nEphys = []
        result_betweenness_nEphys = []

        df_new = nx.from_pandas_edgelist(new_df, source='Channel ID', target='Corr_id', edge_attr=True)
        clustering_coefficient = nx.clustering(df_new)
        degree_centrality = nx.algorithms.centrality.degree_centrality(df_new)
        betweenness_centrality = nx.algorithms.centrality.betweenness_centrality(df_new)
        for i in range(len(self.clusters)):
            for id in clusters_ID[i]:
                if id in list(df_new.nodes):
                    channel_ID.append(id)
                    barcodes.append(str(data['Barcodes_Corr_id'][list(data['Corr_id']).index(id)])[2:-1])
                    # print(str(data['Barcodes_Corr_id'][list(data['Corr_id']).index(id)])[2:-1])
                    # # print([str(bar) for bar in filename_network_topology_metric_statistics_from_nEphys['Barcodes']])
                    # print(str(data['Barcodes_Corr_id'][list(data['Corr_id']).index(id)])[2:-1] in [str(bar) for bar in filename_network_topology_metric_statistics_from_nEphys['Barcodes']])
                    if str(data['Barcodes_Corr_id'][list(data['Corr_id']).index(id)])[2:-1] in [str(bar) for bar in
                                                                                                filename_network_topology_metric_statistics_from_nEphys[
                                                                                                    'Barcodes']]:
                        id_nEphys = list(
                            [str(bar) for bar in filename_network_topology_metric_statistics_from_nEphys['Barcodes']]).index(
                            str(data['Barcodes_Corr_id'][list(data['Corr_id']).index(id)])[2:-1])
                        result_clustering_coefficient_nEphys.append(
                            list(filename_network_topology_metric_statistics_from_nEphys['Clustering Coefficient nEphys'])[
                                id_nEphys])
                        result_centrality_nEphys.append(
                            list(filename_network_topology_metric_statistics_from_nEphys['Centrality nEphys'])[id_nEphys])
                        result_degree_nEphys.append(
                            list(filename_network_topology_metric_statistics_from_nEphys['Degree nEphys'])[id_nEphys])
                        result_betweenness_nEphys.append(
                            list(filename_network_topology_metric_statistics_from_nEphys['Betweenness nEphys'])[id_nEphys])
                        nEphys_coordinate.append(
                            list(filename_network_topology_metric_statistics_from_nEphys['Coordinates in nEphys'])[id_nEphys])

                    if id in clustering_coefficient.keys():
                        result_clustering_coefficient.append(clustering_coefficient[id])
                        result_degree.append(df_new.degree(id))
                        result_centrality.append(degree_centrality[id])
                        result_betweenness.append(betweenness_centrality[id])
                    else:
                        result_clustering_coefficient.append(0)
                        result_degree.append(0)
                        result_centrality.append(0)
                        result_betweenness.append(0)

                    cluster.append(self.clusters[i])

        a = {'Channel ID': channel_ID, 'Barcodes': barcodes, 'Coordinates in nEphys': nEphys_coordinate,
             'Clustering Coefficient SRT': result_clustering_coefficient,
             'Clustering Coefficient nEphys': result_clustering_coefficient_nEphys,
             "Centrality SRT": result_centrality, "Centrality nEphys": result_centrality_nEphys,
             "Degree SRT": result_degree, "Degree nEphys": result_degree_nEphys, 'Betweenness SRT': result_betweenness,
             'Betweenness nEphys': result_betweenness_nEphys, 'Cluster': cluster}
        df = pd.DataFrame.from_dict(a, orient='index').T
        df.to_excel(self.srcfilepath + gene_list_name + '_SRT_nEphys_network_topological_metrics_per_cluster' + ".xlsx", index=False)

    def SRT_network_topology_hub_rich_club_plot(self, percent=0.05, gene_list_name=None):
        """
        Calcuate number of hub nodes and rich clubs in the SRT network topology.

            File input needed:
            -------
                - related files
                - '[gene_list]_SRT_mutual_information_connectivity.xlsx'

            Parameters
            -------

            Returns
            -------

            File output:
            -------
                - '[gene_list]_SRT_hub_rich_club_map_functional_links_without_clusters_[%].png'
                - '[gene_list]_SRT_hub_rich_club_map_functional_links_with_clusters_[%].png'
                - '[gene_list]_SRT_hub_rich_club_node_list_[%].xlsx'
                - '[gene_list]_SRT_hub_rich_club_node_statistics_[%].xlsx'
                - '[gene_list]_SRT_hub_rich_club_node_statistics_[%].png'

        """
        path = self.srcfilepath[:self.srcfilepath.rfind('/')]
        desfilepath = path + '/Correlated_Network_Topological_Metrics/'
        if not os.path.exists(desfilepath):
            os.mkdir(desfilepath)
        # Read related information
        csv_file, tissue_lowres_scalef, features_name, matr_raw, barcodes, img, csv_file_cluster = self.read_related_files()
        barcode_cluster = np.asarray(csv_file_cluster["Barcode"])
        nEphys_cluster = np.asarray(csv_file_cluster["Loupe Clusters"])
        color = ['red' if i == 1 else 'black' for i in csv_file['selection']]
        label = {1: 'Detect points', 0: 'Background'}
        cdict = {1: 'red', 0: 'black'}
        scatter_x = np.asarray(csv_file["pixel_x"] * tissue_lowres_scalef)
        scatter_y = np.asarray(csv_file["pixel_y"] * tissue_lowres_scalef)
        group = np.asarray(csv_file["selection"])
        barcode_CSV = np.asarray(csv_file["barcode"])
        g = 1
        ix = np.where(group == g)
        #################################################################################Filters:
        # Remove spots with fewer than 1000 unique genes
        # Remove mitochondrial genes and ribosomal protein coding genes
        import re

        gene_name = [str(features_name[i])[2:-1] for i in range(len(features_name))]

        filter_gene_id = [i for i in range(len(gene_name)) if
                          len(re.findall(r'^SRS', gene_name[i], flags=re.IGNORECASE)) > 0 or len(
                              re.findall(r'^Mrp', gene_name[i], flags=re.IGNORECASE)) > 0 or len(
                              re.findall(r'^Rp', gene_name[i], flags=re.IGNORECASE)) > 0 or len(
                              re.findall(r'^mt', gene_name[i], flags=re.IGNORECASE)) > 0 or len(
                              re.findall(r'^Ptbp', gene_name[i], flags=re.IGNORECASE)) > 0]
        gene_name = [gene_name[i] for i in range(len(gene_name)) if i not in filter_gene_id]
        matr = np.delete(matr_raw, filter_gene_id, axis=0)
        # calculate UMIs and genes per cell
        umis_per_cell = np.asarray(matr.sum(axis=0)).squeeze()  # Or matr.sum(axis=0)
        genes_per_cell = np.asarray((matr > 0).sum(axis=0)).squeeze()

        ###############delete the nodes with less then 1000 gene count
        deleted_notes = [i for i in range(len(genes_per_cell)) if genes_per_cell[i] <= 1000]
        ###############delete the nodes not in clusters
        deleted_notes_cluster = [i for i in range(len(genes_per_cell)) if
                                 str(barcodes[i])[2:-1] not in barcode_cluster]
        deleted_notes.extend(deleted_notes_cluster)
        deleted_notes = list(np.unique(deleted_notes))
        ##########################################################
        matr = np.delete(matr, deleted_notes, axis=1)
        new_id = [j for i in barcode_CSV for j in range(len(barcodes)) if str(barcodes[j])[2:-1] == i]
        #########################
        x_filter = [scatter_x[ix][i] for i in range(len(scatter_x[ix])) if new_id[i] not in deleted_notes]
        y_filter = [scatter_y[ix][i] for i in range(len(scatter_y[ix])) if new_id[i] not in deleted_notes]
        barcodes_filter = [str(barcodes[i])[2:-1] for i in range(len(barcodes)) if i not in deleted_notes]
        #####################

        mask_id = [i for i in range(len(group)) if group[i] == 1]
        extent = [min([scatter_x[i] for i in mask_id]), max([scatter_x[i] for i in mask_id]),
                  min([scatter_y[i] for i in mask_id]), max([scatter_y[i] for i in mask_id])]

        img_cut = img[int(extent[2]):int(extent[3]) + 2, int(extent[0]):int(extent[1]) + 3,
                  :]  # x and y value set to cut the areas interested
        x_coordinate, y_coordinate = x_filter - extent[0], y_filter - extent[2]
        #########################################################
        filetype_xlsx = gene_list_name + '_SRT_mutual_information_connectivity' + ".xlsx"
        filename_xlsx, Root = self.get_filename_path(self.srcfilepath, filetype_xlsx)
        if len(filename_xlsx) > 0:
            data = pd.read_excel(Root[0] + '/' + filename_xlsx[0])
            data_new = data.copy()
            df = nx.from_pandas_edgelist(data_new, source='Channel ID', target='Corr_id', edge_attr=True)

            #     #########################################################hub nodes detection
            # betweenness_centrality_indices = self.hub_detect(df)
            # betweenness_centrality_indices = np.unique(betweenness_centrality_indices)
            node_strength_nodes = self.node_strength_detect(df, percent=percent)
            node_strength_nodes = np.unique(node_strength_nodes)
            cluster_coefficient_nodes = self.clustering_coefficient_detect(df, percent=percent)
            cluster_coefficient_nodes = np.unique(cluster_coefficient_nodes)
            efficiency_nodes = self.efficiency_detect(df, data_new['Channel ID'], data_new['Corr_id'],percent=percent)
            efficiency_nodes = np.unique(efficiency_nodes)
            hub_nodes_collect = []
            # hub_nodes_collect.extend(betweenness_centrality_indices)
            hub_nodes_collect.extend(node_strength_nodes)
            hub_nodes_collect.extend(cluster_coefficient_nodes)
            hub_nodes_collect.extend(efficiency_nodes)
            #############################################
            s = pd.Series(hub_nodes_collect)
            hub_nodes = list(s.value_counts(normalize=False).index)
            hub_nodes = [int(i) for i in hub_nodes]
            hub_scores = list(s.value_counts(normalize=False).values)
            hub_scores = [int(i) * 2 for i in hub_scores]
            # colorMap_hub = self.get_Groupid_SRT(hub_nodes, ChsGroups=ChsGroups, MeaStreams=MeaStreams)
            ###################################################
            Degree_1 = [key[1] for key in df.degree()]
            channel_ID_1 = [values[0] for values in df.degree()]
            rcc = nx.rich_club_coefficient(df, normalized=False)
            ###############################
            rcc_sort = sorted(rcc.items(), key=lambda k: k[1], reverse=True)
            Value = [rcc_sort[i][1] for i in range(len(rcc_sort))]
            Index = [rcc_sort[i][0] for i in range(len(rcc_sort))]
            ###########################################
            # Value_temp = [i for i in Value if i < 1]
            Value_temp = Value
            median = np.average(Value_temp)
            Temp = [Index[i] for i in range(len(Value)) if Value[i] <= median]
            index = Temp[0]
            ###########################################
            rich_club_node = [channel_ID_1[i] for i in range(len(Degree_1)) if Degree_1[i] >= index]
            rich_club_node = [i for i in rich_club_node if i in hub_nodes]
            rich_club_node_size = [hub_scores[j] for i in rich_club_node for j in range(len(hub_nodes)) if i == hub_nodes[j]]
            # colorMap_rich_club = self.get_Groupid_SRT(rich_club_node, ChsGroups=ChsGroups,
            #                                           MeaStreams=MeaStreams)
            # ############################
            New_coordinates_1 = np.asarray([[x_coordinate[i], y_coordinate[i]] for i in hub_nodes])
            rich_club_coordinates = np.asarray([[x_coordinate[i], y_coordinate[i]] for i in rich_club_node])
            ############all links
            hub_rich_club_nodes_all = hub_nodes + rich_club_node
            cluster_ids, clusters_ID, colorMap_hub_rich_club_nodes_all = self.get_Groupid_SRT(hub_rich_club_nodes_all, barcode_cluster=barcode_cluster, nEphys_cluster=nEphys_cluster, data=data_new)
            New_coordinates = np.asarray([[x_coordinate[i], y_coordinate[i]] for i in hub_rich_club_nodes_all])
            hub_rich_club_nodes_all = np.unique(hub_rich_club_nodes_all)
            data_filter = [[data['Channel ID'][i], data['Corr_id'][i]] for i in range(len(data))]
            data_all_links = [data for data in data_filter if
                              data[0] in hub_rich_club_nodes_all and data[1] in hub_rich_club_nodes_all]

            df_all_links = pd.DataFrame({'Channel ID': [int(i[0]) for i in data_all_links],
                                         'Corr_id': [int(i[1]) for i in data_all_links]})
            # print(type([int(i[0]) for i in data_all_links]),type([int(i[0]) for i in data_all_links][0]))
            df_new_all_links = nx.from_pandas_edgelist(df_all_links, source='Channel ID', target='Corr_id')
            # esmall = [(u, v) for (u, v, d) in df_new_all_links.edges(data=True)]
            # esmall = [(u, v) for (u, v, d) in df_new_all_links.edges(data=True) if d["weight"] >= np.mean(data['coor_Per_data'])]
            node_color_filter = [v for v in df_new_all_links.nodes]
            # print(len(node_color_filter))
            dic_all_links = {}
            for i in range(len(node_color_filter)):
                dic_all_links.update({node_color_filter[i]: (
                x_coordinate[node_color_filter[i]], y_coordinate[node_color_filter[i]])})

            # fig, (ax_1, ax) = plt.subplots(2, 1, figsize=(15, 20))
            fig, ax_1 = plt.subplots()
            hub_nodes = [hub_nodes[i] for i in range(len(hub_nodes)) if hub_nodes[i] in node_color_filter and hub_scores[i]>= max(hub_scores) - 0]  ##########################change here to show more or less nodes and links based on hub score
            max_value = max(hub_scores)
            print(f"The maximum value is: {max_value}")
            # hub_nodes = list(df.nodes()) # Show all hub nodes/rich club nodes.
            hub_scores = [hub_scores[i] for i in range(len(hub_nodes)) if hub_nodes[i] in node_color_filter]

            nodes = nx.draw_networkx_nodes(df_new_all_links, pos=dic_all_links, nodelist=hub_nodes, ax=ax_1, node_color='blue', label='Hub Nodes', node_size=hub_scores)
            nodes.set_edgecolor('none')
            rich_club_node = [i for i in rich_club_node if i in node_color_filter and i in hub_nodes]
            nx.draw_networkx_nodes(df_new_all_links, pos=dic_all_links, ax=ax_1, nodelist=rich_club_node,
                                   node_color='None', node_size=rich_club_node_size, label='Rich Club Nodes',
                                   node_shape=markers.MarkerStyle(marker='o', fillstyle='none'), alpha=0.7,
                                   edgecolors='red', linewidths=0.5)
            nodeset = set(hub_nodes)
            edgelist = [edge for edge in df_new_all_links.edges() if edge[0] in nodeset and edge[1] in nodeset]
            nx.draw_networkx_edges(df_new_all_links, pos=dic_all_links, ax=ax_1, width=0.2, alpha=0.2,
                                   edge_color='grey', edgelist=edgelist)
            ax_1.imshow(img_cut, alpha=1)
            ax_1.set_xlabel('Pixel')
            ax_1.set_ylabel('Pixel')
            # ax_1.set_aspect('equal', 'box')
            ax_1.legend(loc='best', fontsize='small')
            ax_1.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
            colorMapTitle = gene_list_name + '_SRT_hub_rich_club_map_functional_links_without_clusters_' + str(percent)
            fig.savefig(desfilepath + colorMapTitle + ".png", format='png', dpi=600)
            plt.close()
            ######################################################################################
            fig, ax = plt.subplots()
            cm = plt.cm.get_cmap('Set1', len(self.clusters))  # Paired_r
            bul = ax.scatter(New_coordinates[:, 0], New_coordinates[:, 1],
                             c=colorMap_hub_rich_club_nodes_all, cmap=cm, marker='o',
                             edgecolors='None', s=hub_scores + rich_club_node_size, alpha=1)
            ax.imshow(img_cut, alpha=1)
            ax.scatter(New_coordinates_1[:, 0], New_coordinates_1[:, 1], c='', marker='o',
                       edgecolors='blue',
                       linewidth=0.4, s=hub_scores, label='Hub nodes')
            ax.scatter(rich_club_coordinates[:, 0], rich_club_coordinates[:, 1], c='', marker='o',
                       edgecolors='darkred',
                       linewidth=0.5, s=rich_club_node_size, label='Rich-club nodes')
            nx.draw_networkx_edges(df_new_all_links, pos=dic_all_links, ax=ax, width=0.1, alpha=0.1,
                                   edge_color='grey')
            ax.set_xlabel('Pixel')
            ax.set_ylabel('Pixel')
            bul.set_clim(1, len(self.clusters) + 1)
            cbar = fig.colorbar(bul, extend='both', ticks=range(1, len(self.clusters) + 1),
                                label='Clusters',
                                shrink=.7)
            cbar.ax.set_yticklabels(self.clusters)

            ax.legend(loc='upper left', fontsize='xx-small')
            # ax.set_aspect('equal', 'box')
            ax.grid(False)
            ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
            colorMapTitle = gene_list_name + '_SRT_hub_rich_club_map_functional_links_with_clusters_' + str(percent)
            fig.savefig(desfilepath + colorMapTitle + ".png", format='png', dpi=600)
            plt.close()
            # ##################################################
            a = {'Hub nodes': np.unique(hub_nodes), 'Rich club nodes': np.unique(rich_club_node)}
            dataframe = pd.DataFrame.from_dict(a, orient='index').T
            dataframe.to_excel(
                desfilepath + gene_list_name + '_SRT_hub_rich_club_node_list_' + str(percent) + '.xlsx',
                index=False)
            #########################################################################################################
            hub_clusters = []
            rich_club_clusters = []
            for i in range(len(clusters_ID)):
                hub_cluster = 0
                rich_club_cluster = 0
                for j in hub_nodes:
                    if j in clusters_ID[i]:
                        hub_cluster += 1
                for k in rich_club_node:
                    if k in clusters_ID[i]:
                        rich_club_cluster += 1
                hub_clusters.append(hub_cluster)
                rich_club_clusters.append(rich_club_cluster)
            x = np.arange(len(self.clusters))
            fig, ax = plt.subplots(figsize=(15, 10))
            ax.bar(x, hub_clusters, width=0.2, label='Hub Nodes')
            for a, b in zip(x, hub_clusters):
                ax.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=7)
            plt.bar(x + 0.2, rich_club_clusters, width=0.2, label='Rich Club Nodes')
            for a, b in zip(x + 0.2, rich_club_clusters):
                ax.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=7)
            ax.set_xticks(x)
            ax.set_xticklabels(self.clusters)
            ax.set_ylabel('# of Nodes')
            ax.legend()
            colorMapTitle = gene_list_name + '_SRT_hub_rich_club_node_statistics_' + str(percent)
            fig.savefig(desfilepath + colorMapTitle + ".png", format='png', dpi=600)
            plt.close()
            #####################################
            dataframe = pd.DataFrame(
                {'Clusters': self.clusters, 'Hub nodes': hub_clusters,
                 'Rich club nodes': rich_club_clusters})
            dataframe.to_excel(
                desfilepath + gene_list_name + '_SRT_hub_rich_club_node_statistics_' + str(percent) + '.xlsx',
                index=False)

    def SRT_network_topology_hub_rich_club_plot_selected_statistics(self,percent=0.05):
        """

            File input needed:
            -------
                - '[gene_list]_hub_rich_club_node_statistics_[%].xlsx'

            Parameters
            -------

            Returns
            -------

            File output:
            -------
                - '[gene_list]_SRT_hub_rich_club_node_statistics_all.xlsx'
                - '[gene_list]_SRT_hub_rich_club_node_statistics_all.png'
        """

        path = self.srcfilepath[:self.srcfilepath.rfind('/')]
        desfilepath = path + '/Correlated_Network_Topological_Metrics/'
        if not os.path.exists(desfilepath):
            os.mkdir(desfilepath)
        #################read hub rich club values
        filetype_hub = '_SRT_hub_rich_club_node_statistics_' + str(percent) + '.xlsx'
        filename_hub, Root = self.get_filename_path(self.srcfilepath, filetype_hub)
        cluster_all = []
        hub_count = []
        rich_club_count = []
        gene_name_list = []
        for gene_list in column_list:
            for i in range(len(filename_hub)):
                if filename_hub[i][0] != '.' and gene_list in filename_hub[i]:
                    hub_root = Root[i] + '/' + filename_hub[i]
                    cluster_all.extend(pd.read_excel(hub_root)['Clusters'])
                    hub_count.extend(pd.read_excel(hub_root)['Hub nodes'])
                    rich_club_count.extend(pd.read_excel(hub_root)['Rich club nodes'])
                    gene_name_list.extend([gene_list]*len(pd.read_excel(hub_root)['Clusters']))

        dataframe = pd.DataFrame(
            {'Clusters': cluster_all, 'Hub nodes':hub_count,
             'Rich club nodes': rich_club_count,'Gene list name':gene_name_list})
        dataframe.to_excel(desfilepath + 'SRT_hub_rich_club_node_statistics_all' + '.xlsx',index=False)
        ##########################
        fig = plt.figure()
        ax = fig.add_subplot(2, 1, 1)
        # x = 'Clusters', y = "Numbers", hue = "Conditions:", data = data, order = order, ci = None
        sns.barplot(x='Clusters', y='Hub nodes', hue='Gene list name', data=dataframe, hue_order = column_list, ci = None,ax=ax)  # RUN PLOT
        # ax.set_title('Hub Count', fontsize=8)
        ax.legend(loc='best', fontsize='xx-small')
        ax.set_xlabel('')
        ax1 = fig.add_subplot(2, 1, 2)
        sns.barplot(x='Clusters', y='Rich club nodes', hue='Gene list name', data=dataframe, hue_order = column_list, ci = None,
                    ax=ax1)  # RUN PLOT
        # ax1.set_title('Rich Club Count', fontsize=8)
        ax1.legend(loc='best', fontsize='xx-small')
        ax1.set_xlabel('')
        fig.savefig(desfilepath + 'SRT_hub_rich_club_node_statistics_all' + ".png", format='png', dpi=600)
        plt.close()

    def nEphys_network_topology_hub_rich_club_plot(self, percent=0.05):
        """
        Calcuate number of hub nodes and rich clubs in the SRT network topology.

            File input needed:
            -------
                - '[file]_nEphys_functional_connectivity.xlsx'

            Parameters
            -------

            Returns
            -------

            File output:
            -------
                - '[file]_nEphys_hub_rich_club_map_functional_links_without_clusters_[%].png'
                - '[file]_nEphys_hub_rich_club_map_functional_links_with_clusters_[%].png'
                - '[file]_nEphys_hub_rich_club_node_list_[%].xlsx'
                - '[file]_nEphys_hub_rich_club_node_statistics_[%].xlsx'
                - '[file]_nEphys_hub_rich_club_node_statistics_[%].png'

        """

        path = self.srcfilepath[:self.srcfilepath.rfind('/')]
        desfilepath = path + '/Correlated_Network_Topological_Metrics/'
        if not os.path.exists(desfilepath):
            os.mkdir(desfilepath)
        filetype_bxr = '.bxr'
        filename_bxr, Root = self.get_filename_path(self.srcfilepath, filetype_bxr)

        def to_excel(file_list, i):
            expFile = file_list[i]
            if expFile[0] != '.':
                print(expFile)
                if expFile[0] != '.':
                    filehdf5_bxr = h5py.File(self.srcfilepath + expFile, 'r')  # read LFPs bxr files
                    ChsGroups = np.asarray(filehdf5_bxr["3BUserInfo"]["ChsGroups"])
                    if type(ChsGroups['Name'][0]) != str:
                        ChsGroups['Name'] = [i.decode("utf-8") for i in ChsGroups['Name']]
                    MeaChs2ChIDsVector = np.asarray(filehdf5_bxr["3BResults"]["3BInfo"]["MeaChs2ChIDsVector"])
                    MeaStreams = np.asarray(filehdf5_bxr["3BRecInfo"]["3BMeaStreams"]["Raw"]["Chs"])
                    #########################################################
                    filetype_xlsx = expFile[:-4] + '_nEphys_functional_connectivity.xlsx'
                    filename_xlsx, Root = self.get_filename_path(self.srcfilepath, filetype_xlsx)
                    if len(filename_xlsx) > 0:
                        data = pd.read_excel(Root[0] + '/' + filename_xlsx[0])
                        data_new = data.copy()
                        data_new = data_new[data_new['coor_Per_data'] > 0]
                        new_Row = MeaChs2ChIDsVector["Col"] - 1
                        new_Col = MeaChs2ChIDsVector["Row"] - 1
                        df = nx.from_pandas_edgelist(data_new, source='Class_LFPs_ID_Per', target='Corr_ID_Per',
                                                     edge_attr=True)
                        cluster_ids = []
                        clusters_ID = []
                        ########################################################
                        for i in range(len(self.clusters)):
                            cluster_ID = []
                            for k in range(len(ChsGroups['Name'])):
                                if ChsGroups['Name'][k] == self.clusters[i]:
                                    for j in range(len(ChsGroups['Chs'][k])):
                                        cluster_ids.append(
                                            (ChsGroups['Chs'][k][j][0] - 1) * 64 + (ChsGroups['Chs'][k][j][1] - 1))
                                        cluster_ID.append(
                                            (ChsGroups['Chs'][k][j][0] - 1) * 64 + (ChsGroups['Chs'][k][j][1] - 1))
                            clusters_ID.append(cluster_ID)
                        #########################################################hub nodes detection
                        # betweenness_centrality_indices = self.hub_detect(df)
                        # betweenness_centrality_indices = np.unique(betweenness_centrality_indices)
                        node_strength_nodes = self.node_strength_detect(df, percent=percent)
                        node_strength_nodes = np.unique(node_strength_nodes)
                        cluster_coefficient_nodes = self.clustering_coefficient_detect(df, percent=percent)
                        cluster_coefficient_nodes = np.unique(cluster_coefficient_nodes)
                        efficiency_nodes = self.efficiency_detect(df, data_new['Class_LFPs_ID_Per'],
                                                                  data_new['Corr_ID_Per'], percent=percent)
                        efficiency_nodes = np.unique(efficiency_nodes)
                        hub_nodes_collect = []
                        # hub_nodes_collect.extend(betweenness_centrality_indices)
                        hub_nodes_collect.extend(node_strength_nodes)
                        hub_nodes_collect.extend(cluster_coefficient_nodes)
                        hub_nodes_collect.extend(efficiency_nodes)
                        #############################################
                        s = pd.Series(hub_nodes_collect)
                        hub_nodes = list(s.value_counts(normalize=False).index)
                        hub_nodes = [int(i) for i in hub_nodes]
                        hub_scores = list(s.value_counts(normalize=False).values)
                        hub_scores = [int(i) * 2 for i in hub_scores]
                        # colorMap_hub = self.get_Groupid_nEphys_hub(hub_nodes, ChsGroups=ChsGroups, MeaStreams=MeaStreams)
                        ###################################################
                        Degree_1 = [key[1] for key in df.degree()]
                        channel_ID_1 = [values[0] for values in df.degree()]
                        rcc = nx.rich_club_coefficient(df, normalized=False)
                        ###############################
                        rcc_sort = sorted(rcc.items(), key=lambda k: k[1], reverse=True)
                        Value = [rcc_sort[i][1] for i in range(len(rcc_sort))]
                        Index = [rcc_sort[i][0] for i in range(len(rcc_sort))]
                        ###########################################
                        # Value_temp = [i for i in Value if i < 1]
                        Value_temp = Value
                        median = np.std(Value_temp)
                        Temp = [Index[i] for i in range(len(Value)) if Value[i] <= median]
                        index = Temp[0]
                        ###########################################
                        rich_club_node = [channel_ID_1[i] for i in range(len(Degree_1)) if Degree_1[i] >= index]
                        rich_club_node = [i for i in rich_club_node if i in hub_nodes]
                        rich_club_node_size = [hub_scores[j] for i in rich_club_node for j in range(len(hub_nodes)) if
                                               i == hub_nodes[j]]
                        # colorMap_rich_club = self.get_Groupid_nEphys_hub(rich_club_node, ChsGroups=ChsGroups,
                        #                                           MeaStreams=MeaStreams)
                        # ############################
                        New_coordinates_1 = np.asarray([[new_Row[i], new_Col[i]] for i in hub_nodes])
                        rich_club_coordinates = np.asarray([[new_Row[i], new_Col[i]] for i in rich_club_node])
                        ############all links
                        hub_rich_club_nodes_all = hub_nodes + rich_club_node
                        colorMap_hub_rich_club_nodes_all = self.get_Groupid_nEphys_hub(hub_rich_club_nodes_all,ChsGroups=ChsGroups,MeaStreams=MeaStreams)
                        New_coordinates = np.asarray([[new_Row[i], new_Col[i]] for i in hub_rich_club_nodes_all])
                        hub_rich_club_nodes_all = np.unique(hub_rich_club_nodes_all) # Change here to show unique or overlapping hub/richclub nodes
                        data_filter = [[data['Class_LFPs_ID_Per'][i], data['Corr_ID_Per'][i]] for i in range(len(data))
                                       if data['coor_Per_data'][i] > 0]
                        data_all_links = [data for data in data_filter if
                                          data[0] in hub_rich_club_nodes_all and data[1] in hub_rich_club_nodes_all]

                        df_all_links = pd.DataFrame({'Class_LFPs_ID_Per': [int(i[0]) for i in data_all_links],
                                                     'Corr_ID_Per': [int(i[1]) for i in data_all_links]})
                        # print(type([int(i[0]) for i in data_all_links]),type([int(i[0]) for i in data_all_links][0]))
                        df_new_all_links = nx.from_pandas_edgelist(df_all_links, source='Class_LFPs_ID_Per',
                                                                   target='Corr_ID_Per')
                        # esmall = [(u, v) for (u, v, d) in df_new_all_links.edges(data=True)]
                        # esmall = [(u, v) for (u, v, d) in df_new_all_links.edges(data=True) if d["weight"] >= np.mean(data['coor_Per_data'])]
                        node_color_filter = [v for v in df_new_all_links.nodes]
                        # print(len(node_color_filter))
                        dic_all_links = {}
                        for i in range(len(node_color_filter)):
                            dic_all_links.update(
                                {node_color_filter[i]: (new_Row[node_color_filter[i]], new_Col[node_color_filter[i]])})

                        # fig, (ax_1, ax) = plt.subplots(2, 1, figsize=(15, 20))
                        fig, ax_1 = plt.subplots()
                        hub_nodes = [hub_nodes[i] for i in range(len(hub_nodes)) if hub_nodes[i] in node_color_filter and hub_scores[i] >= max(hub_scores) -0]  ########################## change here to show more or less nodes and links based on hub score
                        max_value = max(hub_scores)
                        print(f"The maximum value is: {max_value}")
                        # hub_nodes = list(df.nodes()) # Show all hub nodes/rich club nodes.
                        hub_scores = [hub_scores[i] for i in range(len(hub_nodes)) if hub_nodes[i] in node_color_filter]

                        nodes = nx.draw_networkx_nodes(df_new_all_links, pos=dic_all_links, nodelist=hub_nodes, ax=ax_1,
                                                       node_color='blue', label='Hub Nodes', node_size=hub_scores)
                        nodes.set_edgecolor('none')
                        rich_club_node = [i for i in rich_club_node if
                                          i in node_color_filter and i in hub_nodes]
                        nx.draw_networkx_nodes(df_new_all_links, pos=dic_all_links, ax=ax_1,
                                               nodelist=rich_club_node,
                                               node_color='None', node_size=rich_club_node_size,
                                               label='Rich Club Nodes',
                                               node_shape=markers.MarkerStyle(marker='o', fillstyle='none'), alpha=0.7,
                                               edgecolors='red', linewidths=0.5)
                        nodeset = set(hub_nodes)
                        edgelist = [edge for edge in df_new_all_links.edges() if
                                    edge[0] in nodeset and edge[1] in nodeset]
                        nx.draw_networkx_edges(df_new_all_links, pos=dic_all_links, ax=ax_1, width=0.2, alpha=0.2,
                                               edge_color='grey', edgelist=edgelist)
                        ax_1.set_ylim(64, 0)
                        ax_1.set_xlim(0, 64)
                        ax_1.set_xlabel('electrode')
                        ax_1.set_ylabel('electrode')
                        ax_1.set_aspect('equal', 'box')
                        ax_1.legend(loc='best', fontsize='small')
                        ax_1.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
                        colorMapTitle = expFile[:-4] + '_nEphys_hub_rich_club_map_functional_links_without_clusters_' + str(
                            percent)
                        fig.savefig(desfilepath + colorMapTitle + ".png", format='png', dpi=600)
                        plt.close()
                        ######################################################################################
                        fig, ax = plt.subplots()
                        cm = plt.cm.get_cmap('Set1', len(ChsGroups['Name']))  # Paired_r
                        bul = ax.scatter(New_coordinates[:, 0], New_coordinates[:, 1],
                                         c=colorMap_hub_rich_club_nodes_all, cmap=cm, marker='o',
                                         edgecolors='None', s=hub_scores + rich_club_node_size, alpha=1)
                        ax.scatter(New_coordinates_1[:, 0], New_coordinates_1[:, 1], c='', marker='o',
                                   edgecolors='blue',
                                   linewidth=0.4, s=hub_scores, label='Hub nodes')
                        ax.scatter(rich_club_coordinates[:, 0], rich_club_coordinates[:, 1], c='', marker='o',
                                   edgecolors='darkred',
                                   linewidth=0.5, s=rich_club_node_size, label='Rich-club nodes')
                        nx.draw_networkx_edges(df_new_all_links, pos=dic_all_links, ax=ax, width=0.1, alpha=0.1,
                                               edge_color='grey')
                        ax.set_xlabel('electrode')
                        ax.set_ylabel('electrode')
                        bul.set_clim(1, len(ChsGroups['Name']) + 1)
                        cbar = fig.colorbar(bul, extend='both', ticks=range(1, len(ChsGroups['Name']) + 1),
                                            label='Clusters',
                                            shrink=.7)
                        cbar.ax.set_yticklabels(ChsGroups['Name'])
                        ax.set_ylim(64, 0)
                        ax.set_xlim(0, 64)
                        ax.legend(loc='upper left', fontsize='xx-small')
                        ax.set_aspect('equal', 'box')
                        ax.grid(False)
                        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
                        colorMapTitle = expFile[:-4] + '_nEphys_hub_rich_club_map_functional_links_with_clusters_' + str(
                            percent)
                        fig.savefig(desfilepath + colorMapTitle + ".png", format='png', dpi=600)
                        plt.close()
                        # ##################################################
                        a = {'Hub nodes': np.unique(hub_nodes), 'Rich club nodes': np.unique(rich_club_node)}
                        dataframe = pd.DataFrame.from_dict(a, orient='index').T
                        dataframe.to_excel(desfilepath + expFile[:-4] + '_nEphys_hub_rich_club_node_list_' + str(percent) + '.xlsx',
                            index=False)
                        #########################################################################################################
                        hub_clusters = []
                        rich_club_clusters = []
                        for i in range(len(clusters_ID)):
                            hub_cluster = 0
                            rich_club_cluster = 0
                            for j in hub_nodes:
                                if j in clusters_ID[i]:
                                    hub_cluster += 1
                            for k in rich_club_node:
                                if k in clusters_ID[i]:
                                    rich_club_cluster += 1
                            hub_clusters.append(hub_cluster)
                            rich_club_clusters.append(rich_club_cluster)
                        x = np.arange(len(self.clusters))
                        fig, ax = plt.subplots(figsize=(15, 10))
                        ax.bar(x, hub_clusters, width=0.2, label='Hub Nodes')
                        for a, b in zip(x, hub_clusters):
                            ax.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=7)
                        plt.bar(x + 0.2, rich_club_clusters, width=0.2, label='Rich Club Nodes')
                        for a, b in zip(x + 0.2, rich_club_clusters):
                            ax.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=7)
                        ax.set_xticks(x)
                        ax.set_xticklabels(self.clusters)
                        ax.set_ylabel('# of Nodes')
                        ax.legend()
                        colorMapTitle = expFile[:-4] + '_nEphys_hub_rich_club_node_statistics'
                        fig.savefig(desfilepath + colorMapTitle + ".png", format='png', dpi=600)
                        plt.close()
                        #####################################
                        dataframe = pd.DataFrame(
                            {'Clusters': self.clusters, 'Hub nodes': hub_clusters,
                             'Rich club nodes': rich_club_clusters})
                        dataframe.to_excel(desfilepath + expFile[:-4] + '_nEphys_hub_rich_club_node_statistics_' + str(percent) + '.xlsx',index=False)

        threads = []
        x = 0
        for t in range(len(filename_bxr)):
            t = threading.Thread(target=to_excel, args=(filename_bxr, x))
            threads.append(t)
            x += 1

        for thr in threads:
            thr.start()
            # for thr in threads:
            thr.join()
        print("all over")

    def network_topology_characterization_with_degree_distribution_regional(self,gene_list_name=None,choose_gene = False,type = None):
        """
        Characterize degree distribution of nodes in nEphys and SRT network topology. Plots regionally i.e. hippo and cortex separate.
            File input needed:
            -------
                - '[gene_list]_gene_expression_per_cluster.xlsx'
                - '[gene_list]_SRT_nEphys_network_topological_metrics_per_cluster.xlsx'
            Parameters
            -------

            Returns
            -------

            File output:
            -------
                - '[gene_list]_logarithmic_fit_cCDF_condition_[nEphys or SRT]_Degree_[region].xlsx
                - '[gene_list]_Pareto_linear_binning_condition_[nEphys or SRT]_Degree_[region].xlsx
                - '[gene_list]_Pareto_linear_binning_condition_[nEphys or SRT]_Degree_[region].png
                - '[gene_list]_logarithmic_fit_cCDF_regional_Degree.xlsx'
                - '[gene_list]_logarithmic_fit_cCDF_regional_Degree.png'
        """

        path = self.srcfilepath[:self.srcfilepath.rfind('/')]
        desfilepath = path + '/Network_Topology_Characterization/'
        if not os.path.exists(desfilepath):
            os.mkdir(desfilepath)
        colorMapTitle = gene_list_name + '_logarithmic_fit_cCDF_regional_'
        type_name_nEphys = type + ' nEphys'
        type_name_SRT = type + ' SRT'
        if os.path.exists(desfilepath + colorMapTitle + type + ".xlsx"):
            df_final = pd.read_excel(desfilepath + colorMapTitle + type + ".xlsx")
        else:
            filetype_gene_expression_per_cluster = gene_list_name + '_gene_expression_per_cluster' + ".xlsx"
            filename_gene_expression_per_cluster, Root_gene_expression_per_cluster = self.get_filename_path(self.srcfilepath, filetype_gene_expression_per_cluster)
            filetype_SRT_nEphys_network_topological_metrics_per_cluster = gene_list_name + '_SRT_nEphys_network_topological_metrics_per_cluster' + ".xlsx"
            filename_SRT_nEphys_network_topological_metrics_per_cluster, Root_SRT_nEphys_network_topological_metrics_per_cluster = self.get_filename_path(self.srcfilepath,filetype_SRT_nEphys_network_topological_metrics_per_cluster)
            parameter_values_SRT= []
            parameter_values_nEphys = []
            parameter_values_Con = []
            parameter_values_Cluster = []
            parameter_values_Region = []
            # Gene_name = []
            for con in conditions:
                final_gene_expression_per_cluster = pd.DataFrame()
                for i in range(len(Root_gene_expression_per_cluster)):
                    if con in Root_gene_expression_per_cluster[i] and filename_gene_expression_per_cluster[i][0]!='.':
                        gene_expression_per_cluster = pd.read_excel(Root_gene_expression_per_cluster[i]+'/' + filename_gene_expression_per_cluster[i])
                        final_gene_expression_per_cluster = pd.concat([final_gene_expression_per_cluster,gene_expression_per_cluster], axis=0).reset_index()
                final_SRT_nEphys_network_topological_metrics_per_cluster = pd.DataFrame()
                for j in range(len(Root_SRT_nEphys_network_topological_metrics_per_cluster)):
                    if con in Root_SRT_nEphys_network_topological_metrics_per_cluster[j] and filename_SRT_nEphys_network_topological_metrics_per_cluster[j][0] != '.':
                        SRT_nEphys_network_topological_metrics_per_cluster = pd.read_excel(Root_SRT_nEphys_network_topological_metrics_per_cluster[j]+'/' +filename_SRT_nEphys_network_topological_metrics_per_cluster[j])
                        final_SRT_nEphys_network_topological_metrics_per_cluster = pd.concat([final_SRT_nEphys_network_topological_metrics_per_cluster, SRT_nEphys_network_topological_metrics_per_cluster], axis=0).reset_index()


                df_new_con = final_SRT_nEphys_network_topological_metrics_per_cluster.copy()
                df_gene_expression_per_cluster = final_gene_expression_per_cluster.copy()
                Region_add =  []
                for clu in list(df_new_con["Cluster"]):
                    if clu in Cortex:
                        Region_add.append('Cortex')
                    elif clu in Hippo:
                        Region_add.append('Hippo')
                    else:
                        Region_add.append('Not in Region')
                Region_add = ['Cortex' if clu in Cortex else 'Hippo' for clu in list(df_new_con["Cluster"])]
                Gene_Expression_Level = []
                # gene_name = []
                for bar in df_new_con['Barcodes']:
                    df_new_con_bar = df_gene_expression_per_cluster.copy()
                    df_new_con_bar = df_new_con_bar[df_new_con_bar['Barcode'] == bar]
                    s = pd.Series(range(len(df_new_con_bar)))
                    df_new_con_bar = df_new_con_bar.set_index(s)
                    if choose_gene == False:
                        Gene_Expression_Level.append(np.mean(df_new_con_bar['Gene Expression Level']))
                        # gene_name.append(stats.mode(np.array(df_new_con_bar['gene Name']))[0][0])
                    else:
                        Gene_Expression_Level.append(df_new_con_bar[df_new_con_bar['gene Name'] == choose_gene]['Gene Expression Level'])
                        # gene_name.append(df_new_con_bar[df_new_con_bar['gene Name'] == choose_gene]['gene Name'])

                df_add = pd.DataFrame({'Gene Expression Level':Gene_Expression_Level,'Region': Region_add})
                final = pd.concat([df_new_con, df_add], axis=1)
                # final[type_name] = final[type_name].astype(float, errors='raise')
                final = final[final['Region'] != 'Not in Region']
                s = pd.Series(range(len(final)))
                final = final.set_index(s)
                final['Gene Expression Level'] = final['Gene Expression Level'].astype(float,errors='raise')
                # final.to_excel(desfilepath + colorMapTitle +con + ".xlsx", index=False)


                parameter_values_SRT.extend(final[type_name_SRT])
                parameter_values_nEphys.extend(final[type_name_nEphys])
                parameter_values_Con.extend([con]*len(final[type_name_nEphys]))
                parameter_values_Cluster.extend(final['Cluster'])
                parameter_values_Region.extend(final['Region'])

            a = {type_name_SRT: parameter_values_SRT, type_name_nEphys: parameter_values_nEphys, 'Condition': parameter_values_Con, 'Cluster': parameter_values_Cluster, 'Region': parameter_values_Region}
            df_final = pd.DataFrame.from_dict(a, orient='index').T
            df_final.to_excel(desfilepath + colorMapTitle + type + ".xlsx",index=False)

        ###########################################
        plt.figure(figsize=(15, 8))
        Region_count = 0
        for region in Region:
            ax = plt.subplot(121 + Region_count, frameon=False)
            df_new_Region = df_final.copy()
            df_new_Region = df_new_Region[df_new_Region['Region'] == region]
            s = pd.Series(range(len(df_new_Region)))
            df_new_Region = df_new_Region.set_index(s)
            for con in range(len(conditions)):
                df_con = df_new_Region.copy()
                df_con = df_con[df_con['Condition'] == conditions[con]]
                s = pd.Series(range(len(df_con)))
                df_con = df_con.set_index(s)

                max_value_raw = max(max([i for i in df_con[type_name_SRT] if ~np.isnan(i)]),max([i for i in df_con[type_name_nEphys] if ~np.isnan(i)]))
                max_value = 1
                Count_SRT = [i/max_value for i in df_con[type_name_SRT] if ~np.isnan(i)]
                Count_nEphys = [i/max_value for i in df_con[type_name_nEphys] if ~np.isnan(i)]
                self.Pareto_plot_linear_binning_hist(bin_num=100, min_degree=1, max_degree=1000,Count=list(Count_SRT), color_choose=color_choose, number=con,index=0,name = 'SRT',gene_list_name=gene_list_name,type = type,Region = region)
                self.ccdf_plot(ax=ax, Count=list(Count_SRT), color_choose=color_choose, number=con,index=0,name = 'SRT',gene_list_name=gene_list_name,type = type,max_value_raw = max_value_raw,Region = region)
                self.ccdf_plot(ax=ax, Count=list(Count_nEphys), color_choose=color_choose, number=con,index=1,name = 'nEphys',gene_list_name=gene_list_name,type = type,max_value_raw = max_value_raw,Region = region)
                self.Pareto_plot_linear_binning_hist(bin_num=100, min_degree=1, max_degree=1000,
                                                     Count=list(Count_nEphys), color_choose=color_choose, number=con,index=1,name = 'nEphys',gene_list_name=gene_list_name,type = type,Region = region)
            ###############################
            ax.legend(fontsize=4,loc='upper right', frameon=False, borderaxespad=0.)
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_xlabel('Degree, k')
            # ax.set_ylim(ymin=0.0001,ymax=1.1)
            ax.set_ylabel('Complementary Cumulative Distribution p(k)', fontsize=8, fontweight="bold")
            ax.set_title(Region)
            Region_count += 1
        plt.tight_layout()
        plt.savefig(desfilepath + colorMapTitle + type + ".png", format='png', dpi=600)
        plt.close()

    def network_topology_characterization_with_degree_distribution_pooled(self,gene_list_name=None,choose_gene = False,type = None):
        """
        Characterize degree distribution of nodes in nEphys and SRT network topology. Plots regionally i.e. hippo and cortex separate.
            File input needed:
            -------
                - '[gene_list]_gene_expression_per_cluster.xlsx'
                - '[gene_list]_SRT_nEphys_network_topological_metrics_per_cluster.xlsx'
            Parameters
            -------

            Returns
            -------

            File output:
            -------
                - '[gene_list]_logarithmic_fit_cCDF_condition_[nEphys or SRT]_Degree_pooled.xlsx
                - '[gene_list]_Pareto_linear_binning_condition_[nEphys or SRT]_Degree_pooled.xlsx
                - '[gene_list]_Pareto_linear_binning_condition_[nEphys or SRT]_Degree_pooled.png
                - '[gene_list]_logarithmic_fit_cCDF_pooled_Degree.xlsx'
                - '[gene_list]_logarithmic_fit_cCDF_pooled_Degree.png'
        """

        path = self.srcfilepath[:self.srcfilepath.rfind('/')]
        desfilepath = path + '/Network_Topology_Characterization/'
        if not os.path.exists(desfilepath):
            os.mkdir(desfilepath)
        colorMapTitle = gene_list_name + '_logarithmic_fit_cCDF_pooled_'
        type_name_nEphys = type + ' nEphys'
        type_name_SRT = type + ' SRT'
        if os.path.exists(desfilepath + colorMapTitle + type + ".xlsx"):
            df_final = pd.read_excel(desfilepath + colorMapTitle + type + ".xlsx")
        else:
            filetype_gene_expression_per_cluster = gene_list_name + '_gene_expression_per_cluster' + ".xlsx"
            filename_gene_expression_per_cluster, Root_gene_expression_per_cluster = self.get_filename_path(self.srcfilepath, filetype_gene_expression_per_cluster)
            filetype_SRT_nEphys_network_topological_metrics_per_cluster = gene_list_name + '_SRT_nEphys_network_topological_metrics_per_cluster' + ".xlsx"
            filename_SRT_nEphys_network_topological_metrics_per_cluster, Root_SRT_nEphys_network_topological_metrics_per_cluster = self.get_filename_path(self.srcfilepath,filetype_SRT_nEphys_network_topological_metrics_per_cluster)
            parameter_values_SRT= []
            parameter_values_nEphys = []
            parameter_values_Con = []
            parameter_values_Cluster = []
            parameter_values_Region = []
            # Gene_name = []
            for con in conditions:
                final_gene_expression_per_cluster = pd.DataFrame()
                for i in range(len(Root_gene_expression_per_cluster)):
                    if con in Root_gene_expression_per_cluster[i] and filename_gene_expression_per_cluster[i][0]!='.':
                        gene_expression_per_cluster = pd.read_excel(Root_gene_expression_per_cluster[i]+'/' + filename_gene_expression_per_cluster[i])
                        final_gene_expression_per_cluster = pd.concat([final_gene_expression_per_cluster,gene_expression_per_cluster], axis=0).reset_index()
                final_SRT_nEphys_network_topological_metrics_per_cluster = pd.DataFrame()
                for j in range(len(Root_SRT_nEphys_network_topological_metrics_per_cluster)):
                    if con in Root_SRT_nEphys_network_topological_metrics_per_cluster[j] and filename_SRT_nEphys_network_topological_metrics_per_cluster[j][0] != '.':
                        SRT_nEphys_network_topological_metrics_per_cluster = pd.read_excel(Root_SRT_nEphys_network_topological_metrics_per_cluster[j]+'/' +filename_SRT_nEphys_network_topological_metrics_per_cluster[j])
                        final_SRT_nEphys_network_topological_metrics_per_cluster = pd.concat([final_SRT_nEphys_network_topological_metrics_per_cluster, SRT_nEphys_network_topological_metrics_per_cluster], axis=0).reset_index()


                df_new_con = final_SRT_nEphys_network_topological_metrics_per_cluster.copy()
                df_gene_expression_per_cluster = final_gene_expression_per_cluster.copy()
                Region_add =  []
                for clu in list(df_new_con["Cluster"]):
                    if clu in Cortex:
                        Region_add.append('Cortex')
                    elif clu in Hippo:
                        Region_add.append('Hippo')
                    else:
                        Region_add.append('Not in Region')
                # Region_add = ['Cortex' if clu in Cortex else 'Hippo' for clu in list(df_new_con["Cluster"])]
                Gene_Expression_Level = []
                # gene_name = []
                for bar in df_new_con['Barcodes']:
                    df_new_con_bar = df_gene_expression_per_cluster.copy()
                    df_new_con_bar = df_new_con_bar[df_new_con_bar['Barcode'] == bar]
                    s = pd.Series(range(len(df_new_con_bar)))
                    df_new_con_bar = df_new_con_bar.set_index(s)
                    if choose_gene == False:
                        Gene_Expression_Level.append(np.mean(df_new_con_bar['Gene Expression Level']))
                        # gene_name.append(stats.mode(np.array(df_new_con_bar['gene Name']))[0][0])
                    else:
                        Gene_Expression_Level.append(df_new_con_bar[df_new_con_bar['gene Name'] == choose_gene]['Gene Expression Level'])
                        # gene_name.append(df_new_con_bar[df_new_con_bar['gene Name'] == choose_gene]['gene Name'])

                df_add = pd.DataFrame({'Gene Expression Level':Gene_Expression_Level,'Region': Region_add})
                final = pd.concat([df_new_con, df_add], axis=1)
                # final[type_name] = final[type_name].astype(float, errors='raise')
                final = final[final['Region'] != 'Not in Region']
                s = pd.Series(range(len(final)))
                final = final.set_index(s)
                final['Gene Expression Level'] = final['Gene Expression Level'].astype(float,errors='raise')
                # final.to_excel(self.srcfilepath + colorMapTitle +con + ".xlsx", index=False)


                parameter_values_SRT.extend(final[type_name_SRT])
                parameter_values_nEphys.extend(final[type_name_nEphys])
                parameter_values_Con.extend([con]*len(final[type_name_nEphys]))
                parameter_values_Cluster.extend(final['Cluster'])
                parameter_values_Region.extend(final['Region'])

            a = {type_name_SRT: parameter_values_SRT, type_name_nEphys: parameter_values_nEphys, 'Condition': parameter_values_Con, 'Cluster': parameter_values_Cluster, 'Region': parameter_values_Region}
            df_final = pd.DataFrame.from_dict(a, orient='index').T
            df_final.to_excel(desfilepath + colorMapTitle + type + ".xlsx",index=False)

        ###########################################
        plt.figure(figsize=(15, 8))
        ax = plt.subplot(111, frameon=False)
        df_new_Region = df_final.copy()
        # df_new_Region = df_new_Region[df_new_Region['Region'] == Region]
        # s = pd.Series(range(len(df_new_Region)))
        # df_new_Region = df_new_Region.set_index(s)
        for con in range(len(conditions)):
            df_con = df_new_Region.copy()
            df_con = df_con[df_con['Condition'] == conditions[con]]
            s = pd.Series(range(len(df_con)))
            df_con = df_con.set_index(s)

            max_value_raw = max(max([i for i in df_con[type_name_SRT] if ~np.isnan(i)]),max([i for i in df_con[type_name_nEphys] if ~np.isnan(i)]))
            max_value = 1
            Count_SRT = [i/max_value for i in df_con[type_name_SRT] if ~np.isnan(i)]
            Count_nEphys = [i/max_value for i in df_con[type_name_nEphys] if ~np.isnan(i)]
            self.Pareto_plot_linear_binning_hist(bin_num=100, min_degree=1, max_degree=1000,Count=list(Count_SRT), color_choose=color_choose, number=con,index=0,name = 'SRT',gene_list_name=gene_list_name,type = type,Region = 'pooled')
            self.ccdf_plot(ax=ax, Count=list(Count_SRT), color_choose=color_choose, number=con,index=0,name = 'SRT',gene_list_name=gene_list_name,type = type,max_value_raw = max_value_raw,Region = 'pooled')
            self.ccdf_plot(ax=ax, Count=list(Count_nEphys), color_choose=color_choose, number=con,index=1,name = 'nEphys',gene_list_name=gene_list_name,type = type,max_value_raw = max_value_raw,Region = 'pooled')
            self.Pareto_plot_linear_binning_hist(bin_num=100, min_degree=1, max_degree=1000,
                                                 Count=list(Count_nEphys), color_choose=color_choose, number=con,index=1,name = 'nEphys',gene_list_name=gene_list_name,type = type,Region = 'pooled')
            ###############################
            ax.legend(fontsize=4,loc='upper right', frameon=False, borderaxespad=0.)
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_xlabel('Degree, k')
            # ax.set_ylim(ymin=0.0001,ymax=1.1)
            ax.set_ylabel('Complementary Cumulative Distribution p(k)', fontsize=8, fontweight="bold")
            # ax.set_title(Region)
            # Region_count += 1
        plt.tight_layout()
        plt.savefig(desfilepath + colorMapTitle + type + ".png", format='png', dpi=600)
        plt.close()

    def Pareto_plot_linear_binning_hist(self,bin_num=300, min_degree=30, max_degree=900,Count=None,color_choose = None,number=0,index=0,name = 'ST',gene_list_name=None,type = None,Region = None):  # b:  shape parameter
        """plot the Pareto fit.MLEs(Maximum Likelihood Estimate) for shape (if applicable), location, and scale parameters from data.To shift and/or scale the distribution use the loc and scale parameters. """
        # min_degree, max_degree = min_degree/max_degree, max_degree/max_degree
        path = self.srcfilepath[:self.srcfilepath.rfind('/')]
        desfilepath = path + '/Network_Topology_Characterization/'
        if not os.path.exists(desfilepath):
            os.mkdir(desfilepath)
        color_index = number * 2 + index
        fig, ax = plt.subplots()
        # weights = np.ones_like(Count) / float(len(Count))
        weights = np.ones_like(Count) / float(len(Count))
        (hist, bins, _) = ax.hist(Count, bins=np.linspace(min(Count), max(Count), bin_num),
                                          label=conditions[number]+'_'+name, color=color_choose[color_index], alpha=0.3,
                                          weights=weights)
        parameter = stats.pareto.fit([i for i in Count]) # if i <= max_degree and i >= min_degree # MLEs(Maximum Likelihood Estimate) for shape (if applicable), location, and scale parameters from data.
        node_id = [i for i in range(len(hist))]# if hist[i] > min_node_threshold
        x = [bins[:-1][i] for i in node_id]
        y = [hist[i] for i in node_id]

        Hist = stats.pareto.pdf(x, b=parameter[0], loc=parameter[1], scale=parameter[2])
        Hist = [i/np.sum(Hist) for i in Hist]
        ax.plot(x, Hist, color=color_choose[color_index],label='Pareto fit', linewidth=2)

        a = {'Raw_Degree': x, 'Raw_hist': y, 'pareto Fit Degree': x, 'Fit Hist': Hist,
             'Shape parameter': [parameter[0]]}
        df = pd.DataFrame.from_dict(a, orient='index').T
        df.to_excel(desfilepath + gene_list_name+ '_Pareto_linear_binning_' + conditions[number]+'_'+name+'_'+type+'_'+Region + ".xlsx", index=False)

        ax.legend(fontsize=4,loc='upper right', frameon=False, borderaxespad=0.)
        ax.set_xlabel('Degree, k')
        ax.set_ylabel('p(k)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax1.spines['bottom'].set_visible(False)
        # ax1.spines['left'].set_visible(False)
        # ax.set_ylim(ymin=-0.001)
        # ax.set_xlim(xmin=0, xmax=1000)
        ax.set_xlim(xmin=min_degree, xmax=max_degree)
        colorMapTitle = gene_list_name+ '_Pareto_linear_binning_' + conditions[number]+'_'+name+'_'+type+'_'+Region  # 'Truncated_Power_law_for_connection_log_binning'
        fig.savefig(desfilepath + colorMapTitle + ".png", format='png', dpi=600)
        plt.close()

    def ccdf_plot(self, ax=None, Count=None, color_choose=None, number=0, index=0, name='SRT', gene_list_name=None,type=None, max_value_raw=0, Region=None):
        #################################################
        path = self.srcfilepath[:self.srcfilepath.rfind('/')]
        desfilepath = path + '/Network_Topology_Characterization/'
        if not os.path.exists(desfilepath):
            os.mkdir(desfilepath)
        color_index = number * 2 + index
        count_log = [i for i in np.log(Count) if ~np.isinf(i)]
        import powerlaw
        Count = np.array(Count) / max_value_raw
        param = stats.lognorm.fit(Count)
        unique_data, unique_indices = np.unique(Count, return_index=True)
        bins = unique_data
        bins = np.sort(np.asarray(bins))

        # bins = np.linspace(min(Count),max(Count),100)
        cdf_fit = stats.lognorm.sf(bins, param[0], loc=param[1], scale=param[2])
        bins = bins * max_value_raw
        ax.plot(bins, cdf_fit, color=color_choose[color_index],
                label='logarithmic_fit' + '[' + r'$\mu$' + '=' + str(
                    round(np.mean(count_log), 1)) + ', ' + r'$\sigma$' + '=' + str(
                    round(np.std(count_log), 1)) + ']', alpha=0.3)
        ############################
        Count = np.array(Count) * max_value_raw
        data, CDF = powerlaw.cdf(Count, survival=True)
        from sklearn.metrics import r2_score
        r2 = r2_score(CDF, cdf_fit)
        # ax_connect.plot(x[::-1], p, color=color_choose[con],label='Truncated power law fit['+ r'$\beta$' +'(exponent)'+ '=' + str(round(popt[1], 2))+', '+ r'$\kappa$' + '(cutoff value)'+'=' + str(round(popt[2], 2))+']',linewidth=1)
        powerlaw.plot_ccdf(Count, survival=True, color=color_choose[color_index], label=conditions[number] + '_' + name,
                           marker='x', linewidth=0, markersize=2, ax=ax, alpha=1)

        # X, prop = powerlaw.cdf(Count, survival=True)
        a = {'Raw_Degree(bins)': data, 'Raw_cCDF': CDF, 'Fit Degree(bins)': bins, 'Fit cCDF': cdf_fit,
             'mu(log)': [np.mean(count_log)], 'sigma(log)': [np.std(count_log)], 'R2 Values': [r2]}
        df = pd.DataFrame.from_dict(a, orient='index').T
        df.to_excel(desfilepath + gene_list_name + '_logarithmic_fit_cCDF_' + conditions[
            number] + '_' + name + '_' + type + '_' + Region + ".xlsx", index=False)

if __name__ == '__main__':
    srcfilepath = r'Z:/ANALYSES/SPATIOSCALES- 10X genomics/Data/'  # main path
    ############Basic Statistic####################################
    Analysis = MEASeqX_Project(srcfilepath)
    ################################################################# Calculate network functional connectivity and plot gephi dynamic maps
    Analysis.nEphys_functional_connectivity(low=1, high=100, Analysis_Item='LFPs', Start_Time=0,Stop_Time=60000) # Step 1 individual
    Analysis.nEphys_functional_connectivity_excel_to_gephi() # Step 2 individual
    for gene_list in column_list: # Step 3 individual
        Analysis.SRT_mutual_information_connectivity_excel_to_gephi(gene_list_name=gene_list)
    ################################################################# nEphys and SRT Network Topological Metrics
    Analysis.network_topological_metrics() # Step 4 individual
    Analysis.coordinates_for_network_topological_metrics()  # Step 5 individual
    for gene_list in column_list: # Step 6 individual
        Analysis.network_topological_feature_statistics_per_node(gene_list_name=gene_list)
    ################################################################# Network topology characterization - hub rich club nodes
    for gene_list in column_list: # Step 7 individual
        Analysis.SRT_network_topology_hub_rich_club_plot(percent=0.20,gene_list_name=gene_list)
    Analysis.SRT_network_topology_hub_rich_club_plot_selected_statistics(percent=0.20) # Step 8 individual
    Analysis.nEphys_network_topology_hub_rich_club_plot(percent=0.20)  # Step 9 individual
    ################################################################# Network topology characterization - degree distribution
    Analysis.network_topology_characterization_with_degree_distribution_regional(gene_list_name='IEGs', choose_gene=False,type='Degree') # Step 7 pooled condition plotting (main path should contain the condition subfolders)
    Analysis.network_topology_characterization_with_degree_distribution_pooled(gene_list_name='IEGs', choose_gene=False, type='Degree') # Step 7 pooled condition plotting (main path should contain the condition subfolders)




