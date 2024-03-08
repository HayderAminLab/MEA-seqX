# -*- coding: utf-8 -*-
"""
Created on Dec 12 2021
@author:  BIONICS_LAB
@company: DZNE
"""
import matplotlib.pyplot as plt
from anndata import AnnData
import sklearn as sk
from sklearn.decomposition import PCA
from scipy.signal import butter, lfilter, hilbert
from scipy import signal,stats
from sklearn import preprocessing
import scanpy as sc
import h5py
import numpy as np
import pandas as pd
import json
from ast import literal_eval
import os
import seaborn as sns
import scipy.sparse as sp_sparse
import matplotlib.image as mpimg
import statsmodels.stats.multitest as sm
from matplotlib.axes._axes import _log as matplotlib_axes_logger
import help_functions.LFP_denoising as LFP_denoising
matplotlib_axes_logger.setLevel('ERROR')

"""
The following input parameters are used for specific gene list and multiple gene plotting correlated with network activity features. 
To compare conditions put the path for datasets in the input parameters and label condition name i.e. SD and ENR and assign desired color

"""

rows = 64
cols = 64

column_list = ["IEGs"]  ### "Hipoo Signaling Pathway","Synaptic Vescicles_Adhesion","Receptors and channels","Synaptic plasticity","Hippocampal Neurogenesis","IEGs"
select_genes = ['Arc', 'Bdnf', 'Egr1', 'Egr3', 'Egr4', 'Fosb']  ### >5 genes needed

conditions = ['SD', 'ENR']
condition1_path = r'Z:/ANALYSES/SPATIOSCALES- 10X genomics/Data/SD/'
condition2_path = r'Z:/ANALYSES/SPATIOSCALES- 10X genomics/Data/ENR/'

color = ['silver', 'dodgerblue']  # color for pooled plotting of conditions
color_choose = ['silver', 'dodgerblue','red','green','purple'] # color for multiple gene plots

network_activity_feature = ['LFPRate','Delay','Energy','Frequency','Amplitude','positive_peaks','negative_peaks','positive_peak_count','negative_peak_count','CT','CV2','Fano']

quantile_value = 0.75

Prediction_Limits = 0.8

Region = ['Cortex', 'Hippo']
Cortex = ['EC', 'PC']
Hippo = ['DG','Hilus','CA3','CA1']

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
        Get network activity feature raw data
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

    def quantile_bound(self,data):#quantile = 0
        # if quantile != 0:
        #     quantile_value = quantile
        s = pd.Series(data)
        return s.quantile(0.5), s.quantile(0.5-quantile_value/2), s.quantile(0.5+quantile_value/2)

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

    def relation_p_values(self, varx=None, vary=None):
        if len(varx) > 0:
            if isinstance(varx[1], float):
                varx = varx
                vary = vary
            else:
                varx = [i[0] for i in varx]
                print("vary",vary)
                try:
                    vary = [i[0] for i in vary]
                except:
                    vary = vary
            print(varx[1])
            keep_id = [i for i in range(len(varx)) if varx[i] == varx[i] and vary[i] == vary[i]]
            varx = np.asarray(varx)[keep_id]
            vary = np.asarray(vary)[keep_id]
            if len(varx) > 0:
                # print(varx)
                # print(vary)
                # mask = ~np.isnan(varx) & ~np.isnan(vary) & ~np.isinf(varx) & ~np.isinf(vary)
                # print('percentage:',len([i for i in varx[mask] if i == 0])/len(varx[mask]))
                # if len([i for i in varx if i == 0])/len(varx[mask]) > 0.5:
                #     corr, p_value = 0,0
                # else:
                corr, p_value = stats.spearmanr(varx, vary)
            else:
                corr, p_value = 0, 0
        else:
            corr, p_value = 0, 0
        # slope, intercept, r, p, se = stats.linregress(varx[mask], vary[mask])
        return corr, p_value

    def get_fft(self,snippet,samplingrate, low=1, high=100):
        analytic_signal = hilbert(snippet)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0 * np.pi) * samplingrate)
        y = [value for value in instantaneous_frequency if value <= high and value >= low]

        return y

    def bibiplot(self,score = None, coeff = None, labels=None, cluster_num=None,ax = None,xlabel = 1,ylabel = 2):
        xs = score[:, 0]
        ys = score[:, 1]
        n = coeff.shape[0]
        scalex = 1.0 / (xs.max() - xs.min())
        scaley = 1.0 / (ys.max() - ys.min())
        cm = plt.cm.get_cmap('Set1', len(np.unique(cluster_num)))

        ax.scatter(np.asarray(xs) * scalex, np.asarray(ys) * scaley, c=np.asarray(cluster_num), cmap=cm)
        for i in range(n):
            # print('value:',coeff[i, 0], coeff[i, 1])
            ax.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='black', alpha=0.5)
            if labels is None:
                ax.text(coeff[i, 0] * 1.05, coeff[i, 1] * 1.05, "Var" + str(i + 1), color='black', ha='center',va='center')
            else:
                ax.text(coeff[i, 0] * 1.05, coeff[i, 1] * 1.05, labels[i], color='black', ha='center', va='center')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal', 'box')
        ax.set_ylabel("PC{}".format(ylabel))
        ax.set_xlabel("PC{}".format(xlabel))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.grid(False)
        ####################
        def legend_without_duplicate_labels(axes):
            handles, labels = axes.get_legend_handles_labels()
            unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
            axes.legend(*zip(*unique),fontsize='xx-small')

        legend_without_duplicate_labels(ax)

    def k_means(self,data, num_clus=3, steps=200):

        # Convert data to Numpy array
        cluster_data = np.array(data)

        # Initialize by randomly selecting points in the data
        center_init = np.random.randint(0, cluster_data.shape[0], num_clus)

        # Create a list with center coordinates
        center_init = cluster_data[center_init, :]

        # Repeat clustering  x times
        for _ in range(steps):

            # Calculate distance of each data point to cluster center
            distance = []
            for center in center_init:
                tmp_distance = np.sqrt(np.sum((cluster_data - center) ** 2, axis=1))

                # Adding smalle random noise to the data to avoid matching distances to centroids
                tmp_distance = tmp_distance + np.abs(np.random.randn(len(tmp_distance)) * 0.0001)
                distance.append(tmp_distance)

            # Assign each point to cluster based on minimum distance
            _, cluster = np.where(np.transpose(distance == np.min(distance, axis=0)))

            # Find center of mass for each cluster
            center_init = []
            for i in range(num_clus):
                center_init.append(cluster_data[cluster == i, :].mean(axis=0).tolist())

        return cluster, center_init, distance

    def network_activity_features(self, low=1, high=100):
        """
        Determine specified network activity feature from nEphys data.

            File input needed:
            -------
                - related files
                - '[file].bxr'
                - denoising files

            Parameters
            -------

            Returns
            -------

            File output:
            -------
                - '_network_activity_features_per_cluster.xlsx'
        """

        filehdf5_bxr_name = '.bxr'
        filehdf5_bxr_file, filehdf5_bxr_Root = self.get_filename_path(self.srcfilepath, filehdf5_bxr_name)

        for i in range(len(filehdf5_bxr_file)):
            if filehdf5_bxr_file[i][0] != '.':
                filehdf5_bxr_root = filehdf5_bxr_Root[i] + '/' + filehdf5_bxr_file[i]
                expFile = filehdf5_bxr_file[i]
        filehdf5_bxr = h5py.File(filehdf5_bxr_root, 'r')

        ChsGroups = np.asarray(filehdf5_bxr["3BUserInfo"]["ChsGroups"])
        if type(ChsGroups['Name'][0]) != str:
            ChsGroups['Name'] = [i.decode("utf-8") for i in ChsGroups['Name']]
        MeaChs2ChIDsVector = np.asarray(filehdf5_bxr["3BResults"]["3BInfo"]["MeaChs2ChIDsVector"])
        maxVolt = np.asarray(filehdf5_bxr["3BRecInfo"]["3BRecVars"]['MaxVolt'])[0]
        minVolt = np.asarray(filehdf5_bxr["3BRecInfo"]["3BRecVars"]['MinVolt'])[0]
        tQLevel = np.asarray(filehdf5_bxr["3BRecInfo"]["3BRecVars"]['BitDepth'])[0]
        QLevel = np.power(2, tQLevel)  # quantized levels corresponds to 2^num of bit to encode the signal
        fromQLevelToUVolt = (maxVolt - minVolt) / QLevel
        stepVolt = (maxVolt - minVolt) / maxVolt
        signalInversion = np.asarray(filehdf5_bxr["3BRecInfo"]["3BRecVars"]["SignalInversion"])[0]
        samplingRate = np.asarray(filehdf5_bxr["3BRecInfo"]["3BRecVars"]["SamplingRate"])[0]
        ########################################################################################################Open if data is already denoised; close to run denoising
        lfpChId_raw = np.asarray(filehdf5_bxr["3BResults"]["3BChEvents"]["LfpChIDs"])
        lfpTimes_raw = np.asarray(filehdf5_bxr["3BResults"]["3BChEvents"]["LfpTimes"]) / samplingRate
        LfpForms = np.asarray(filehdf5_bxr["3BResults"]["3BChEvents"]["LfpForms"])
        ########################################################################################################and open these two to run denoising
        Analysis = LFP_denoising.LFPAnalysis_Function(self.srcfilepath,condition_choose='BS')  # condition_choose ='OB' or 'BS'
        lfpChId_raw, lfpTimes_raw, LfpForms = Analysis.AnalyzeExp(expFile=expFile)
        active_id = np.unique(lfpChId_raw)
        # print(lfpTimes_raw[-1])
        cluster_ID = [(ChsGroups['Chs'][i][j][0] - 1) * 64 + (ChsGroups['Chs'][i][j][1] - 1) for i in range(len(ChsGroups['Chs'])) for j in range(len(ChsGroups['Chs'][i]))]
        cluster_Name = [ChsGroups['Name'][i] for i in range(len(ChsGroups['Chs'])) for j in range(len(ChsGroups['Chs'][i]))]

        lfpChId = [i for i in cluster_ID if i in active_id]
        cluster_Name = [cluster_Name[i] for i in range(len(cluster_ID)) if cluster_ID[i] in active_id]
        #################
        numLFPs = len(lfpChId_raw)  # extract the total number of detected LFPs
        LFPLength = len(LfpForms) // numLFPs  # extract the LFP length
        tempWaveForms = LfpForms
        tempWaveForms = np.array(tempWaveForms)
        tempWaveForms = tempWaveForms.reshape(numLFPs, LFPLength)[:]
        lfpChId_raw_se = pd.Series(list(lfpChId_raw))
        #############################
        Delay = []
        lfpChId_series = pd.Series(lfpChId_raw)
        energy_all = []

        index = pd.Index(list(lfpChId_raw))
        result = index.value_counts()
        LFP_rate = []
        CV2_all = []
        Fano = []
        positive_peaks = []
        negative_peaks = []
        positive_peak_count = []
        negative_peak_count = []
        CT = []
        amplitude = []
        frequency = []
        for id in lfpChId:
            try:
            # print('Check1')
                energy = 0.0
                positive_peak_count_value = 0
                negative_peak_count_value = 0
                Indecies = list(lfpChId_raw_se[lfpChId_raw_se == id].index)
                positive_peaks_temp = []
                negative_peaks_temp = []
                CT_temp = []
                wave_form_all = []
                for Index_id in Indecies:
                    wave = list((tempWaveForms[Index_id][:np.nonzero(tempWaveForms[Index_id])[0][-1]] - (4096.0 / 2)) * stepVolt * signalInversion)
                    wave = self.filter_data(wave, samplingRate, low=low, high=high)
                    wave_form_all.extend(wave)
                    for el in np.abs(wave):
                        energy = energy + el # Compute the absolute power by approximating the area under the curve

                    ########################################
                    zero_point_index = [index_0 for index_0 in range(len(wave) - 1) if
                                        (wave[index_0] < 0 and wave[index_0 + 1] >= 0) or (
                                                wave[index_0] >= 0 and wave[index_0 + 1] < 0)]
                    # positive_peak = [index_0 for index_0 in range(len(zero_point_index)) if wave[zero_point_index[index_0]+1]>=wave[zero_point_index[index_0]]]
                    positive_peak_temp = [index for index in range(1, len(wave) - 1) if wave[index] > wave[index - 1] and wave[index] > wave[index + 1]]
                    positive_peak_count_value+=len(positive_peak_temp)
                    negative_peak_temp = [index for index in range(1, len(wave) - 1) if wave[index] < wave[index - 1] and wave[index] < wave[index + 1]]
                    # negative_peak = [index_0 for index_0 in range(len(zero_point_index)) if wave[zero_point_index[index_0] + 1] < wave[zero_point_index[index_0]]]
                    negative_peak_count_value += len(negative_peak_temp)

                    CT_temp.append(np.mean([((zero_point_index[index_0 + 1] - zero_point_index[index_0]) - (
                            zero_point_index[index_0] - zero_point_index[index_0 - 1])) / (
                                               zero_point_index[index_0 + 1] - zero_point_index[index_0 - 1]) if
                                       wave[zero_point_index[index_0 - 1]] > wave[
                                           zero_point_index[index_0 - 1] + 1] else ((zero_point_index[index_0] -
                                                                                     zero_point_index[index_0 - 1]) - (
                                                                                            zero_point_index[
                                                                                                index_0 + 1] -
                                                                                            zero_point_index[
                                                                                                index_0])) / (
                                                                                           zero_point_index[
                                                                                               index_0 + 1] -
                                                                                           zero_point_index[
                                                                                               index_0 - 1]) for
                                       index_0 in range(1, len(zero_point_index) - 1)]))
                    # print(flank_midpoints)

                    positive_peaks_temp.append(max(wave))
                    negative_peaks_temp.append(min(wave))

                    #############################################

                freq = self.get_fft(wave_form_all, samplingRate, low=low, high=high)
                # freqs = np.max(freq)
                frequency.append(np.mean(freq))

                energy = energy / samplingRate
                energy_all.append(energy)
                LFP_rate.append(result[id] * 60 / (int(lfpTimes_raw[-1]) + 1))
                ###############################
                index_for_id = list(lfpChId_series[lfpChId_series == id].index)
                time_for_id = pd.Series([lfpTimes_raw[i] for i in index_for_id])
                #############
                Fano.append(stats.variation([lfpTimes_raw[i] for i in index_for_id]))
                positive_peaks.append(np.mean(positive_peaks_temp))
                negative_peaks.append(np.mean(negative_peaks_temp))
                positive_peak_count.append(positive_peak_count_value)
                negative_peak_count.append(negative_peak_count_value)

                if np.isnan(np.mean([i for i in CT_temp if i==i])):
                    CT.append(0)
                else:
                    CT.append(np.mean([i for i in CT_temp if i==i]))
                if abs(np.mean(positive_peaks_temp))>=abs(np.mean(negative_peaks_temp)):
                    amplitude.append(np.mean(positive_peaks_temp))
                else:
                    amplitude.append(np.mean(negative_peaks_temp))

                # IEI = pd.Series.diff(time_for_id)
                IEI = pd.Series.diff(pd.Series([0]+[lfpTimes_raw[i] for i in index_for_id]))
                ############################
                SRT = list([k for k in IEI if k==k])
                CV2 = [2 * abs(SRT[i] - SRT[i + 1]) / (SRT[i] + SRT[i + 1]) for i in range(len(SRT) - 1)]
                CV2_all.append(np.mean(CV2))
               ################################
                Delay.append(np.mean([i for i in IEI if ~np.isnan(i)]))
                print('Check5')

            except:
                Delay.append(0)
                energy_all.append(0)
                LFP_rate.append(0)
                CV2_all.append(0)
                Fano.append(0)
                positive_peaks.append(0)
                negative_peaks.append(0)
                positive_peak_count.append(0)
                negative_peak_count.append(0)
                CT.append(0)
                amplitude.append(0)
                frequency.append(0)
        new_Row = MeaChs2ChIDsVector["Col"] - 1
        new_Col = MeaChs2ChIDsVector["Row"] - 1
        print(LFP_rate)
        # print(len(positive_peak_count),len(negative_peak_count),len(CT),len(positive_peaks),len(negative_peaks),len(lfpChId),len(energy_all))
        df = pd.DataFrame({'Channel ID': lfpChId, 'LFP Rate(Event/min)': LFP_rate, 'Delay(s)': Delay,'Energy':energy_all,'Frequency':frequency, 'Amplitude(uV)': amplitude,
                           'Mean Positive Peaks(uV)': positive_peaks, 'Mean Negative Peaks(uV)': negative_peaks,'Positive Peak Count':positive_peak_count, 'Negative Peak Count': negative_peak_count,
                           'CT': CT, 'CV2': CV2_all, 'Fano Factor': Fano, 'Cluster': cluster_Name,'x_coordinate': [new_Row[i] for i in lfpChId],'y_coordinate': [new_Col[i] for i in lfpChId]})
        df.to_excel(self.srcfilepath + expFile[:-4] + "_network_activity_features_per_cluster" + ".xlsx", index=False)

    def coordinates_for_network_activity_features(self):
        """
        Provide the transcriptomic and electrophysiologic overlay coordinates for network activity features.

            File input needed:
            -------
                - related files
                - '[file].bxr'
                - 'SRT_nEphys_Multiscale_Coordinates.xlsx'

            Parameters
            -------

            Returns
            -------

            File output:
            -------
                - 'SRT_nEphys_Multiscale_Coordinates_for_network_activity_features.xlsx'
        """

        filetype_SRT_nEphys_Coordinates = 'SRT_nEphys_Multiscale_Coordinates.xlsx'
        filename_SRT_nEphys_Coordinates, Root = self.get_filename_path(self.srcfilepath, filetype_SRT_nEphys_Coordinates)
        for i in range(len(filename_SRT_nEphys_Coordinates)):
            if filename_SRT_nEphys_Coordinates[i][0] != '.':
                SRT_nEphys_Coordinates_root = Root[i] + '/' + filename_SRT_nEphys_Coordinates[i]

        data_SRT_nEphys_Coordinates = pd.read_excel(SRT_nEphys_Coordinates_root)

        filehdf5_bxr_name = '.bxr'
        filehdf5_bxr_file, filehdf5_bxr_Root = self.get_filename_path(self.srcfilepath, filehdf5_bxr_name)

        for i in range(len(filehdf5_bxr_file)):
            if filehdf5_bxr_file[i][0] != '.':
                filehdf5_bxr_root = filehdf5_bxr_Root[i] + '/' + filehdf5_bxr_file[i]
                expFile = filehdf5_bxr_file[i]
        data_nEphys = pd.read_excel(self.srcfilepath + expFile[:-4] + '_network_activity_features_per_cluster' + ".xlsx")
        filehdf5_bxr = h5py.File(filehdf5_bxr_root, 'r')
        MeaChs2ChIDsVector = np.asarray(filehdf5_bxr["3BResults"]["3BInfo"]["MeaChs2ChIDsVector"])
        coordinate_nEphys = [[MeaChs2ChIDsVector["Col"][id] - 1, MeaChs2ChIDsVector["Row"][id] - 1] for id in
                           data_nEphys['Channel ID']]
        cluster = []
        barcodes = []
        nEphys_coordinate = []
        result_Lfprate_nEphys = []
        result_delay_nEphys= []
        result_energy_nEphys = []
        result_frequency_nEphys = []
        result_amplitude_nEphys = []
        result_positive_peaks_nEphys = []
        result_negative_peaks_nEphys = []
        result_positive_peak_count_nEphys = []
        result_negative_peak_count_nEphys = []
        result_CT_nEphys = []
        result_CV2_nEphys = []
        result_Fano_nEphys = []

        for cor in range(len(data_SRT_nEphys_Coordinates['Barcodes'])): #####LFP Rate
            related_LFP_cor = literal_eval(data_SRT_nEphys_Coordinates['Coordinates in nEphys'][cor])
            if len(related_LFP_cor) > 0:
                result_Lfprate_nEphys_mean = []  # clustering coefficient
                for cor_in_nEphys in related_LFP_cor:
                    if cor_in_nEphys in list(coordinate_nEphys):
                        index_in_nEphys = list(coordinate_nEphys).index(cor_in_nEphys)
                        result_Lfprate_nEphys_mean.append(float(data_nEphys['LFP Rate(Event/min)'][index_in_nEphys]))
                    else:
                        result_Lfprate_nEphys_mean.append(0)
                cluster.append(data_SRT_nEphys_Coordinates['Cluster'][cor])
                barcodes.append(data_SRT_nEphys_Coordinates['Barcodes'][cor])
                nEphys_coordinate.append(related_LFP_cor)
                result_Lfprate_nEphys.append(np.mean([i for i in result_Lfprate_nEphys_mean if i == i]))

        for cor in range(len(data_SRT_nEphys_Coordinates['Barcodes'])): #####Delay
            related_delay_cor = literal_eval(data_SRT_nEphys_Coordinates['Coordinates in nEphys'][cor])
            if len(related_delay_cor) > 0:
                result_delay_nEphys_mean = []
                for cor_in_nEphys in related_delay_cor:
                    if cor_in_nEphys in list(coordinate_nEphys):
                        index_in_nEphys = list(coordinate_nEphys).index(cor_in_nEphys)
                        result_delay_nEphys_mean.append(float(data_nEphys['Delay(s)'][index_in_nEphys]))
                    else:
                        result_delay_nEphys_mean.append(0)
                cluster.append(data_SRT_nEphys_Coordinates['Cluster'][cor])
                barcodes.append(data_SRT_nEphys_Coordinates['Barcodes'][cor])
                nEphys_coordinate.append(related_delay_cor)
                result_delay_nEphys.append(np.mean([i for i in result_delay_nEphys_mean if i == i]))

        for cor in range(len(data_SRT_nEphys_Coordinates['Barcodes'])):  #####Energy
            related_energy_cor = literal_eval(data_SRT_nEphys_Coordinates['Coordinates in nEphys'][cor])
            if len(related_energy_cor) > 0:
                result_energy_nEphys_mean = []
                for cor_in_nEphys in related_energy_cor:
                    if cor_in_nEphys in list(coordinate_nEphys):
                        index_in_nEphys = list(coordinate_nEphys).index(cor_in_nEphys)
                        result_energy_nEphys_mean.append(float(data_nEphys['Energy'][index_in_nEphys]))
                    else:
                        result_energy_nEphys_mean.append(0)
                cluster.append(data_SRT_nEphys_Coordinates['Cluster'][cor])
                barcodes.append(data_SRT_nEphys_Coordinates['Barcodes'][cor])
                nEphys_coordinate.append(related_energy_cor)
                result_energy_nEphys.append(np.mean([i for i in result_energy_nEphys_mean if i == i]))

        for cor in range(len(data_SRT_nEphys_Coordinates['Barcodes'])): #####Frequency
            related_frequency_cor = literal_eval(data_SRT_nEphys_Coordinates['Coordinates in nEphys'][cor])
            if len(related_frequency_cor) > 0:
                result_frequency_nEphys_mean = []
                for cor_in_nEphys in related_frequency_cor:
                    if cor_in_nEphys in list(coordinate_nEphys):
                        index_in_nEphys = list(coordinate_nEphys).index(cor_in_nEphys)
                        result_frequency_nEphys_mean.append(float(data_nEphys['Frequency'][index_in_nEphys]))
                    else:
                        result_delay_nEphys_mean.append(0)
                cluster.append(data_SRT_nEphys_Coordinates['Cluster'][cor])
                barcodes.append(data_SRT_nEphys_Coordinates['Barcodes'][cor])
                nEphys_coordinate.append(related_delay_cor)
                result_frequency_nEphys.append(np.mean([i for i in result_frequency_nEphys_mean if i == i]))

        for cor in range(len(data_SRT_nEphys_Coordinates['Barcodes'])): #####Amplitude
            related_amplitude_cor = literal_eval(data_SRT_nEphys_Coordinates['Coordinates in nEphys'][cor])
            if len(related_amplitude_cor) > 0:
                result_amplitude_nEphys_mean = []
                for cor_in_nEphys in related_amplitude_cor:
                    if cor_in_nEphys in list(coordinate_nEphys):
                        index_in_nEphys = list(coordinate_nEphys).index(cor_in_nEphys)
                        result_amplitude_nEphys_mean.append(float(data_nEphys['Amplitude(uV)'][index_in_nEphys]))
                    else:
                        result_amplitude_nEphys_mean.append(0)
                cluster.append(data_SRT_nEphys_Coordinates['Cluster'][cor])
                barcodes.append(data_SRT_nEphys_Coordinates['Barcodes'][cor])
                nEphys_coordinate.append(related_delay_cor)
                result_amplitude_nEphys.append(np.mean([i for i in result_amplitude_nEphys_mean if i == i]))

        for cor in range(len(data_SRT_nEphys_Coordinates['Barcodes'])):  #####Positive Peaks
            related_positive_peaks_cor = literal_eval(data_SRT_nEphys_Coordinates['Coordinates in nEphys'][cor])
            if len(related_positive_peaks_cor) > 0:
                result_positive_peaks_nEphys_mean = []
                for cor_in_nEphys in related_positive_peaks_cor:
                    if cor_in_nEphys in list(coordinate_nEphys):
                        index_in_nEphys = list(coordinate_nEphys).index(cor_in_nEphys)
                        result_positive_peaks_nEphys_mean.append(
                            float(data_nEphys['Mean Positive Peaks(uV)'][index_in_nEphys]))
                    else:
                        result_positive_peaks_nEphys_mean.append(0)
                cluster.append(data_SRT_nEphys_Coordinates['Cluster'][cor])
                barcodes.append(data_SRT_nEphys_Coordinates['Barcodes'][cor])
                nEphys_coordinate.append(related_delay_cor)
                result_positive_peaks_nEphys.append(np.mean([i for i in result_positive_peaks_nEphys_mean if i == i]))

        for cor in range(len(data_SRT_nEphys_Coordinates['Barcodes'])):  #####Negative Peaks
            related_negative_peaks_cor = literal_eval(data_SRT_nEphys_Coordinates['Coordinates in nEphys'][cor])
            if len(related_negative_peaks_cor) > 0:
                result_negative_peaks_nEphys_mean = []
                for cor_in_nEphys in related_negative_peaks_cor:
                    if cor_in_nEphys in list(coordinate_nEphys):
                        index_in_nEphys = list(coordinate_nEphys).index(cor_in_nEphys)
                        result_negative_peaks_nEphys_mean.append(
                            float(data_nEphys['Mean Negative Peaks(uV)'][index_in_nEphys]))
                    else:
                        result_negative_peaks_nEphys_mean.append(0)
                cluster.append(data_SRT_nEphys_Coordinates['Cluster'][cor])
                barcodes.append(data_SRT_nEphys_Coordinates['Barcodes'][cor])
                nEphys_coordinate.append(related_delay_cor)
                result_negative_peaks_nEphys.append(np.mean([i for i in result_negative_peaks_nEphys_mean if i == i]))

        for cor in range(len(data_SRT_nEphys_Coordinates['Barcodes'])):  #####Positive Peak Count
            related_positive_peak_count_cor = literal_eval(data_SRT_nEphys_Coordinates['Coordinates in nEphys'][cor])
            if len(related_positive_peak_count_cor) > 0:
                result_positive_peak_count_nEphys_mean = []
                for cor_in_nEphys in related_positive_peak_count_cor:
                    if cor_in_nEphys in list(coordinate_nEphys):
                        index_in_nEphys = list(coordinate_nEphys).index(cor_in_nEphys)
                        result_positive_peak_count_nEphys_mean.append(
                            float(data_nEphys['Positive Peak Count'][index_in_nEphys]))
                    else:
                        result_positive_peak_count_nEphys_mean.append(0)
                cluster.append(data_SRT_nEphys_Coordinates['Cluster'][cor])
                barcodes.append(data_SRT_nEphys_Coordinates['Barcodes'][cor])
                nEphys_coordinate.append(related_delay_cor)
                result_positive_peak_count_nEphys.append(np.mean([i for i in result_positive_peak_count_nEphys_mean if i == i]))

        for cor in range(len(data_SRT_nEphys_Coordinates['Barcodes'])):  #####Negative Peak Count
            related_negative_peak_count_cor = literal_eval(data_SRT_nEphys_Coordinates['Coordinates in nEphys'][cor])
            if len(related_negative_peak_count_cor) > 0:
                result_negative_peak_count_nEphys_mean = []
                for cor_in_nEphys in related_negative_peak_count_cor:
                    if cor_in_nEphys in list(coordinate_nEphys):
                        index_in_nEphys = list(coordinate_nEphys).index(cor_in_nEphys)
                        result_negative_peak_count_nEphys_mean.append(
                            float(data_nEphys['Negative Peak Count'][index_in_nEphys]))
                    else:
                        result_negative_peak_count_nEphys_mean.append(0)
                cluster.append(data_SRT_nEphys_Coordinates['Cluster'][cor])
                barcodes.append(data_SRT_nEphys_Coordinates['Barcodes'][cor])
                nEphys_coordinate.append(related_delay_cor)
                result_negative_peak_count_nEphys.append(np.mean([i for i in result_negative_peak_count_nEphys_mean if i == i]))

        for cor in range(len(data_SRT_nEphys_Coordinates['Barcodes'])): #####CT
            related_CT_cor = literal_eval(data_SRT_nEphys_Coordinates['Coordinates in nEphys'][cor])
            if len(related_CT_cor) > 0:
                result_CT_nEphys_mean = []
                for cor_in_nEphys in related_CT_cor:
                    if cor_in_nEphys in list(coordinate_nEphys):
                        index_in_nEphys = list(coordinate_nEphys).index(cor_in_nEphys)
                        result_CT_nEphys_mean.append(float(data_nEphys['CT'][index_in_nEphys]))
                    else:
                        result_CT_nEphys_mean.append(0)
                cluster.append(data_SRT_nEphys_Coordinates['Cluster'][cor])
                barcodes.append(data_SRT_nEphys_Coordinates['Barcodes'][cor])
                nEphys_coordinate.append(related_delay_cor)
                result_CT_nEphys.append(np.mean([i for i in result_CT_nEphys_mean if i == i]))

        for cor in range(len(data_SRT_nEphys_Coordinates['Barcodes'])): #####CV2
            related_CV2_cor = literal_eval(data_SRT_nEphys_Coordinates['Coordinates in nEphys'][cor])
            if len(related_CV2_cor) > 0:
                result_CV2_nEphys_mean = []
                for cor_in_nEphys in related_CV2_cor:
                    if cor_in_nEphys in list(coordinate_nEphys):
                        index_in_nEphys = list(coordinate_nEphys).index(cor_in_nEphys)
                        result_CV2_nEphys_mean.append(float(data_nEphys['CV2'][index_in_nEphys]))
                    else:
                        result_CV2_nEphys_mean.append(0)
                cluster.append(data_SRT_nEphys_Coordinates['Cluster'][cor])
                barcodes.append(data_SRT_nEphys_Coordinates['Barcodes'][cor])
                nEphys_coordinate.append(related_delay_cor)
                result_CV2_nEphys.append(np.mean([i for i in result_CV2_nEphys_mean if i == i]))

        for cor in range(len(data_SRT_nEphys_Coordinates['Barcodes'])): #####Fano
            related_Fano_cor = literal_eval(data_SRT_nEphys_Coordinates['Coordinates in nEphys'][cor])
            if len(related_Fano_cor) > 0:
                result_Fano_nEphys_mean = []
                for cor_in_nEphys in related_Fano_cor:
                    if cor_in_nEphys in list(coordinate_nEphys):
                        index_in_nEphys = list(coordinate_nEphys).index(cor_in_nEphys)
                        result_Fano_nEphys_mean.append(float(data_nEphys['Fano Factor'][index_in_nEphys]))
                    else:
                        result_Fano_nEphys_mean.append(0)
                cluster.append(data_SRT_nEphys_Coordinates['Cluster'][cor])
                barcodes.append(data_SRT_nEphys_Coordinates['Barcodes'][cor])
                nEphys_coordinate.append(related_delay_cor)
                result_Fano_nEphys.append(np.mean([i for i in result_Fano_nEphys_mean if i == i]))

        a = {'Barcodes': barcodes, 'Coordinates in nEphys': nEphys_coordinate, 'LFPRate nEphys': result_Lfprate_nEphys, 'Delay nEphys': result_delay_nEphys, 'Energy nEphys': result_energy_nEphys,
             'Frequency nEphys': result_frequency_nEphys,'Amplitude nEphys': result_amplitude_nEphys, 'Positive Peaks nEphys': result_positive_peaks_nEphys, 'Negative Peaks nEphys': result_negative_peaks_nEphys,
             'Positive Peak Count nEphys': result_positive_peak_count_nEphys, 'Negative Peak Count nEphys': result_negative_peak_count_nEphys, 'CT nEphys': result_CT_nEphys, 'CV2 nEphys': result_CV2_nEphys, 'Fano nEphys': result_Fano_nEphys, 'Cluster': cluster}
        df = pd.DataFrame.from_dict(a, orient='index').T
        df.to_excel(self.srcfilepath + 'SRT_nEphys_Multiscale_Coordinates_for_network_activity_features' + ".xlsx", index=False)



    def gene_expression_network_activity_features_correlation(self, gene_list_name=None, network_activity_feature='LFPRate'):
        """
        Correlate specified network activity feature from nEphys data with gene expression values from SRT gene lists.

            File input needed:
            -------
                - '[file].bxr'
                - '[gene_list]_gene_expression_per_cluster.xlsx'
                - '_network_activity_features_per_cluster.xlsx'
                - 'SRT_nEphys_Multiscale_Coordinates.xlsx'

            Parameters
            -------

            Returns
            -------

            File output:
            -------
                - '[gene_list]_gene_expression_network_activity_features.xlsx'
                - '[gene_list]_gene_expression_network_activity_features_[network_activity_feature].xlsx'
        """
        path = self.srcfilepath[:self.srcfilepath.rfind('/')]
        desfilepath = path + '/Correlated_Network_Activity_Features/'
        if not os.path.exists(desfilepath):
            os.mkdir(desfilepath)

        filetype_gene = gene_list_name + '_gene_expression_per_cluster.xlsx'
        filename_gene, Root = self.get_filename_path(self.srcfilepath, filetype_gene)
        for i in range(len(filename_gene)):
            if filename_gene[i][0] != '.':
                print(filename_gene[i])
                gene_root = os.path.join(Root[i], filename_gene[i])

        print("gene_root:", gene_root)

        data_gene = pd.read_excel(gene_root)

        gene_name_list = np.unique(data_gene['gene Name'])

        if os.path.exists(self.srcfilepath + gene_list_name + '_gene_expression_network_activity_features1' + ".xlsx"):
            df = pd.read_excel(self.srcfilepath + gene_list_name + '_gene_expression_network_activity_features' + ".xlsx")
        else:
            ###################################Choose gene way 2
            filehdf5_bxr_name = '.bxr'
            filehdf5_bxr_file, filehdf5_bxr_Root = self.get_filename_path(self.srcfilepath, filehdf5_bxr_name)

            for i in range(len(filehdf5_bxr_file)):
                if filehdf5_bxr_file[i][0] != '.':
                    expFile = filehdf5_bxr_file[i]

            data_nEphys = pd.read_excel(self.srcfilepath + expFile[:-4] + "_network_activity_features_per_cluster.xlsx")
            coordinate_nEphys = [[data_nEphys['x_coordinate'][i], data_nEphys['y_coordinate'][i]] for i in
                               range(len(data_nEphys['x_coordinate']))]
            ###################################Choose gene way 2
            filetype_SRT_nEphys_Coordinates =  'SRT_nEphys_Multiscale_Coordinates.xlsx'
            filename_SRT_nEphys_Coordinates, Root = self.get_filename_path(self.srcfilepath, filetype_SRT_nEphys_Coordinates)
            for i in range(len(filename_SRT_nEphys_Coordinates)):
                if filename_SRT_nEphys_Coordinates[i][0] != '.' and filename_SRT_nEphys_Coordinates[i][0] != '~':
                    SRT_nEphys_Coordinates_root = Root[i] + '/' + filename_SRT_nEphys_Coordinates[i]

            # column_list = ["Synaptic plasticity", "Hippo Signaling Pathway", "Hippocampal Neurogenesis", "Synaptic Vescicles/Adhesion", "Receptors and channels", "IEGs"]
            print(SRT_nEphys_Coordinates_root)
            data_SRT_nEphys_Coordinates = pd.read_excel(SRT_nEphys_Coordinates_root)
            SRT_coordinate = [literal_eval(i) for i in data_SRT_nEphys_Coordinates['Coordinates in SRT']]
            SRT_Barcodes = list(data_SRT_nEphys_Coordinates['Barcodes'])
            Barcode_all = []
            Gene_Name_all = []
            Gene_Expression_Level_all = []
            Cluster_all = []
            LFP_all = []

            delay_all = []
            energy_all = []
            CV2_all = []
            Fano_all = []
            #######
            positive_peaks_all = []
            negative_peaks_all = []
            positive_peak_count_all = []
            negative_peak_count_all = []
            CT_all = []
            amplitude_all = []
            frequency_all = []
            ################
            channel_position_all = []
            nEphys_coordinate = []
            for gene in gene_name_list:
                ####################get the position of gene
                data_new_gene = data_gene.copy()
                data_new_gene = data_new_gene[data_new_gene['gene Name'] == gene]
                s = pd.Series(range(len(data_new_gene)))
                data_new_gene = data_new_gene.set_index(s)
                SRT_position = data_new_gene['Channel Position']
                Barcode = data_new_gene['Barcode']
                # Gene_Name = data_new_gene['gene Name']
                Gene_Expression_Level = data_new_gene['Gene Expression Level']
                Cluster = data_new_gene['Cluster']
                x_position_gene = [float(i[1:i.rfind(',')]) for i in SRT_position]
                y_position_gene = [float(i[i.rfind(',') + 1:-1]) for i in SRT_position]
                coordinate = [[x_position_gene[i], y_position_gene[i]] for i in range(len(y_position_gene))]
                for cor in range(len(coordinate)):
                    if Cluster[cor] != 'Not in Cluster':
                        if Barcode[cor] in SRT_Barcodes:
                            channel_position_all.append(coordinate[cor])
                            index_in_correlation = list(SRT_Barcodes).index(Barcode[cor])
                            related_LFP_cor = literal_eval(
                                data_SRT_nEphys_Coordinates['Coordinates in nEphys'][index_in_correlation])

                            nEphys_coordinate.append(related_LFP_cor)
                            Gene_Expression_Level_all.append(float(Gene_Expression_Level[cor]))
                            lfp_rate_mean = []
                            delay_mean = []
                            energy_mean = []
                            frequency_mean = []
                            amplitude_mean = []
                            positive_peaks_mean = []
                            negative_peaks_mean = []
                            positive_peak_count_mean = []
                            negative_peak_count_mean = []
                            CT_mean = []
                            CV2_mean = []
                            Fano_mean = []

                            for cor_in_nEphys in related_LFP_cor:
                                if cor_in_nEphys in list(coordinate_nEphys):
                                    index_in_nEphys = list(coordinate_nEphys).index(cor_in_nEphys)
                                    lfp_rate_mean.append(float(data_nEphys['LFP Rate(Event/min)'][index_in_nEphys]))
                                    delay_mean.append(float(data_nEphys['Delay(s)'][index_in_nEphys]))
                                    energy_mean.append(float(data_nEphys['Energy'][index_in_nEphys]))
                                    frequency_mean.append(float(data_nEphys['Frequency'][index_in_nEphys]))
                                    amplitude_mean.append(float(data_nEphys['Amplitude(uV)'][index_in_nEphys]))
                                    positive_peaks_mean.append(
                                        float(data_nEphys['Mean Positive Peaks(uV)'][index_in_nEphys]))
                                    negative_peaks_mean.append(
                                        float(data_nEphys['Mean Negative Peaks(uV)'][index_in_nEphys]))
                                    if abs(float(data_nEphys['Mean Positive Peaks(uV)'][index_in_nEphys])) >= abs(
                                            float(data_nEphys['Mean Negative Peaks(uV)'][index_in_nEphys])):
                                        amplitude_mean.append(float(data_nEphys['Mean Positive Peaks(uV)'][index_in_nEphys]))
                                    else:
                                        amplitude_mean.append(float(data_nEphys['Mean Negative Peaks(uV)'][index_in_nEphys]))
                                    positive_peak_count_mean.append(
                                        float(data_nEphys['Positive Peak Count'][index_in_nEphys]))
                                    negative_peak_count_mean.append(
                                        float(data_nEphys['Negative Peak Count'][index_in_nEphys]))
                                    CT_mean.append(float(data_nEphys['CT'][index_in_nEphys]))
                                    CV2_mean.append(float(data_nEphys['CV2'][index_in_nEphys]))
                                    Fano_mean.append(float(data_nEphys['Fano Factor'][index_in_nEphys]))
                                else:
                                    lfp_rate_mean.append(0)
                                    delay_mean.append(0)
                                    energy_mean.append(0)
                                    frequency_mean.append(0)
                                    amplitude_mean.append(0)
                                    positive_peaks_mean.append(0)
                                    negative_peaks_mean.append(0)
                                    positive_peak_count_mean.append(0)
                                    negative_peak_count_mean.append(0)
                                    CT_mean.append(0)
                                    CV2_mean.append(0)
                                    Fano_mean.append(0)

                            Gene_Name_all.append(gene)
                            LFP_all.append(np.mean([i for i in lfp_rate_mean if i == i]))
                            delay_all.append(np.mean([i for i in delay_mean if i == i]))
                            energy_all.append(np.mean([i for i in energy_mean if i == i]))
                            CV2_all.append(np.mean([i for i in CV2_mean if i == i]))
                            Fano_all.append(np.mean([i for i in Fano_mean if i == i]))
                            Cluster_all.append(Cluster[cor])
                            Barcode_all.append(Barcode[cor])
                            positive_peaks_all.append(np.mean([i for i in positive_peaks_mean if i == i]))
                            negative_peaks_all.append(np.mean([i for i in negative_peaks_mean if i == i]))
                            positive_peak_count_all.append(np.mean([i for i in positive_peak_count_mean if i == i]))
                            negative_peak_count_all.append(np.mean([i for i in negative_peak_count_mean if i == i]))
                            CT_all.append(np.mean([i for i in CT_mean if i == i]))
                            amplitude_all.append(np.mean([i for i in amplitude_mean if i == i]))
                            frequency_all.append(np.mean([i for i in frequency_mean if i == i]))

            a = {'Barcode': Barcode_all, 'Channel Position in SRT': channel_position_all,
                 'Coordinates in nEphys': nEphys_coordinate,
                 'Gene Expression Level(Norm)': [i for i in Gene_Expression_Level_all], 'LFP Rate(Event/min)': LFP_all,
                 'Delay(s)': delay_all, 'Energy': energy_all, 'Frequency': frequency_all, 'Amplitude(uV)': amplitude_all,
                 'Mean Positive Peaks(uV)': positive_peaks_all, 'Mean Negative Peaks(uV)': negative_peaks_all,
                 'Positive Peak Count': positive_peak_count_all, 'Negative Peak Count': negative_peak_count_all,
                 'CT': CT_all, 'CV2': CV2_all, 'Fano Factor': Fano_all, "Gene Name": Gene_Name_all,
                 "Cluster": Cluster_all}
            df = pd.DataFrame.from_dict(a, orient='index').T
            df.to_excel(desfilepath + gene_list_name + '_gene_expression_network_activity_features' + ".xlsx",
                        index=False)
        ########################
        writer = pd.ExcelWriter(
            desfilepath + gene_list_name + '_gene_expression_network_activity_feature' + '_' + network_activity_feature + ".xlsx",
            engine='xlsxwriter')
        fig, ax = plt.subplots(nrows=int(len(list(gene_name_list)) / 5) + 1, ncols=5,
                               figsize=(20, 10))  # , facecolor='None'

        if network_activity_feature == 'LFPRate':  ##network_activity_feature = 'LFPRate','Delay','Energy'
            type_name = 'LFP Rate(Event/min)'
        elif network_activity_feature == 'Delay':
            type_name = 'Delay(s)'
        elif network_activity_feature == 'Energy':
            type_name = 'Energy'
        elif network_activity_feature == 'Frequency':
            type_name = 'Frequency'
        elif network_activity_feature == 'Amplitude':
            type_name = 'Amplitude(uV)'
        elif network_activity_feature == 'positive_peaks':
            type_name = 'Mean Positive Peaks(uV)'
        elif network_activity_feature == 'negative_peaks':
            type_name = 'Mean Negative Peaks(uV)'
        elif network_activity_feature == 'positive_peak_count':
            type_name = 'Positive Peak Count'
        elif network_activity_feature == 'negative_peak_count':
            type_name = 'Negative Peak Count'
        elif network_activity_feature == 'CT':
            type_name = 'CT'
        elif network_activity_feature == 'CV2':
            type_name = 'CV2'
        else:
            type_name = 'Fano Factor'

        k = 0
        # print(gene_name_list)
        for gene_name_choose in list(gene_name_list):
            if gene_name_choose != 'PROX1':
                df_new = df.copy()
                df_new = df_new[df_new['Gene Name'] == gene_name_choose]
                s = pd.Series(range(len(df_new)))
                df_new = df_new.set_index(s)
                df_new[type_name] = df_new[type_name].astype(float, errors='raise')
                df_new['Gene Expression Level(Norm)'] = df_new['Gene Expression Level(Norm)'].astype(float,
                                                                                                     errors='raise')
                group_result = df_new.groupby(['Gene Expression Level(Norm)', 'Cluster'])[type_name].mean()
                final = group_result.reset_index()
                final.to_excel(writer, sheet_name=gene_name_choose, index=False)
                # sns.scatterplot(data=final,x='LFP Rate(Event/min)', y='Gene Expression Level(Norm)', hue="Cluster",ax = ax[int(k / 5), int(k % 5)],hue_order = self.clusters,s=8)

                for clu in self.clusters:
                    df_new_clu = final.copy()
                    df_new_clu = df_new_clu[df_new_clu["Cluster"] == clu]
                    s = pd.Series(range(len(df_new_clu)))
                    df_new_clu = df_new_clu.set_index(s)
                    sns.regplot(data=df_new_clu, x=type_name, y='Gene Expression Level(Norm)',
                                ax=ax[int(k / 5), int(k % 5)], label=clu, scatter_kws={"s": 5}, ci=None)
            # plt.setp(ax[int(k / 5), int(k % 5)].get_legend().get_texts(), fontsize='5')  # for legend text
            # plt.setp(ax[int(k / 5), int(k % 5)].get_legend().get_title(), fontsize='8')  # for legend title
            ax[int(k / 5), int(k % 5)].legend(loc='upper left', fontsize='xx-small')
            # ax[int(k / 5), int(k % 5)].spines['top'].set_visible(False)
            # ax[int(k / 5), int(k % 5)].spines['right'].set_visible(False)

            # ax[int(k / 5), int(k % 5)].set_yscale('log')
            ax[int(k / 5), int(k % 5)].set_xscale('log')

            ax[int(k / 5), int(k % 5)].set_title(gene_name_choose)
            # ax[int(k / 5), int(k % 5)].legend(loc='best', fontsize='xx-small')
            # ax[int(k / 5), int(k % 5)].set_aspect('equal', 'box')

            k += 1
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.5, hspace=0.5)
        colorMapTitle = gene_list_name + '_gene_expression_network_activity_feature' + '_' + network_activity_feature
        fig.savefig(desfilepath + colorMapTitle + ".png", format='png', dpi=600)
        plt.close()
        #############
        writer.save()

    def gene_expression_network_activity_features_correlation_pooled_statistics_per_cluster(self, gene_list_name=None, network_activity_feature='LFPRate'):
        """
        Compare correlation statistics per cluster between input conditions i.e. SD and ENR.

             File input needed:
             -------
                 - related files
                 - '[file].bxr'
                 - 'gene_list_all.xlsx'
                 - '[gene_list]_gene_expression_network_activity_features.xlsx'
                 - '[gene_list]_gene_expression_network_activity_features_[network_activity_feature].xlsx'

             Parameters
             -------

             Returns
             -------

             File output:
             -------
                 - '[gene_list]_gene_expression_network_activity_feature_per_cluster_pooled_[network_activity_feature].xlsx'
                 - '[gene_list]_gene_expression_network_activity_feature_per_cluster_pooled_[network_activity_feature].png'
                 - '[gene_list]_gene_expression_network_activity_feature_pooled_statistics_[network_activity_feature].xlsx'
                 - '[gene_list]_gene_expression_network_activity_feature_pooled_[network_activity_feature].png'
                 - '[gene_list]_correlation_coefficient_gene_expression_network_activity_feature_[network_activity_feature].xlsx'
         """
        path = self.srcfilepath[:self.srcfilepath.rfind('/')]
        desfilepath = path + '/Correlated_Network_Activity_Features_Pooled_Statistics/'
        if not os.path.exists(desfilepath):
            os.mkdir(desfilepath)
        if network_activity_feature == 'LFPRate':  ##network_activity_feature = 'LFPRate','Delay','Energy'
            type_name = 'LFP Rate(Event/min)'
        elif network_activity_feature == 'Delay':
            type_name = 'Delay(s)'
        elif network_activity_feature == 'Energy':
            type_name = 'Energy'
        elif network_activity_feature == 'Frequency':
            type_name = 'Frequency'
        elif network_activity_feature == 'Amplitude':
            type_name = 'Amplitude(uV)'
        elif network_activity_feature == 'positive_peaks':
            type_name = 'Mean Positive Peaks(uV)'
        elif network_activity_feature == 'negative_peaks':
            type_name = 'Mean Negative Peaks(uV)'
        elif network_activity_feature == 'positive_peak_count':
            type_name = 'Positive Peak Count'
        elif network_activity_feature == 'negative_peak_count':
            type_name = 'Negative Peak Count'
        elif network_activity_feature == 'CT':
            type_name = 'CT'
        elif network_activity_feature == 'CV2':
            type_name = 'CV2'
        else:
            type_name = 'Fano Factor'
        if os.path.exists(
                self.srcfilepath + gene_list_name + "_gene_expression_network_activity_feature" + '_' + network_activity_feature + ".xlsx"):
            df_LFP_Rate_all = pd.read_excel(
                self.srcfilepath + gene_list_name + "_gene_expression_network_activity_feature" + '_' + network_activity_feature + ".xlsx")
            filetype_gene = 'gene_list_all.xlsx'
            filename_gene, Root_gene = self.get_filename_path(self.srcfilepath, filetype_gene)
            for i in range(len(filename_gene)):
                if filename_gene[i][0] != '.':
                    gene_root = Root_gene[i] + '/' + filename_gene[i]

            select_genes = list(pd.read_excel(gene_root)[gene_list_name])
            select_genes = [i for i in select_genes if type(i) == str]
        else:
            filetype_xlsx = gene_list_name + '_gene_expression_network_activity_feature' + '_' + network_activity_feature + ".xlsx"
            filename_xlsx, Root = self.get_filename_path(self.srcfilepath, filetype_xlsx)

            filetype_gene = 'gene_list_all.xlsx'
            filename_gene, Root_gene = self.get_filename_path(self.srcfilepath, filetype_gene)
            for i in range(len(filename_gene)):
                if filename_gene[i][0] != '.':
                    gene_root = Root_gene[i] + '/' + filename_gene[i]

            select_genes = list(pd.read_excel(gene_root)[gene_list_name])
            select_genes = [i for i in select_genes if type(i) == str]

            Gene_Expression_Level_all = []
            LFP_Rate_all = []
            CLuster_all = []
            Condition_all = []
            file_name_all = []
            gene_name_all = []

            for gene_name in select_genes:
                for i in range(len(filename_xlsx)):
                    if filename_xlsx[i][0] != '.' and filename_xlsx[i][0] != '~':
                        filetype_xlsx_root = Root[i] + '/' + filename_xlsx[i]
                        reader = pd.ExcelFile(filetype_xlsx_root)
                        if gene_name in reader.sheet_names:
                            sheet = pd.read_excel(reader, sheet_name=gene_name)
                            choose_id = [i for i in range(len(sheet[type_name])) if ~np.isnan(sheet[type_name][i])]
                            Gene_Expression_Level_all.extend(
                                [sheet['Gene Expression Level(Norm)'][i] for i in choose_id])
                            LFP_Rate_all.extend(list([sheet[type_name][i] for i in choose_id]))
                            CLuster_all.extend([sheet['Cluster'][i] for i in choose_id])
                            filetype_bxr = '.bxr'
                            print(path)
                            filename_bxr, Root_bxr = self.get_filename_path(path, filetype_bxr)
                            # if len(filename_bxr)>0:
                            print(filename_bxr)
                            file_name_all.extend([filename_bxr[0][:-4]] * len(choose_id))
                            name = 'No condition'
                            for con_name in conditions:
                                if con_name in Root[i]:
                                    name = con_name
                            Condition_all.extend([name] * len(choose_id))
                            gene_name_all.extend([gene_name] * len(choose_id))

            # print(type_name)
            # print(len(gene_name_all),len(Gene_Expression_Level_all),len(LFP_Rate_all),len(CLuster_all),len(file_name_all),len(Condition_all))
            df_LFP_Rate_all = pd.DataFrame(
                {'Gene Name': gene_name_all, 'Gene Expression Level': Gene_Expression_Level_all,
                 type_name: LFP_Rate_all, 'Cluster': CLuster_all,
                 'File Name': file_name_all, 'Condition': Condition_all})
            df_LFP_Rate_all.to_excel(
                desfilepath + gene_list_name + "_gene_expression_network_activity_feature_per_cluster_pooled" + '_' + network_activity_feature + ".xlsx",
                index=False)
        ####################################################
        condition = conditions
        fig, ax = plt.subplots(nrows=len(self.clusters), ncols=len(list(select_genes)),
                               figsize=(30, 15))  # , facecolor='None'
        writer = pd.ExcelWriter(
            desfilepath + gene_list_name + '_gene_expression_network_activity_feature_pooled_statistics' + '_' + network_activity_feature + '.xlsx',
            engine='xlsxwriter')
        fig_per_gene, ax_per_gene = plt.subplots(nrows=int(len(select_genes) / 5) + 1, ncols=5,
                                                 figsize=(20, 20))  # , facecolor='None'
        gene_count = 0
        k = 0
        Correlation_coefficient = []
        gene_name_list_all = []
        condition_all = []
        cluster_all = []
        Correlation_coefficient_cluster = []
        gene_name_list_all_cluster = []
        condition_all_cluster = []

        for gene_name_choose in list(select_genes):
            df_new = df_LFP_Rate_all.copy()
            df_new = df_new[df_new['Gene Name'] == gene_name_choose]
            s = pd.Series(range(len(df_new)))
            df_new = df_new.set_index(s)
            df_new[type_name] = df_new[type_name].astype(float, errors='raise')
            df_new['Gene Expression Level'] = df_new['Gene Expression Level'].astype(float, errors='raise')
            ##########################
            # color = ['blue', 'green']
            l = 0
            for con in condition:
                df_new_con = df_new.copy()
                df_new_con = df_new_con[df_new_con['Condition'] == con]
                s = pd.Series(range(len(df_new_con)))
                df_new_con = df_new_con.set_index(s)
                ################################
                df_new_con[type_name] = df_new_con[type_name].astype(float, errors='raise')
                df_new_con['Gene Expression Level'] = df_new_con['Gene Expression Level'].astype(float, errors='raise')
                group_result = df_new_con.groupby(['Gene Expression Level'])[type_name].mean()
                final = group_result.reset_index()
                final = final.fillna(0)
                A = list(final[type_name])
                B = list(final['Gene Expression Level'])
                # id_nan = [i for i in range(len(A)) if ~np.isnan(A[i])]
                # A = [A[i] for i in id_nan]
                # B = [B[i] for i in id_nan]
                if len(A) > 0:
                    # final = final.drop([i for i in range(len(A)) if np.isnan(A[i])], axis=0)
                    ##############################gene expression value denoising
                    mean_B, low_threshold_B, high_threshold_B = self.quantile_bound(B)
                    # keep_B_id = [i for i in range(len(B)) if B[i] <= high_threshold_B]
                    # B = [B[i] for i in keep_B_id]
                    # A = [A[i] for i in keep_B_id]
                    # final = final.drop([i for i in range(len(B)) if B[i] > high_threshold_B], axis=0)

                    mean_A, low_threshold_A, high_threshold_A = self.quantile_bound(A)
                    # keep_A_id = [i for i in range(len(A)) if A[i] <= high_threshold_A and A[i] >= low_threshold_A]
                    # B = [B[i] for i in keep_A_id]
                    # A = [A[i] for i in keep_A_id]
                    final = final.drop([i for i in range(len(A)) if
                                        A[i] > high_threshold_A and A[i] < low_threshold_A and B[i] > high_threshold_B],
                                       axis=0)
                    ##############sort x
                    conbin = []
                    for i in range(len(A)):
                        conbin.append((A[i], B[i]))

                    def takeOne(elem):
                        return elem[0]

                    conbin.sort(key=takeOne)
                    # ########################
                    if len(conbin) > 0:
                        new_x = np.linspace([i[0] for i in conbin][0], [i[0] for i in conbin][-1])
                        ##############################
                        slope, intercept, r, p, se = stats.linregress([i[0] for i in conbin], [i[1] for i in conbin])
                        Correlation_coefficient.append(r)
                        gene_name_list_all.append(gene_name_choose)
                        condition_all.append(con)

                        ax_per_gene[int(k / 5), int(k % 5)].plot(new_x, intercept + slope * new_x, color=color[l])

                        sns.regplot(data=final, x=type_name, y='Gene Expression Level',
                                    ax=ax_per_gene[int(k / 5), int(k % 5)], label=con, scatter_kws={"s": 6}, ci=None,
                                    fit_reg=False, color=color[l])
                        df_add = pd.DataFrame(
                            {'Regreesion fit Formula': ['Y = ' + str(slope) + '*X+' + str(intercept)]})
                        new_df = pd.concat([final, df_add], axis=1)
                        new_df.to_excel(writer, sheet_name=gene_name_choose + '_' + con, index=False)
                        l += 1
            ax_per_gene[int(k / 5), int(k % 5)].legend(loc='upper left', fontsize='xx-small')

            # ax_per_gene[gene_count].set_xscale('log')
            # ax_per_gene[gene_count].set_yscale('log')
            # ax_per_gene[gene_count].set_aspect('equal', 'box')
            # ax_per_gene[int(k / 5), int(k % 5)].set_box_aspect(1)
            ax_per_gene[int(k / 5), int(k % 5)].set_title(gene_name_choose)
            ax_per_gene[int(k / 5), int(k % 5)].set_xlabel('')
            ax_per_gene[int(k / 5), int(k % 5)].set_ylabel('')
            # ax_per_gene[int(k / 5), int(k % 5)].spines['top'].set_visible(False)
            # ax_per_gene[int(k / 5), int(k % 5)].spines['right'].set_visible(False)
            # ax_per_gene[int(k / 5), int(k % 5)].spines['bottom'].set_visible(False)
            # ax_per_gene[int(k / 5), int(k % 5)].spines['left'].set_visible(False)
            # plt.setp(ax_per_gene[int(k / 5), int(k % 5)].get_xticklabels(), visible=False)
            # plt.setp(ax_per_gene[int(k / 5), int(k % 5)].get_yticklabels(), visible=False)
            # ax_per_gene[int(k / 5), int(k % 5)].set_xticks([])
            # ax_per_gene[int(k / 5), int(k % 5)].set_yticks([])
            k += 1

            cluster_count = 0

            for clu in self.clusters:
                df_new_clu = df_new.copy()
                df_new_clu = df_new_clu[df_new_clu["Cluster"] == clu]
                s = pd.Series(range(len(df_new_clu)))
                df_new_clu = df_new_clu.set_index(s)
                ###########################################
                l = 0
                for con in condition:
                    df_new_clu_con = df_new_clu.copy()
                    df_new_clu_con = df_new_clu_con[df_new_clu_con['Condition'] == con]
                    s = pd.Series(range(len(df_new_clu_con)))
                    df_new_clu_con = df_new_clu_con.set_index(s)
                    ################################
                    df_new_clu_con[type_name] = df_new_clu_con[type_name].astype(float, errors='raise')
                    df_new_clu_con['Gene Expression Level'] = df_new_clu_con['Gene Expression Level'].astype(float,
                                                                                                             errors='raise')
                    group_result = df_new_clu_con.groupby(['Gene Expression Level'])[type_name].mean()
                    final_clu = group_result.reset_index()
                    final_clu = final_clu.fillna(0)
                    A_clu = list(final_clu[type_name])
                    B_clu = list(final_clu['Gene Expression Level'])
                    # id_nan = [i for i in range(len(A_clu)) if ~np.isnan(A_clu[i])]
                    # A_clu = [A_clu[i] for i in id_nan]
                    # B_clu = [B_clu[i] for i in id_nan]
                    if len(A_clu) > 0:
                        # final_clu = final_clu.drop([i for i in range(len(A_clu)) if np.isnan(A_clu[i])], axis=0)
                        ##############################gene expression value denoising
                        # mean_B, low_threshold_B, high_threshold_B = self.quantile_bound(B_clu)
                        # keep_B_id = [i for i in range(len(B_clu)) if B_clu[i] <= high_threshold_B]
                        # B_clu = [B_clu[i] for i in keep_B_id]
                        # A_clu = [A_clu[i] for i in keep_B_id]
                        # final_clu = final_clu.drop([i for i in range(len(B_clu)) if B_clu[i] > high_threshold_B], axis=0)
                        #
                        # mean_A, low_threshold_A, high_threshold_A = self.quantile_bound(A_clu)
                        # keep_A_id = [i for i in range(len(A_clu)) if A_clu[i] <= high_threshold_A and A_clu[i] >= low_threshold_A]
                        # B_clu = [B_clu[i] for i in keep_A_id]
                        # A_clu = [A_clu[i] for i in keep_A_id]
                        # final_clu = final_clu.drop([i for i in range(len(A_clu)) if A_clu[i] > high_threshold_A and A_clu[i] < low_threshold_A], axis=0)
                        ##############sort x
                        conbin_clu = []
                        for i in range(len(A_clu)):
                            conbin_clu.append((A_clu[i], B_clu[i]))

                        def takeOne(elem):
                            return elem[0]

                        conbin_clu.sort(key=takeOne)
                        # print(conbin_clu)
                        # ########################
                        new_x = np.linspace([i[0] for i in conbin_clu][0], [i[0] for i in conbin_clu][-1])
                        ##############################
                        slope, intercept, r_clu, p, se = stats.linregress([i[0] for i in conbin_clu],
                                                                          [i[1] for i in conbin_clu])
                        cluster_all.append(clu)
                        Correlation_coefficient_cluster.append(r_clu)
                        gene_name_list_all_cluster.append(gene_name_choose)
                        condition_all_cluster.append(con)
                        ax[cluster_count, gene_count].plot(new_x, intercept + slope * new_x, color=color[l])
                        ############################
                        sns.regplot(data=final_clu, x=type_name, y='Gene Expression Level',
                                    ax=ax[cluster_count, gene_count], fit_reg=False, label=con, scatter_kws={"s": 3},
                                    ci=None, color=color[l])
                        l += 1
                # plt.setp(ax[int(k / 5), int(k % 5)].get_legend().get_texts(), fontsize='5')  # for legend text
                # plt.setp(ax[int(k / 5), int(k % 5)].get_legend().get_title(), fontsize='8')  # for legend title
                ax[cluster_count, gene_count].legend(loc='upper left', fontsize='xx-small')
                ax[cluster_count, gene_count].set_xscale('log')
                ax[cluster_count, gene_count].set_title(gene_name_choose + '_' + clu)

                cluster_count += 1

            gene_count += 1
        fig.tight_layout()
        fig.subplots_adjust(wspace=1, hspace=2)
        colorMapTitle = gene_list_name + '_gene_expression_network_activity_feature_per_cluster_pooled' + '_' + network_activity_feature
        fig.savefig(desfilepath + colorMapTitle + ".png", format='png', dpi=600)
        # plt.close()

        fig_per_gene.tight_layout()
        # fig_per_gene.subplots_adjust(wspace=1)
        colorMapTitle = gene_list_name + '_gene_expression_network_activity_feature_pooled' + '_' + network_activity_feature
        fig_per_gene.savefig(desfilepath + colorMapTitle + ".png", format='png', dpi=600)
        plt.close()
        writer.save()
        ###############################

        a = {'Gene Name for global': gene_name_list_all, 'Correlation coefficient global': Correlation_coefficient,
             "Condition for global": condition_all, "Cluster": cluster_all,
             "Gene Name for cluster": gene_name_list_all_cluster,
             "Correlation coefficient for cluster": Correlation_coefficient_cluster,
             "Condition for cluster": condition_all_cluster}
        df = pd.DataFrame.from_dict(a, orient='index').T
        df.to_excel(
            desfilepath + gene_list_name + '_correlation_coefficient_gene_expression_network_activity_feature' + '_' + network_activity_feature + ".xlsx",
            index=False)

    def gene_expression_network_activity_features_correlation_pooled_statistics_per_region(self,gene_list_name=None,choose_gene = False,bin_num = 100):
        """
         Compare correlation statistics per region between input conditions i.e. SD and ENR.

             File input needed:
             -------
                 - related files
                 - '[file].bxr'
                 - 'gene_list_all.xlsx'
                 - '[gene_list]_gene_expression_network_activity_feature_per_cluster_pooled_[network_activity_feature].xlsx'

             Parameters
             -------

             Returns
             -------

             File output:
             -------
                 - '[gene_list]_gene_expression_network_activity_feature_per_cluster_pooled_[network_activity_feature].xlsx'
                 - '[gene_list]_gene_expression_network_activity_feature_per_cluster_pooled_[network_activity_feature].png'
         """
        path = self.srcfilepath[:self.srcfilepath.rfind('/')]
        desfilepath = path + '/Correlated_Network_Activity_Features_Pooled_Statistics/'
        if not os.path.exists(desfilepath):
            os.mkdir(desfilepath)
        fig, ax = plt.subplots(nrows=2, ncols=len(network_activity_feature),figsize=(40, 10))  # , facecolor='None'
        writer = pd.ExcelWriter(desfilepath + gene_list_name + '_gene_expression_network_activity_feature_per_region_pooled'+'.xlsx',engine='xlsxwriter')
        type_count = 0
        for type in list(network_activity_feature):
            print(type)
            if type == 'LFPRate':  ##network_activity_feature = 'LFPRate','Delay','Energy'
                type_name = 'LFP Rate(Event/min)'
            elif type == 'Delay':
                type_name = 'Delay(s)'
            elif type == 'Energy':
                type_name = 'Energy'
            elif type == 'Frequency':
                type_name = 'Frequency'
            elif type == 'Amplitude':
                type_name = 'Amplitude(uV)'
            elif type == 'positive_peaks':
                type_name = 'Mean Positive Peaks(uV)'
            elif type == 'negative_peaks':
                type_name = 'Mean Negative Peaks(uV)'
            elif type == 'positive_peak_count':
                type_name = 'Positive Peak Count'
            elif type == 'negative_peak_count':
                type_name = 'Negative Peak Count'
            elif type == 'CT':
                type_name = 'CT'
            elif type == 'CV2':
                type_name = 'CV2'
            else:
                type_name = 'Fano Factor'
            print('A',type_name)
            df_LFP_Rate_all = pd.read_excel(desfilepath + gene_list_name + "_gene_expression_network_activity_feature_per_cluster_pooled" + '_' + type + ".xlsx")
            con_count = 0
            for con in conditions:
                df_new_con = df_LFP_Rate_all.copy()
                df_new_con = df_new_con[df_new_con['Condition'] == con]
                s = pd.Series(range(len(df_new_con)))
                df_new_con = df_new_con.set_index(s)
                if choose_gene == False:
                    df_new_con = df_new_con
                else:
                    df_new_con_gene = df_new_con.copy()
                    df_new_con_gene = df_new_con_gene[df_new_con_gene['Gene Name'] == choose_gene]
                    s = pd.Series(range(len(df_new_con_gene)))
                    df_new_con = df_new_con_gene.set_index(s)

                Region_add = ['Cortex' if clu in Cortex else 'Hippo' for clu in list(df_new_con["Cluster"])]
                df_add = pd.DataFrame({'Region': Region_add})
                final = pd.concat([df_new_con, df_add], axis=1)
                # final.fillna(0)

                ##############################gene expression value denoising
                print(final.columns)
                A = list(final[type_name])

                final_filter = final
                if len(A) > 0:
                    final_filter[type_name] = final_filter[type_name].astype(float, errors='raise')
                    final_filter['Gene Expression Level'] = final_filter['Gene Expression Level'].astype(float,errors='raise')
                    final = final_filter

                    #######################
                    Region_count = 0
                    Correlation_coefficient = []
                    Region_list = []
                    p_value_Region = []
                    SEM_Region = []
                    Formula = []

                    for region in Region:
                        df_new_Region = final.copy()
                        df_new_Region = df_new_Region[df_new_Region['Region'] == region]
                        s = pd.Series(range(len(df_new_Region)))
                        df_new_Region = df_new_Region.set_index(s)
                        ##############################gene expression value denoising
                        A = list(df_new_Region[type_name])
                        B = list(df_new_Region['Gene Expression Level'])
                        mean_B, low_threshold_B, high_threshold_B = self.quantile_bound(B)
                        mean_A, low_threshold_A, high_threshold_A = self.quantile_bound(A)
                        df_new_Region = df_new_Region.drop([i for i in range(len(B)) if B[i] >= high_threshold_B or B[i] <= low_threshold_B or A[i] >= high_threshold_A or A[i] <= low_threshold_A], axis=0)

                        #################################
                        df_new_Region = df_new_Region.groupby(by=['Region', 'Gene Expression Level'], dropna=True)[type_name].mean().reset_index()
                        df_new_Region = df_new_Region.groupby(by=['Region', type_name], dropna=True)['Gene Expression Level'].mean().reset_index()

                        # print(df_new_Region)
                        bins = list(np.linspace(min(df_new_Region['Gene Expression Level']), max(df_new_Region['Gene Expression Level']), bin_num))
                        labels = [(i+j)/2 for i, j in zip(bins[:-1], bins[1:])]
                        bin_Gene = pd.cut(df_new_Region['Gene Expression Level'],bins=bins, labels=labels, include_lowest=True)
                        df_new_Region = df_new_Region.groupby(by=['Region',bin_Gene], dropna=True)[type_name].mean().unstack(fill_value=0).stack().reset_index(name=type_name)

                        ##############sort x
                        A = list(df_new_Region[type_name])
                        B = list(df_new_Region['Gene Expression Level'])
                        conbin = []
                        for i in range(len(A)):
                            conbin.append((A[i], B[i]))

                        def takeOne(elem):
                            return elem[0]

                        conbin.sort(key=takeOne)
                        # ########################
                        if len(conbin) > 0:
                            varx = np.asarray([i[0] for i in conbin])
                            varx = np.asarray([i/max(varx) for i in varx])
                            vary = np.asarray([i[1] for i in conbin])
                            vary = np.asarray([i/max(vary) for i in vary])
                            mask = ~np.isnan(varx) & ~np.isnan(vary)& ~np.isinf(varx) & ~np.isinf(vary)
                            slope, intercept, r, p, se = stats.linregress(varx[mask], vary[mask])
                            # slope, intercept, r, p, se = stats.linregress([np.log10(i[0]) for i in conbin], [np.log10(i[1]) for i in conbin])
                            print('Y = ' + str(slope) + '*X+' + str(intercept))
                            Formula.append('Y = ' + str(slope) + '*X+' + str(intercept))
                            Correlation_coefficient.append(r)
                            Region_list.append(region)
                            p_value_Region.append(p)
                            SEM_Region.append(se) #Standard error of the estimated gradient.

                            ###########################################
                            # x = np.asarray([i[0] for i in conbin])
                            # y = np.asarray([i[1] for i in conbin])
                            # from scipy.optimize import curve_fit
                            # def myExpFunc(x, a, b):
                            #     return a * np.power(x, b)
                            #
                            # newX = np.logspace(np.log10(min(x)),np.log10(max(x)), base=20)  # Makes a nice domain for the fitted curves.
                            # # Goes from 10^0 to 10^3
                            # # This avoids the sorting and the swarm of lines.
                            #
                            # popt, pcov = curve_fit(myExpFunc, x, y)
                            # ax[Region_count, type_count].plot(newX, myExpFunc(newX, *popt), color=color[con_count],label='Y='+"{0:.2f}*x**{1:.2f}".format(*popt),linewidth=2)
                            #
                            # sns.regplot(data=df_new_Region, x=type_name, y='Gene Expression Level',
                            #             ax=ax[Region_count, type_count], fit_reg=False, label=con,
                            #             scatter_kws={"s": 6, 'marker':'o', 'edgecolors':None}, ci=None, color=color[con_count])
                            #######################################
                            # p = sns.regplot(data=df_new_Region, x=type_name, y='Gene Expression Level',
                            #                 ax=ax[Region_count, type_count], label=con + ':' + str(round(r, 2)),
                            #                 scatter_kws={"s": 6, 'marker': 'o', 'edgecolors': None}, ci=60,
                            #                 fit_reg=True, color=color[con_count], order=1, truncate=True, robust=True)
                            y = np.array(df_new_Region[type_name])
                            x = np.array(df_new_Region['Gene Expression Level'])
                            # Modeling with Numpy
                            def equation(a, b):
                                """Return a 1D polynomial."""
                                return np.polyval(a, b)

                            p, cov = np.polyfit(x, y, 1,cov=True)  # parameters and covariance from of the fit of 1-D polynom.
                            y_model = equation(p,x)  # model using the fit parameters; NOTE: parameters here are coefficients

                            # Statistics
                            n = len(y)  # number of observations
                            m = p.size  # number of parameters
                            dof = n - m  # degrees of freedom
                            t = stats.t.ppf(Prediction_Limits, n - m)  # used for CI and PI bands

                            # Estimates of Error in Data/Model
                            resid = y - y_model
                            chi2 = np.sum((resid / y_model) ** 2)  # chi-squared; estimates error in data
                            chi2_red = chi2 / dof  # reduced chi-squared; measures goodness of fit
                            s_err = np.sqrt(np.sum(resid ** 2) / dof)  # standard deviation of the error

                            # Plotting --------------------------------------------------------------------
                            # Data
                            ax[Region_count, type_count].plot(
                                x, y, "o", color=color[con_count], markersize=8,
                                markeredgewidth=1, markeredgecolor=color[con_count], markerfacecolor="None",label=con+':'+str(round(r,2))
                            )

                            # Fit
                            ax[Region_count, type_count].plot(x, y_model, "-", color=color[con_count], linewidth=1.5, alpha=0.5) #, label="Fit"

                            x2 = np.linspace(np.min(x), np.max(x), 100)
                            y2 = equation(p, x2)

                            # Confidence Interval (select one)
                            # plot_ci_manual(t, s_err, n, x, x2, y2, ax=ax)
                            ci = t * s_err * np.sqrt(1 / n + (x2 - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))
                            ax[Region_count, type_count].fill_between(x2, y2 + ci, y2 - ci, color=color[con_count],alpha=0.5)
                            # plot_ci_bootstrap(x, y, resid, ax=ax)

                            # Prediction Interval
                            pi = t * s_err * np.sqrt(1 + 1 / n + (x2 - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))
                            ax[Region_count, type_count].fill_between(x2, y2 + pi, y2 - pi, color="None", linestyle="--")
                            ax[Region_count, type_count].plot(x2, y2 - pi, "--", color=color[con_count], label=str(Prediction_Limits*100)+"% Prediction Limits")
                            ax[Region_count, type_count].plot(x2, y2 + pi, "--", color=color[con_count])

                            # Figure Modifications --------------------------------------------------------
                            # Borders
                            # ax[Region_count, type_count].spines["top"].set_color("0.5")
                            # ax[Region_count, type_count].spines["bottom"].set_color("0.5")
                            # ax[Region_count, type_count].spines["left"].set_color("0.5")
                            # ax[Region_count, type_count].spines["right"].set_color("0.5")
                            ax[Region_count, type_count].get_xaxis().set_tick_params(direction="out")
                            ax[Region_count, type_count].get_yaxis().set_tick_params(direction="out")
                            ax[Region_count, type_count].xaxis.tick_bottom()
                            ax[Region_count, type_count].yaxis.tick_left()

                            # Labels
                            # plt.title("Fit Plot for Weight", fontsize="14", fontweight="bold")

                            # Custom legend
                            handles, labels = ax[Region_count, type_count].get_legend_handles_labels()
                            display = (0, 1)
                            anyArtist = plt.Line2D((0, 1), (0, 0), color="#b9cfe7")  # create custom artists
                            legend = plt.legend(
                                [handle for i, handle in enumerate(handles) if i in display] + [anyArtist],
                                [label for i, label in enumerate(labels) if i in display] + [str(Prediction_Limits*100)+"% Prediction Limits"],
                                loc=9, bbox_to_anchor=(0, -0.21, 1., 0.102), ncol=3, mode="expand"
                            )
                            frame = legend.get_frame().set_edgecolor("0.5")

                            #############################################


                            # p = sns.regplot(data=df_new_Region, x=type_name, y='Gene Expression Level',
                            #             ax=ax[Region_count, type_count], label=con+':'+str(round(r,2)), scatter_kws={"s": 6, 'marker':'o', 'edgecolors':None}, ci=60,
                            #             fit_reg=True, color=color[con_count], order=1,truncate = True,robust=True)

                            ax[Region_count, type_count].legend(loc='best', fontsize='xx-small')
                            ax[Region_count, type_count].spines['top'].set_visible(False)
                            ax[Region_count, type_count].spines['right'].set_visible(False)
                            # ax[Region_count, type_count].set_yscale('log')
                            # ax[Region_count, type_count].set_xscale('log')
                            # ax[Region_count, type_count].set_xlim(xmin=min(x),xmax=max(x))
                            # ax[Region_count, type_count].set_ylim(ymin=0)
                            # ax[Region_count, type_count].spines['bottom'].set_visible(False)
                            # ax[Region_count, type_count].spines['left'].set_visible(False)
                            #                             x = np.array(df_new_Region[type_name])
                            # y = np.array(df_new_Region['Gene Expression Level'])
                            ax[Region_count, type_count].set_xlabel('Gene Expression Level')
                            if type_count == 0:
                                ax[Region_count, type_count].set_ylabel(Region)
                            else:
                                ax[Region_count, type_count].set_ylabel(type_name)

                            Region_count += 1

                    final_add = pd.DataFrame({'Formula':Formula,'Correlation coefficient': Correlation_coefficient,'p_value':p_value_Region,'SEM':SEM_Region,'Region':Region_list})
                    final_all = pd.concat([final, final_add], axis=1)
                    final_all.to_excel(writer, sheet_name=con+'_'+type, index=False)
                    ###########################################

                con_count += 1

            type_count+=1

        fig.tight_layout()
        fig.subplots_adjust()
        colorMapTitle = gene_list_name + '_gene_expression_network_activity_feature_per_region_pooled'
        fig.savefig(desfilepath + colorMapTitle + ".png", format='png', dpi=600)

        writer.save()

    def all_gene_expression(self):
        """
         Provide gene expression values per SRT spatial spot with filter.
         Filter removes genes based on gene expression level. If only a few nodes had a highly expressed gene and the rest low or no expression it would be removed. Without filter allows all gene expression in for analysis.

             File input needed:
             -------
                 - related files

             Parameters
             -------

             Returns
             -------

             File output:
             -------
                 - 'all_gene_expression_per_coordinate.xlsx'
         """
        # Read related information
        csv_file, tissue_lowres_scalef, features_name, matr_raw, barcodes, img, csv_file_cluster = self.read_related_files()
        barcode_cluster = np.asarray(csv_file_cluster["Barcode"])
        scatter_x = np.asarray(csv_file["pixel_x"])
        scatter_y = np.asarray(csv_file["pixel_y"])
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
        genes_per_cell = np.asarray((matr > 0).sum(axis=0)).squeeze()
        ###############delete the nodes with less then 1000 gene count
        deleted_notes = [i for i in range(len(genes_per_cell)) if genes_per_cell[i] <= 1000]
        ###############delete the nodes not in clusters
        deleted_notes_cluster = [i for i in range(len(genes_per_cell)) if str(barcodes[i])[2:-1] not in barcode_cluster]
        deleted_notes.extend(deleted_notes_cluster)
        deleted_notes = list(np.unique(deleted_notes))
        # ##########################################################
        matr = np.delete(matr, deleted_notes, axis=1)

        barcodes_filter = [barcodes[i] for i in range(len(barcodes)) if i not in deleted_notes]
        new_id = [j for i in barcode_CSV for j in range(len(barcodes)) if str(barcodes[j])[2:-1] == i]

        x_filter = [scatter_x[ix][i] for i in range(len(scatter_x[ix])) if new_id[i] not in deleted_notes]
        y_filter = [scatter_y[ix][i] for i in range(len(scatter_y[ix])) if new_id[i] not in deleted_notes]

        adata = AnnData(np.array(matr))
        sc.pp.normalize_total(adata, inplace=True)
        gene_expression = adata.X
        genes_expression_count = np.asarray((matr > 0).sum(axis=1)).squeeze()

        ############################ choose top gene
        from operator import itemgetter
        indices, genes_expression_count_sorted = zip(*sorted(enumerate(genes_expression_count), key=itemgetter(1), reverse=True))
        ###################################Choose gene 1 way

        top_gene_indices = indices
        df_gene_count = pd.DataFrame({'Gene Name': [gene_name[i] for i in indices], 'Gene Count sorted': list(genes_expression_count_sorted)})

        # #####################################
        gene_expression_filter = [np.asarray(gene_expression)[i] for i in top_gene_indices]
        gene_expression_series = np.asarray(gene_expression_filter)
        # gene_expression_series = np.asarray(gene_expression_filter).T
        # print(gene_expression_series.shape)

        ###################################################mean expression nodes count
        gene_expression_list = list(gene_expression_series[gene_expression_series > 0])
        mean_B, low_threshold_B, high_threshold_B = self.quantile_bound(gene_expression_list)
        gene_expression_series[gene_expression_series >= high_threshold_B] = 0
        mean_expression_nodes = np.mean([int(np.asarray((np.asarray(gene_expression_series[i]) != 0).sum())) for i in range(len(gene_expression_series))])
        std_expression_nodes = np.std([int(np.asarray((np.asarray(gene_expression_series[i]) != 0).sum())) for i in range(len(gene_expression_series))])
        # ##############################################################
        id_no_expression = [i for i in range(len(gene_expression_series)) if sum(gene_expression_series[i]) == 0 or int(np.asarray((np.asarray(gene_expression_series[i]) != 0).sum())) <= mean_expression_nodes + std_expression_nodes]
        # id_no_expression = [i for i in range(len(gene_expression_series)) if sum(gene_expression_series[i]) == 0]
        gene_expression_series = np.delete(gene_expression_series, id_no_expression, axis=0)
        gene_expression_series = np.asarray(gene_expression_series).T
        # print(gene_expression_series.shape)
        # #filter gene Tripathy SJ, Toker L, Li B, Crichlow C-L, Tebaykin D, Mancarci BO, et al. (2017) Transcriptomic correlates of neuron electrophysiological diversity. PLoS Comput Biol 13(10): e1005814. https://doi.org/10.1371/journal. pcbi.1005814
        expression_each_gene_mean = np.asarray(gene_expression_series).mean(axis=0)
        expression_each_gene_std = np.asarray(gene_expression_series).std(axis=0)

        index_mean = np.where(np.array(expression_each_gene_mean) >= np.asarray(gene_expression_series).mean())[0]
        index_std = np.where(np.array(expression_each_gene_std) >= np.asarray(gene_expression_series).std())[0]
        keep_gene_id = set(index_mean) & set(index_std)
        filter_gene_expression_matrix = gene_expression_series[:, list(keep_gene_id)]
        remain_gene_name = np.asarray(df_gene_count['Gene Name'])[list(keep_gene_id)]

        coordinates = [[x_filter[i], y_filter[i]] for i in range(len(x_filter)) if i not in id_no_expression]
        barcodes_filter_1 = [str(barcodes_filter[i])[2:-1] for i in range(len(barcodes_filter)) if i not in id_no_expression]
        gene_expression_matrix = pd.DataFrame(filter_gene_expression_matrix)
        # print(gene_expression_matrix.shape)
        gene_expression_matrix.columns = list(remain_gene_name)
        gene_expression_matrix.insert(0, 'Coordinates(Pixel)', coordinates)
        gene_expression_matrix.insert(0, 'Barcodes', barcodes_filter_1)
        gene_expression_matrix.to_excel(self.srcfilepath + "all_gene_expression_per_coordinate.xlsx", index=False)

    def all_gene_expression_without_filter(self):
        """
         Provide gene expression values per SRT spatial spot without filter.
         Filter removes genes based on gene expression level. If only a few nodes had a highly expressed gene and the rest low or no expression it would be removed. Without filter allows all gene expression in for analysis.

             File input needed:
             -------
                 - related files

             Parameters
             -------

             Returns
             -------

             File output:
             -------
                 - 'all_gene_expression_per_coordinate_without_filter.npy'
                 - 'all_gene_expression_per_coordinate_without_filter_columns.npy'
         """
        # Read related information
        final_filter = pd.read_excel(
            self.srcfilepath + 'all_gene_expression_per_coordinate' + ".xlsx")
        gene_exist = list(final_filter.columns)[2:]
        csv_file, tissue_lowres_scalef, features_name, matr_raw, barcodes, img, csv_file_cluster = self.read_related_files()
        barcode_cluster = np.asarray(csv_file_cluster["Barcode"])
        scatter_x = np.asarray(csv_file["pixel_x"])
        scatter_y = np.asarray(csv_file["pixel_y"])
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
        genes_per_cell = np.asarray((matr > 0).sum(axis=0)).squeeze()
        ###############delete the nodes with less then 1000 gene count
        deleted_notes = [i for i in range(len(genes_per_cell)) if genes_per_cell[i] <= 1000]
        ###############delete the nodes not in clusters
        deleted_notes_cluster = [i for i in range(len(genes_per_cell)) if str(barcodes[i])[2:-1] not in barcode_cluster]
        deleted_notes.extend(deleted_notes_cluster)
        deleted_notes = list(np.unique(deleted_notes))
        # ##########################################################
        matr = np.delete(matr, deleted_notes, axis=1)

        barcodes_filter = [barcodes[i] for i in range(len(barcodes)) if i not in deleted_notes]
        new_id = [j for i in barcode_CSV for j in range(len(barcodes)) if str(barcodes[j])[2:-1] == i]

        x_filter = [scatter_x[ix][i] for i in range(len(scatter_x[ix])) if new_id[i] not in deleted_notes]
        y_filter = [scatter_y[ix][i] for i in range(len(scatter_y[ix])) if new_id[i] not in deleted_notes]

        adata = AnnData(np.array(matr))
        sc.pp.normalize_total(adata, inplace=True)
        gene_expression = adata.X
        genes_expression_count = np.asarray((matr > 0).sum(axis=1)).squeeze()

        ############################ choose top gene
        # from operator import itemgetter
        # indices, genes_expression_count_sorted = zip(
        #     *sorted(enumerate(genes_expression_count), key=itemgetter(1), reverse=True))
        ###################################Choose gene 1 way

        # top_gene_indices = indices
        df_gene_count = pd.DataFrame({'Gene Name': gene_name, 'Gene Count sorted': list(genes_expression_count)})

        # #####################################
        # gene_expression_filter = np.asarray(gene_expression)
        gene_expression_series = np.asarray(gene_expression)
        # gene_expression_series = np.asarray(gene_expression_filter).T
        # print(gene_expression_series.shape)

        ###################################################mean expression nodes count
        # gene_expression_list = list(gene_expression_series[gene_expression_series > 0])
        # mean_B, low_threshold_B, high_threshold_B = self.quantile_bound(gene_expression_list, quantile=0.95)
        # gene_expression_series[gene_expression_series >= high_threshold_B] = 0
        # mean_expression_nodes = np.mean([int(np.asarray((np.asarray(gene_expression_series[i]) != 0).sum())) for i in range(len(gene_expression_series))])
        # std_expression_nodes = np.std([int(np.asarray((np.asarray(gene_expression_series[i]) != 0).sum())) for i in range(len(gene_expression_series))])
        # ##############################################################
        # id_no_expression = [i for i in range(len(gene_expression_series)) if sum(gene_expression_series[i]) == 0 or int(np.asarray((np.asarray(gene_expression_series[i]) != 0).sum())) <= mean_expression_nodes + std_expression_nodes]
        id_no_expression = [i for i in range(len(gene_expression_series)) if sum(gene_expression_series[i]) == 0]
        gene_expression_series = np.delete(gene_expression_series, id_no_expression, axis=0)

        gene_expression_series = np.asarray(gene_expression_series).T
        # print(gene_expression_series.shape)
        # #filter gene Tripathy SJ, Toker L, Li B, Crichlow C-L, Tebaykin D, Mancarci BO, et al. (2017) Transcriptomic correlates of neuron electrophysiological diversity. PLoS Comput Biol 13(10): e1005814. https://doi.org/10.1371/journal. pcbi.1005814
        # expression_each_gene_mean = np.asarray(gene_expression_series).mean(axis=0)
        # expression_each_gene_std = np.asarray(gene_expression_series).std(axis=0)

        # index_mean = np.where(np.array(expression_each_gene_mean) >= np.asarray(gene_expression_series).mean())[0]
        # index_std = np.where(np.array(expression_each_gene_std) >= np.asarray(gene_expression_series).std())[0]
        # keep_gene_id = set(index_mean) & set(index_std)
        filter_gene_expression_matrix = gene_expression_series
        remain_gene_name = np.asarray(df_gene_count['Gene Name'])

        coordinates = [[x_filter[i], y_filter[i]] for i in range(len(x_filter))]
        barcodes_filter_1 = [str(barcodes_filter[i])[2:-1] for i in range(len(barcodes_filter))]
        gene_expression_matrix = pd.DataFrame(filter_gene_expression_matrix)

        gene_expression_matrix.columns = list(
            [remain_gene_name[i] for i in range(len(remain_gene_name)) if i not in id_no_expression])
        gene_expression_matrix.insert(0, 'Coordinates(Pixel)', coordinates)
        gene_expression_matrix.insert(0, 'Barcodes', barcodes_filter_1)
        gene_deleted = list(set(gene_exist) & set(gene_expression_matrix.columns))
        gene_expression_matrix = gene_expression_matrix.drop(gene_exist, axis=1)
        np.save(self.srcfilepath + "all_gene_expression_per_coordinate_without_filter", gene_expression_matrix)
        np.save(self.srcfilepath + "all_gene_expression_per_coordinate_without_filter_columns",
                list(gene_expression_matrix.columns))
        # gene_expression_matrix.to_excel(self.srcfilepath + "all_gene_expression_per_coordinate_without_filter.xlsx", index=False)

    def all_gene_expression_network_activity_features_correlation(self):
        """
         Correlate specified network activity feature from nEphys data with all gene expression values from SRT data with filter.
         Filter removes genes based on gene expression level. If only a few nodes had a highly expressed gene and the rest low or no expression it would be removed. Without filter allows all gene expression in for analysis.

             File input needed:
             -------
                 - '[file].bxr'
                 - 'all_gene_expression_per_coordinate.xlsx'
                 - '_network_activity_features_per_cluster.xlsx'
                 - 'SRT_nEphys_Multiscale_Coordinates.xlsx'

             Parameters
             -------

             Returns
             -------

             File output:
             -------
                 - '_all_gene_expression_network_activity_features_p_values.xlsx'
         """

        gene_expression_matrix = pd.read_excel(self.srcfilepath + "all_gene_expression_per_coordinate.xlsx")
        gene_all_names = list(gene_expression_matrix.columns)[2:]
        ###################################Choose gene way 2
        filehdf5_bxr_name = '.bxr'
        filehdf5_bxr_file, filehdf5_bxr_Root = self.get_filename_path(self.srcfilepath, filehdf5_bxr_name)

        for i in range(len(filehdf5_bxr_file)):
            if filehdf5_bxr_file[i][0] != '.':
                expFile = filehdf5_bxr_file[i]

        data_nEphys = pd.read_excel(self.srcfilepath + expFile[:-4] + "_network_activity_features_per_cluster.xlsx")
        coordinate_nEphys = [[data_nEphys['x_coordinate'][i], data_nEphys['y_coordinate'][i]] for i in range(len(data_nEphys['x_coordinate']))]
        ###################################Choose gene way 2
        filetype_SRT_nEphys_Coordinates = 'SRT_nEphys_Multiscale_Coordinates.xlsx'
        filename_SRT_nEphys_Coordinates, Root = self.get_filename_path(self.srcfilepath, filetype_SRT_nEphys_Coordinates)
        for i in range(len(filename_SRT_nEphys_Coordinates)):
            if filename_SRT_nEphys_Coordinates[i][0] != '.':
                SRT_nEphys_Coordinates_root = Root[i] + '/' + filename_SRT_nEphys_Coordinates[i]

        data_SRT_nEphys_Coordinates_raw = pd.read_excel(SRT_nEphys_Coordinates_root)

        Gene_Name_all = []
        Parameters_all = []
        Cluster_all = []
        Correlation_all = []
        p_values_all = []
        Condition_all = []
        File_name_all = []

        cluster_all = self.clusters
        cluster_all.append('All')
        for gene in gene_all_names:
            for clu in cluster_all:
                ####################get the position of gene
                data_SRT_clu = data_SRT_nEphys_Coordinates_raw.copy()
                if clu != 'All':
                    data_SRT_clu = data_SRT_clu[data_SRT_clu['Cluster'] == clu]
                    s = pd.Series(range(len(data_SRT_clu)))
                    data_SRT_clu = data_SRT_clu.set_index(s)

                Barcode = data_SRT_clu['Barcodes']
                data_SRT_nEphys_Coordinates = data_SRT_clu['Coordinates in nEphys']

                barcode_index = list(np.where(np.in1d(np.asarray(gene_expression_matrix['Barcodes']), Barcode))[0])
                gene_expression_list = np.asarray(gene_expression_matrix[gene])[barcode_index]
                Gene_last = np.asarray(gene_expression_matrix['Barcodes'])[barcode_index]

                LFP_all = []
                delay_all = []
                energy_all = []
                frequency_all = []
                amplitude_all = []
                positive_peaks_all = []
                negative_peaks_all = []
                positive_peak_count_all = []
                negative_peak_count_all = []
                CT_all = []
                CV2_all = []
                Fano_all = []

                for cor in range(len(Gene_last)):
                    index_in_correlation = list(Barcode).index(Gene_last[cor])
                    related_LFP_cor = literal_eval(data_SRT_nEphys_Coordinates[index_in_correlation])
                    lfp_rate_mean = []
                    delay_mean = []
                    energy_mean = []
                    frequency_mean = []
                    amplitude_mean = []
                    positive_peaks_mean = []
                    negative_peaks_mean = []
                    positive_peak_count_mean = []
                    negative_peak_count_mean = []
                    CT_mean = []
                    CV2_mean = []
                    Fano_mean = []
                    for cor_in_nEphys in related_LFP_cor:
                        if cor_in_nEphys in list(coordinate_nEphys):
                            index_in_nEphys = list(coordinate_nEphys).index(cor_in_nEphys)
                            lfp_rate_mean.append(float(data_nEphys['LFP Rate(Event/min)'][index_in_nEphys]))
                            delay_mean.append(float(data_nEphys['Delay(s)'][index_in_nEphys]))
                            energy_mean.append(float(data_nEphys['Energy'][index_in_nEphys]))
                            frequency_mean.append(float(data_nEphys['Frequency'][index_in_nEphys]))
                            amplitude_mean.append(float(data_nEphys['Amplitude(uV)'][index_in_nEphys]))
                            positive_peaks_mean.append(
                                float(data_nEphys['Mean Positive Peaks(uV)'][index_in_nEphys]))
                            negative_peaks_mean.append(
                                float(data_nEphys['Mean Negative Peaks(uV)'][index_in_nEphys]))
                            if abs(float(data_nEphys['Mean Positive Peaks(uV)'][index_in_nEphys])) >= abs(
                                    float(data_nEphys['Mean Negative Peaks(uV)'][index_in_nEphys])):
                                amplitude_mean.append(float(data_nEphys['Mean Positive Peaks(uV)'][index_in_nEphys]))
                            else:
                                amplitude_mean.append(float(data_nEphys['Mean Negative Peaks(uV)'][index_in_nEphys]))
                            positive_peak_count_mean.append(
                                float(data_nEphys['Positive Peak Count'][index_in_nEphys]))
                            negative_peak_count_mean.append(
                                float(data_nEphys['Negative Peak Count'][index_in_nEphys]))
                            CT_mean.append(float(data_nEphys['CT'][index_in_nEphys]))
                            CV2_mean.append(float(data_nEphys['CV2'][index_in_nEphys]))
                            Fano_mean.append(float(data_nEphys['Fano Factor'][index_in_nEphys]))

                        else:
                            lfp_rate_mean.append(0)
                            delay_mean.append(0)
                            energy_mean.append(0)
                            frequency_mean.append(0)
                            amplitude_mean.append(0)
                            positive_peaks_mean.append(0)
                            negative_peaks_mean.append(0)
                            positive_peak_count_mean.append(0)
                            negative_peak_count_mean.append(0)
                            CT_mean.append(0)
                            CV2_mean.append(0)
                            Fano_mean.append(0)

                    LFP_all.append(np.mean([i if i == i else 0 for i in lfp_rate_mean]))
                    delay_all.append(np.mean([i if i == i else 0 for i in delay_mean]))
                    energy_all.append(np.mean([i if i == i else 0 for i in energy_mean]))
                    frequency_all.append(np.mean([i if i == i else 0 for i in frequency_mean]))
                    amplitude_all.append(np.mean([i if i == i else 0 for i in amplitude_mean]))
                    positive_peaks_all.append(np.mean([i if i == i else 0 for i in positive_peaks_mean]))
                    negative_peaks_all.append(np.mean([i if i == i else 0 for i in negative_peaks_mean]))
                    positive_peak_count_all.append(np.mean([i if i == i else 0 for i in positive_peak_count_mean]))
                    negative_peak_count_all.append(np.mean([i if i == i else 0 for i in negative_peak_count_mean]))
                    CT_all.append(np.mean([i if i == i else 0 for i in CT_mean]))
                    CV2_all.append(np.mean([i if i == i else 0 for i in CV2_mean]))
                    Fano_all.append(np.mean([i if i == i else 0 for i in Fano_mean]))

                ################################
                corr_LFP,p_LFP = self.relation_p_values(varx=gene_expression_list, vary=LFP_all)
                Gene_Name_all.append(gene)
                Parameters_all.append('LFPRate')
                Cluster_all.append(clu)
                Correlation_all.append(corr_LFP)
                p_values_all.append(p_LFP)
                File_name_all.append(expFile[:-4])
                for con_name in conditions:
                    if con_name in expFile[:-4]:
                        Condition_all.append(con_name)
                ################################
                corr_delay, p_delay = self.relation_p_values(varx=gene_expression_list, vary=delay_all)
                Gene_Name_all.append(gene)
                Parameters_all.append('Delay')
                Cluster_all.append(clu)
                Correlation_all.append(corr_delay)
                p_values_all.append(p_delay)
                File_name_all.append(expFile[:-4])
                for con_name in conditions:
                    if con_name in expFile[:-4]:
                        Condition_all.append(con_name)
                ################################
                corr_energy, p_energy = self.relation_p_values(varx=gene_expression_list, vary=energy_all)
                Gene_Name_all.append(gene)
                Parameters_all.append('Energy')
                Cluster_all.append(clu)
                Correlation_all.append(corr_energy)
                p_values_all.append(p_energy)
                File_name_all.append(expFile[:-4])
                for con_name in conditions:
                    if con_name in expFile[:-4]:
                        Condition_all.append(con_name)
                ################################
                corr_frequency, p_frequency = self.relation_p_values(varx=gene_expression_list, vary=frequency_all)
                Gene_Name_all.append(gene)
                Parameters_all.append('Frequency')
                Cluster_all.append(clu)
                Correlation_all.append(corr_frequency)
                p_values_all.append(p_frequency)
                File_name_all.append(expFile[:-4])
                for con_name in conditions:
                    if con_name in expFile[:-4]:
                        Condition_all.append(con_name)
                ################################
                corr_amplitude, p_amplitude = self.relation_p_values(varx=gene_expression_list, vary=amplitude_all)
                Gene_Name_all.append(gene)
                Parameters_all.append('Amplitude')
                Cluster_all.append(clu)
                Correlation_all.append(corr_amplitude)
                p_values_all.append(p_amplitude)
                File_name_all.append(expFile[:-4])
                for con_name in conditions:
                    if con_name in expFile[:-4]:
                        Condition_all.append(con_name)
                ################################
                corr_positive_peaks, p_positive_peaks = self.relation_p_values(varx=gene_expression_list, vary=positive_peaks_all)
                Gene_Name_all.append(gene)
                Parameters_all.append('positive peaks')
                Cluster_all.append(clu)
                Correlation_all.append(corr_positive_peaks)
                p_values_all.append(p_positive_peaks)
                File_name_all.append(expFile[:-4])
                for con_name in conditions:
                    if con_name in expFile[:-4]:
                        Condition_all.append(con_name)
                ################################
                corr_negative_peaks, p_negative_peaks = self.relation_p_values(varx=gene_expression_list, vary=negative_peaks_all)
                Gene_Name_all.append(gene)
                Parameters_all.append('negative peaks')
                Cluster_all.append(clu)
                Correlation_all.append(corr_negative_peaks)
                p_values_all.append(p_negative_peaks)
                File_name_all.append(expFile[:-4])
                for con_name in conditions:
                    if con_name in expFile[:-4]:
                        Condition_all.append(con_name)
                ################################
                corr_positive_peak_count, p_positive_peak_count = self.relation_p_values(varx=gene_expression_list,vary=positive_peak_count_all)
                Gene_Name_all.append(gene)
                Parameters_all.append('positive_peak_count')
                Cluster_all.append(clu)
                Correlation_all.append(corr_positive_peak_count)
                p_values_all.append(p_positive_peak_count)
                File_name_all.append(expFile[:-4])
                for con_name in conditions:
                    if con_name in expFile[:-4]:
                        Condition_all.append(con_name)
                ################################
                corr_negative_peak_count, p_negative_peak_count = self.relation_p_values(varx=gene_expression_list, vary=negative_peak_count_all)
                Gene_Name_all.append(gene)
                Parameters_all.append('negative_peak_count')
                Cluster_all.append(clu)
                Correlation_all.append(corr_negative_peak_count)
                p_values_all.append(p_negative_peak_count)
                File_name_all.append(expFile[:-4])
                for con_name in conditions:
                    if con_name in expFile[:-4]:
                        Condition_all.append(con_name)
                ################################
                corr_CT, p_CT = self.relation_p_values(varx=gene_expression_list,vary=CT_all)
                # network_activity_feature = ['Amplitude', 'Frequency']
                Gene_Name_all.append(gene)
                Parameters_all.append('CT')
                Cluster_all.append(clu)
                Correlation_all.append(corr_CT)
                p_values_all.append(p_CT)
                File_name_all.append(expFile[:-4])
                for con_name in conditions:
                    if con_name in expFile[:-4]:
                        Condition_all.append(con_name)
                ################################
                corr_CV2, p_CV2 = self.relation_p_values(varx=gene_expression_list, vary=CV2_all)
                Gene_Name_all.append(gene)
                Parameters_all.append('CV2')
                Cluster_all.append(clu)
                Correlation_all.append(corr_CV2)
                p_values_all.append(p_CV2)
                File_name_all.append(expFile[:-4])
                for con_name in conditions:
                    if con_name in expFile[:-4]:
                        Condition_all.append(con_name)
                ################################
                corr_Fano, p_Fano = self.relation_p_values(varx=gene_expression_list, vary=Fano_all)
                Gene_Name_all.append(gene)
                Parameters_all.append('Fano')
                Cluster_all.append(clu)
                Correlation_all.append(corr_Fano)
                p_values_all.append(p_Fano)
                File_name_all.append(expFile[:-4])
                for con_name in conditions:
                    if con_name in expFile[:-4]:
                        Condition_all.append(con_name)

        a = {'Gene Name': Gene_Name_all, 'Parameters': Parameters_all,'Correlation': Correlation_all,
             'p_values': p_values_all,
             "Cluster": Cluster_all,'File name':File_name_all,'Condition':Condition_all}
        df = pd.DataFrame.from_dict(a, orient='index').T
        df.to_excel(self.srcfilepath + expFile[:-4] + '_all_gene_expression_network_activity_features_p_values' + ".xlsx",
                    index=False)

    def all_gene_expression_network_activity_features_correlation_without_filter(self):
        """
         Correlate specified network activity feature from nEphys data with all gene expression values from SRT data without filter.
         Filter removes genes based on gene expression level. If only a few nodes had a highly expressed gene and the rest low or no expression it would be removed. Without filter allows all gene expression in for analysis.

             File input needed:
             -------
                 - '[file].bxr'
                 - 'all_gene_expression_per_coordinate_without_filter.npy'
                 - 'all_gene_expression_per_coordinate_without_filter_columns.npy'
                 - '_network_activity_features_per_cluster.xlsx'
                 - 'SRT_nEphys_Multiscale_Coordinates.xlsx'

             Parameters
             -------

             Returns
             -------

             File output:
             -------
                 - '_all_gene_expression_network_activity_features_p_values_without_filter.npy'
         """

        data = np.load(self.srcfilepath + "all_gene_expression_per_coordinate_without_filter.npy", allow_pickle=True)
        columns_matrix = np.load(self.srcfilepath + "all_gene_expression_per_coordinate_without_filter_columns.npy", allow_pickle=True)
        columns_matrix = list(columns_matrix)
        gene_expression_matrix = pd.DataFrame(data, columns=columns_matrix)
        # print(gene_expression_matrix)
        gene_all_names = columns_matrix[2:]
        ###################################Choose gene way 2
        filehdf5_bxr_name = '.bxr'
        filehdf5_bxr_file, filehdf5_bxr_Root = self.get_filename_path(self.srcfilepath, filehdf5_bxr_name)

        for i in range(len(filehdf5_bxr_file)):
            if filehdf5_bxr_file[i][0] != '.':
                expFile = filehdf5_bxr_file[i]

        data_nEphys = pd.read_excel(self.srcfilepath + expFile[:-4] + "_network_activity_features_per_cluster.xlsx")
        coordinate_nEphys = [[data_nEphys['x_coordinate'][i], data_nEphys['y_coordinate'][i]] for i in range(len(data_nEphys['x_coordinate']))]
        ###################################Choose gene way 2
        filetype_SRT_nEphys_Coordinates = 'SRT_nEphys_Multiscale_Coordinates.xlsx'
        filename_SRT_nEphys_Coordinates, Root = self.get_filename_path(self.srcfilepath, filetype_SRT_nEphys_Coordinates)
        for i in range(len(filename_SRT_nEphys_Coordinates)):
            if filename_SRT_nEphys_Coordinates[i][0] != '.':
                SRT_nEphys_Coordinates_root = Root[i] + '/' + filename_SRT_nEphys_Coordinates[i]

        data_SRT_nEphys_Coordinates_raw = pd.read_excel(SRT_nEphys_Coordinates_root)

        Gene_Name_all = []
        Parameters_all = []
        Cluster_all = []
        Correlation_all = []
        p_values_all = []
        Condition_all = []
        File_name_all = []

        cluster_all = self.clusters
        cluster_all.append('All')
        for gene in gene_all_names:
            for clu in cluster_all:
                ####################get the position of gene
                data_SRT_clu = data_SRT_nEphys_Coordinates_raw.copy()
                if clu != 'All':
                    data_SRT_clu = data_SRT_clu[data_SRT_clu['Cluster'] == clu]
                    s = pd.Series(range(len(data_SRT_clu)))
                    data_SRT_clu = data_SRT_clu.set_index(s)

                Barcode = data_SRT_clu['Barcodes']
                data_SRT_nEphys_Coordinates = data_SRT_clu['Coordinates in nEphys']

                barcode_index = list(np.where(np.in1d(np.asarray(gene_expression_matrix['Barcodes']), Barcode))[0])
                gene_expression_list = np.asarray(gene_expression_matrix[gene])[barcode_index]
                Gene_last = np.asarray(gene_expression_matrix['Barcodes'])[barcode_index]

                LFP_all = []
                delay_all = []
                energy_all = []
                frequency_all = []
                amplitude_all = []
                positive_peaks_all = []
                negative_peaks_all = []
                positive_peak_count_all = []
                negative_peak_count_all = []
                CT_all = []
                CV2_all = []
                Fano_all = []

                for cor in range(len(Gene_last)):
                    index_in_correlation = list(Barcode).index(Gene_last[cor])
                    related_LFP_cor = literal_eval(data_SRT_nEphys_Coordinates[index_in_correlation])
                    lfp_rate_mean = []
                    delay_mean = []
                    energy_mean = []
                    frequency_mean = []
                    amplitude_mean = []
                    positive_peaks_mean = []
                    negative_peaks_mean = []
                    positive_peak_count_mean = []
                    negative_peak_count_mean = []
                    CT_mean = []
                    CV2_mean = []
                    Fano_mean = []
                    for cor_in_nEphys in related_LFP_cor:
                        if cor_in_nEphys in list(coordinate_nEphys):
                            index_in_nEphys = list(coordinate_nEphys).index(cor_in_nEphys)
                            lfp_rate_mean.append(float(data_nEphys['LFP Rate(Event/min)'][index_in_nEphys]))
                            delay_mean.append(float(data_nEphys['Delay(s)'][index_in_nEphys]))
                            energy_mean.append(float(data_nEphys['Energy'][index_in_nEphys]))
                            frequency_mean.append(float(data_nEphys['Frequency'][index_in_nEphys]))
                            amplitude_mean.append(float(data_nEphys['Amplitude(uV)'][index_in_nEphys]))
                            positive_peaks_mean.append(
                                float(data_nEphys['Mean Positive Peaks(uV)'][index_in_nEphys]))
                            negative_peaks_mean.append(
                                float(data_nEphys['Mean Negative Peaks(uV)'][index_in_nEphys]))
                            if abs(float(data_nEphys['Mean Positive Peaks(uV)'][index_in_nEphys])) >= abs(
                                    float(data_nEphys['Mean Negative Peaks(uV)'][index_in_nEphys])):
                                amplitude_mean.append(float(data_nEphys['Mean Positive Peaks(uV)'][index_in_nEphys]))
                            else:
                                amplitude_mean.append(float(data_nEphys['Mean Negative Peaks(uV)'][index_in_nEphys]))
                            positive_peak_count_mean.append(
                                float(data_nEphys['Positive Peak Count'][index_in_nEphys]))
                            negative_peak_count_mean.append(
                                float(data_nEphys['Negative Peak Count'][index_in_nEphys]))
                            CT_mean.append(float(data_nEphys['CT'][index_in_nEphys]))
                            CV2_mean.append(float(data_nEphys['CV2'][index_in_nEphys]))
                            Fano_mean.append(float(data_nEphys['Fano Factor'][index_in_nEphys]))

                        else:
                            lfp_rate_mean.append(0)
                            delay_mean.append(0)
                            energy_mean.append(0)
                            frequency_mean.append(0)
                            amplitude_mean.append(0)
                            positive_peaks_mean.append(0)
                            negative_peaks_mean.append(0)
                            positive_peak_count_mean.append(0)
                            negative_peak_count_mean.append(0)
                            CT_mean.append(0)
                            CV2_mean.append(0)
                            Fano_mean.append(0)

                    LFP_all.append(np.mean([i if i == i else 0 for i in lfp_rate_mean]))
                    delay_all.append(np.mean([i if i == i else 0 for i in delay_mean]))
                    energy_all.append(np.mean([i if i == i else 0 for i in energy_mean]))
                    frequency_all.append(np.mean([i if i == i else 0 for i in frequency_mean]))
                    amplitude_all.append(np.mean([i if i == i else 0 for i in amplitude_mean]))
                    positive_peaks_all.append(np.mean([i if i == i else 0 for i in positive_peaks_mean]))
                    negative_peaks_all.append(np.mean([i if i == i else 0 for i in negative_peaks_mean]))
                    positive_peak_count_all.append(np.mean([i if i == i else 0 for i in positive_peak_count_mean]))
                    negative_peak_count_all.append(np.mean([i if i == i else 0 for i in negative_peak_count_mean]))
                    CT_all.append(np.mean([i if i == i else 0 for i in CT_mean]))
                    CV2_all.append(np.mean([i if i == i else 0 for i in CV2_mean]))
                    Fano_all.append(np.mean([i if i == i else 0 for i in Fano_mean]))
                    # ################################
                corr_LFP, p_LFP = self.relation_p_values(varx=gene_expression_list, vary=LFP_all)
                Gene_Name_all.append(gene)
                Parameters_all.append('LFPRate')
                Cluster_all.append(clu)
                Correlation_all.append(corr_LFP)
                p_values_all.append(p_LFP)
                File_name_all.append(expFile[:-4])
                for con_name in conditions:
                    if con_name in expFile[:-4]:
                        Condition_all.append(con_name)
                ################################
                corr_delay, p_delay = self.relation_p_values(varx=gene_expression_list, vary=delay_all)
                Gene_Name_all.append(gene)
                Parameters_all.append('Delay')
                Cluster_all.append(clu)
                Correlation_all.append(corr_delay)
                p_values_all.append(p_delay)
                File_name_all.append(expFile[:-4])
                for con_name in conditions:
                    if con_name in expFile[:-4]:
                        Condition_all.append(con_name)
                ################################
                corr_energy, p_energy = self.relation_p_values(varx=gene_expression_list, vary=energy_all)
                Gene_Name_all.append(gene)
                Parameters_all.append('Energy')
                Cluster_all.append(clu)
                Correlation_all.append(corr_energy)
                p_values_all.append(p_energy)
                File_name_all.append(expFile[:-4])
                for con_name in conditions:
                    if con_name in expFile[:-4]:
                        Condition_all.append(con_name)
                ################################
                corr_frequency, p_frequency = self.relation_p_values(varx=gene_expression_list, vary=frequency_all)
                Gene_Name_all.append(gene)
                Parameters_all.append('Frequency')
                Cluster_all.append(clu)
                Correlation_all.append(corr_frequency)
                p_values_all.append(p_frequency)
                File_name_all.append(expFile[:-4])
                for con_name in conditions:
                    if con_name in expFile[:-4]:
                        Condition_all.append(con_name)
                ################################
                corr_amplitude, p_amplitude = self.relation_p_values(varx=gene_expression_list, vary=amplitude_all)
                Gene_Name_all.append(gene)
                Parameters_all.append('Amplitude')
                Cluster_all.append(clu)
                Correlation_all.append(corr_amplitude)
                p_values_all.append(p_amplitude)
                File_name_all.append(expFile[:-4])
                for con_name in conditions:
                    if con_name in expFile[:-4]:
                        Condition_all.append(con_name)
                ################################
                corr_positive_peaks, p_positive_peaks = self.relation_p_values(varx=gene_expression_list, vary=positive_peaks_all)
                Gene_Name_all.append(gene)
                Parameters_all.append('positive peaks')
                Cluster_all.append(clu)
                Correlation_all.append(corr_positive_peaks)
                p_values_all.append(p_positive_peaks)
                File_name_all.append(expFile[:-4])
                for con_name in conditions:
                    if con_name in expFile[:-4]:
                        Condition_all.append(con_name)
                ################################
                corr_negative_peaks, p_negative_peaks = self.relation_p_values(varx=gene_expression_list, vary=negative_peaks_all)
                Gene_Name_all.append(gene)
                Parameters_all.append('negative peaks')
                Cluster_all.append(clu)
                Correlation_all.append(corr_negative_peaks)
                p_values_all.append(p_negative_peaks)
                File_name_all.append(expFile[:-4])
                for con_name in conditions:
                    if con_name in expFile[:-4]:
                        Condition_all.append(con_name)
                ################################
                corr_positive_peak_count, p_positive_peak_count = self.relation_p_values(varx=gene_expression_list, vary=positive_peak_count_all)
                Gene_Name_all.append(gene)
                Parameters_all.append('positive_peak_count')
                Cluster_all.append(clu)
                Correlation_all.append(corr_positive_peak_count)
                p_values_all.append(p_positive_peak_count)
                File_name_all.append(expFile[:-4])
                for con_name in conditions:
                    if con_name in expFile[:-4]:
                        Condition_all.append(con_name)
                ################################
                corr_negative_peak_count, p_negative_peak_count = self.relation_p_values(varx=gene_expression_list,vary=negative_peak_count_all)
                Gene_Name_all.append(gene)
                Parameters_all.append('negative_peak_count')
                Cluster_all.append(clu)
                Correlation_all.append(corr_negative_peak_count)
                p_values_all.append(p_negative_peak_count)
                File_name_all.append(expFile[:-4])
                for con_name in conditions:
                    if con_name in expFile[:-4]:
                        Condition_all.append(con_name)
                ################################
                corr_CT, p_CT = self.relation_p_values(varx=gene_expression_list,vary=CT_all)
                Gene_Name_all.append(gene)
                Parameters_all.append('CT')
                Cluster_all.append(clu)
                Correlation_all.append(corr_CT)
                p_values_all.append(p_CT)
                File_name_all.append(expFile[:-4])
                for con_name in conditions:
                    if con_name in expFile[:-4]:
                        Condition_all.append(con_name)
                ################################
                corr_CV2, p_CV2 = self.relation_p_values(varx=gene_expression_list, vary=CV2_all)
                Gene_Name_all.append(gene)
                Parameters_all.append('CV2')
                Cluster_all.append(clu)
                Correlation_all.append(corr_CV2)
                p_values_all.append(p_CV2)
                File_name_all.append(expFile[:-4])
                for con_name in conditions:
                    if con_name in expFile[:-4]:
                        Condition_all.append(con_name)
                ################################
                corr_Fano, p_Fano = self.relation_p_values(varx=gene_expression_list, vary=Fano_all)
                Gene_Name_all.append(gene)
                Parameters_all.append('Fano')
                Cluster_all.append(clu)
                Correlation_all.append(corr_Fano)
                p_values_all.append(p_Fano)
                File_name_all.append(expFile[:-4])
                for con_name in conditions:
                    if con_name in expFile[:-4]:
                        Condition_all.append(con_name)

        a = {'Gene Name': Gene_Name_all, 'Parameters': Parameters_all ,'Correlation': Correlation_all,
             'p_values': p_values_all,
             "Cluster": Cluster_all ,'File name' :File_name_all ,'Condition' :Condition_all}
        df = pd.DataFrame.from_dict(a, orient='index').T
        np.save(self.srcfilepath + expFile[:-4] + '_all_gene_expression_network_activity_features_p_values_without_filter', df)
        # df.to_excel(self.srcfilepath + expFile[:-4] + '_all_gene_expression_network_activity_features_p_values_without_filter' + ".xlsx",index=False)

    def all_gene_expression_network_activity_features_correlation_pooled_statistics(self):
        """
         Compare correlation statistics between input conditions i.e. SD and ENR.

             File input needed:
             -------
                 - '_all_gene_expression_network_activity_features_p_values.xlsx'

             Parameters
             -------

             Returns
             -------

             File output:
             -------
                 - 'all_gene_expression_network_activity_features_p_values_pooled_statistics.xlsx'
                 - 'all_gene_expression_network_activity_features_p_values_pooled_statistics.png'
         """

        path = self.srcfilepath[:self.srcfilepath.rfind('/')]
        desfilepath = path + '/Correlated_Network_Activity_Features_Pooled_Statistics/'
        if not os.path.exists(desfilepath):
            os.mkdir(desfilepath)
        cluster_all = self.clusters
        cluster_all.append('All')
        if os.path.exists(self.srcfilepath + 'all_gene_expression_network_activity_features_p_values' + ".xlsx"):
            final = pd.read_excel(self.srcfilepath + 'all_gene_expression_network_activity_features_p_values_' + ".xlsx")
        else:
            filetype_correlation = '_all_gene_expression_network_activity_features_p_values.xlsx'
            filename_correlation, Root_correlation = self.get_filename_path(self.srcfilepath, filetype_correlation)
            final_Gene_Expression_network_activity_feature = pd.DataFrame()
            for i in range(len(filename_correlation)):
                if filename_correlation[i][0] != '.':
                    gene_root = Root_correlation[i] + '/' + filename_correlation[i]
                    Correlation_choose = pd.read_excel(gene_root)
                    Correlation_choose.dropna(axis=0, how='any', inplace=True)
                    final_Gene_Expression_network_activity_feature = pd.concat([final_Gene_Expression_network_activity_feature, Correlation_choose],axis=0)

            p_adj = []
            p_adj_pattern = []
            final_Gene_Expression_network_activity_feature_new = pd.DataFrame()
            for clu in cluster_all:
                df_new_clu = final_Gene_Expression_network_activity_feature.copy()
                df_new_clu = df_new_clu[df_new_clu["Cluster"] == clu]
                s = pd.Series(range(len(df_new_clu)))
                df_new_clu = df_new_clu.set_index(s)
                # df_new_clu['p_values'] = df_new_clu['p_values'].div(10)
                final_Gene_Expression_network_activity_feature_new = pd.concat(
                    [final_Gene_Expression_network_activity_feature_new, df_new_clu],
                    axis=0)

                pval1 = np.asarray(df_new_clu['p_values'])
                reject, pvalscorr = sm.multipletests(pval1, method='fdr_tsbh')[:2]  # family-wise error rate: alpha
                p_adj.extend(pvalscorr)
                for i in pvalscorr:
                    if i <= 0.01:
                        p_adj_pattern.append('p_adj<=0.01')
                    elif i > 0.01 and i < 0.05:
                        p_adj_pattern.append('0.01< p_adj < 0.05')
                    elif i > 0.05 and i < 0.1:
                        p_adj_pattern.append('0.05<= p_adj < 0.1')
                    else:
                        p_adj_pattern.append('Rejected')
            df_add = pd.DataFrame({'p_adj': p_adj, 'p_adj_pattern': p_adj_pattern})
            s = pd.Series(range(len(final_Gene_Expression_network_activity_feature_new)))
            final_Gene_Expression_network_activity_feature_new = final_Gene_Expression_network_activity_feature_new.set_index(s)
            final = pd.concat([final_Gene_Expression_network_activity_feature_new, df_add], axis=1)
            final.to_excel(desfilepath + 'all_gene_expression_network_activity_features_p_values_pooled_statistics' + ".xlsx",index=False)

        ##################################
        p_adj_pattern_uni = ['p_adj<=0.01', '0.01< p_adj < 0.05', '0.05<= p_adj < 0.1']
        fig, ax = plt.subplots(nrows=len(cluster_all), ncols=len(conditions), figsize=(30, 40))  # , facecolor='None'
        clu_count = 0
        for clu in cluster_all:
            con_count = 0
            final_clu = final.copy()
            final_clu = final_clu[final_clu['p_adj_pattern'] != 'Rejected']
            s = pd.Series(range(len(final_clu)))
            final_clu = final_clu.set_index(s)

            final_clu = final_clu.copy()
            final_clu = final_clu[final_clu["Cluster"] == clu]
            s = pd.Series(range(len(final_clu)))
            final_clu = final_clu.set_index(s)
            for con in conditions:
                final_clu_con = final_clu.copy()
                final_clu_con = final_clu_con[final_clu_con['Condition'] == con]
                s = pd.Series(range(len(final_clu_con)))
                final_clu_con = final_clu_con.set_index(s)
                pattern_count = 0
                for patten in p_adj_pattern_uni:
                    final_clu_con_pattern = final_clu_con.copy()
                    final_clu_con_pattern = final_clu_con_pattern[final_clu_con_pattern['p_adj_pattern'] == patten]
                    s = pd.Series(range(len(final_clu_con_pattern)))

                    final_clu_con_pattern = final_clu_con_pattern.set_index(s)
                    try:
                        sns.barplot(x=final_clu_con_pattern.Parameters.value_counts().index,
                                    y=final_clu_con_pattern.Parameters.value_counts(), ax=ax[clu_count, con_count],
                                    color=color_choose[pattern_count],label = patten)
                    except:
                        continue
                    pattern_count += 1
                try:
                    sns.countplot(x='Parameters',data=final_clu_con,ax = ax[clu_count, con_count], palette=color_choose[:len(p_adj_pattern_uni)],hue='p_adj_pattern')
                except:
                    continue
                ax[clu_count, con_count].legend(loc='best', fontsize='xx-small')
                ax[clu_count, con_count].tick_params(axis="x", labelsize=6, labelrotation=20)
                ax[clu_count, con_count].spines['top'].set_visible(False)
                ax[clu_count, con_count].spines['right'].set_visible(False)
                if con_count == 0:
                    ax[clu_count, con_count].set_ylabel(clu, fontsize=8)
                else:
                    ax[clu_count, con_count].set_ylabel('Gene Count', fontsize=8)
                ax[clu_count, con_count].set_title(con, fontsize=10, fontweight="bold")
                con_count += 1

            clu_count += 1
        fig.tight_layout()
        fig.savefig(desfilepath + 'all_gene_expression_network_activity_features_p_values_pooled_statistics' + ".png",
                    format='png', dpi=600)
        plt.close()

    def all_gene_expression_network_activity_features_correlation_pooled_statistics_without_filter(self):
        """
         Compare correlation statistics between input conditions i.e. SD and ENR.

             File input needed:
             -------
                 - '_all_gene_expression_network_activity_features_p_values_without_filter.npy'

             Parameters
             -------

             Returns
             -------

             File output:
             -------
                 - 'all_gene_expression_network_activity_features_p_values_without_filter_[con]_[clu]_[par].xlsx'
         """

        path = self.srcfilepath[:self.srcfilepath.rfind('/')]
        desfilepath = path + '/Correlated_Network_Activity_Features_Pooled_Statistics/'
        if not os.path.exists(desfilepath):
            os.mkdir(desfilepath)

        cluster_all = self.clusters
        cluster_all.append('All')
        filetype_correlation = '_all_gene_expression_network_activity_features_p_values_without_filter.npy'
        filename_correlation, Root_correlation = self.get_filename_path(self.srcfilepath, filetype_correlation)
        final_Gene_Expression_network_activity_feature = pd.DataFrame()
        for i in range(len(filename_correlation)):
            if filename_correlation[i][0] != '.':
                gene_root = Root_correlation[i] + '/' + filename_correlation[i]
                # data = pd.read_excel(gene_root)
                data = np.load(gene_root, allow_pickle=True)
                columns_matrix = ['Gene Name', 'Parameters', 'Correlation', 'p_values', "Cluster", 'File name',
                                  'Condition']
                Correlation_choose = pd.DataFrame(data, columns=columns_matrix)
                Correlation_choose.dropna(axis=0, how='any', inplace=True)
                final_Gene_Expression_network_activity_feature = pd.concat(
                    [final_Gene_Expression_network_activity_feature, Correlation_choose], axis=0)

            p_adj = []
            p_adj_pattern = []
            final_Gene_Expression_network_activity_feature_new = pd.DataFrame()
            for clu in cluster_all:
                df_new_clu = final_Gene_Expression_network_activity_feature.copy()
                df_new_clu = df_new_clu[df_new_clu["Cluster"] == clu]
                s = pd.Series(range(len(df_new_clu)))
                df_new_clu = df_new_clu.set_index(s)
                # df_new_clu['p_values'] = df_new_clu['p_values'].div(10)
                final_Gene_Expression_network_activity_feature_new = pd.concat(
                    [final_Gene_Expression_network_activity_feature_new, df_new_clu],
                    axis=0)

                pval1 = np.asarray(df_new_clu['p_values'])
                reject, pvalscorr = sm.multipletests(pval1, method='fdr_tsbh')[:2]  # family-wise error rate: alpha
                p_adj.extend(pvalscorr)
                for i in pvalscorr:
                    if i <= 0.01:
                        p_adj_pattern.append('p_adj<=0.01')
                    elif i > 0.01 and i < 0.05:
                        p_adj_pattern.append('0.01< p_adj < 0.05')
                    elif i > 0.05 and i < 0.1:
                        p_adj_pattern.append('0.05<= p_adj < 0.1')
                    else:
                        p_adj_pattern.append('Rejected')

            df_add = pd.DataFrame({'p_adj': p_adj, 'p_adj_pattern': p_adj_pattern})
            s = pd.Series(range(len(final_Gene_Expression_network_activity_feature_new)))
            final_Gene_Expression_network_activity_feature_new = final_Gene_Expression_network_activity_feature_new.set_index(
                s)
            final = pd.concat([final_Gene_Expression_network_activity_feature_new, df_add], axis=1)
            final = final[final['p_adj_pattern'] != 'Rejected']
            ##############################################
            for con in conditions:
                for clu in cluster_all:
                    for par in network_activity_feature:
                        df_new_clu = final.copy()
                        df_new_clu = df_new_clu[df_new_clu["Cluster"] == clu]
                        s = pd.Series(range(len(df_new_clu)))
                        df_new_clu = df_new_clu.set_index(s)

                        df_new_clu = df_new_clu[df_new_clu["Condition"] == con]
                        s = pd.Series(range(len(df_new_clu)))
                        df_new_clu = df_new_clu.set_index(s)

                        df_new_clu = df_new_clu[df_new_clu["Parameters"] == par]
                        s = pd.Series(range(len(df_new_clu)))
                        df_new_clu = df_new_clu.set_index(s)
                        ####################################
                        df_new_clu.to_excel(
                            desfilepath + 'all_gene_expression_network_activity_features_p_values_without_filter_' + con + '_' + clu + '_' + par + ".xlsx",
                            index=False)

    def biplot_for_gene_expression_electrophysiological_features(self, gene_list_name=None, choose_gene=False,k_means_auto=True,num_clus= 6,parameter_clusters_independent = True,cluster_based_on_structrue=True):
        """
         Plot biplots of specified network activity features from nEphys data with gene expression values from SRT gene lists.

             File input needed:
             -------
                 - '[gene_list]_gene_expression_network_activity_features.xlsx'

             Parameters
             -------

             Returns
             -------

             File output:
             -------
                 - '[gene_list]_gene_expression_clusters_based_on_kmeans.xlsx'
                 - '[gene_list]_network_activity_features_clusters_based_on_kmeans.xlsx'
                 - '[gene_list]_gene_expression_network_activity_features_biplot.png'
         """

        path = self.srcfilepath[:self.srcfilepath.rfind('/')]
        desfilepath = path + '/Correlated_Network_Activity_Features/'
        if not os.path.exists(desfilepath):
            os.mkdir(desfilepath)
        df_concat = pd.DataFrame()
        for gene_list in column_list:
            df_list = pd.read_excel(desfilepath + gene_list + '_gene_expression_network_activity_features' + ".xlsx")
            df_concat = pd.concat([df_concat, df_list], axis=0, ignore_index=False)
        df = df_concat

        if choose_gene:
            gene_name_list = select_genes
        else:
            gene_name_list = np.unique(df['Gene Name'])
        spot_uni = df[df['Gene Name'] == gene_name_list[0]]['Barcode']
        df_gene = pd.DataFrame() #{'Barcode':spot_uni}
        ###################################



        for gene_name_choose in list(gene_name_list):
            df_new = df.copy()
            df_new = df_new[df_new['Gene Name'] == gene_name_choose]
            s = pd.Series(range(len(df_new)))
            df_new = df_new.set_index(s)
            df_new['Gene Expression Level(Norm)'] = df_new['Gene Expression Level(Norm)'].astype(float, errors='raise')
            df_gene[gene_name_choose] = df_new['Gene Expression Level(Norm)']

        # Apply min-max scaling
        df_gene_copy = df_gene.copy()
        gene_expression_series = df_gene_copy.to_numpy()
        scaler = sk.preprocessing.MinMaxScaler()
        dataset_scaled_gene = scaler.fit_transform(gene_expression_series)
        # Do PCA
        pca = PCA()
        pca_result = pca.fit_transform(dataset_scaled_gene)
        if cluster_based_on_structrue:
            parameter_clusters_independent = True
            spot_uni_cluster = df[df['Gene Name'] == gene_name_list[0]]['Cluster']
            cluster = []
            for clu in spot_uni_cluster:
                cluster.append(np.where(np.array(self.clusters) == clu)[0][0])

        else:
            if k_means_auto == True:
                from sklearn.cluster import KMeans
                distortions = []
                K = range(1, 15)
                for k in K:
                    kmeanModel = KMeans(n_clusters=k)
                    kmeanModel.fit(pca_result)
                    distortions.append(kmeanModel.inertia_)
                #################choose the best k
                interval = 0
                K = 0
                for i in range(1, len(distortions) - 1):
                    if interval < abs(distortions[i] - distortions[i - 1]) / abs(
                            distortions[i + 1] - distortions[i]) and abs(distortions[i] - distortions[i - 1]) - abs(
                        distortions[i + 1] - distortions[i]) > 0:
                        interval = abs(distortions[i] - distortions[i - 1]) / abs(distortions[i + 1] - distortions[i])
                        K = i + 2
                num_clus = K
            else:
                num_clus = num_clus
            cluster, centers, distance = self.k_means(pca_result, num_clus)
        df_gene_raw = df_gene.copy
        df_gene['Barcode'] = spot_uni # {'Barcode':spot_uni}
        df_gene['Cluster'] = cluster
        df_gene.to_excel(desfilepath + gene_list_name + "_gene_expression_clusters_based_on_kmeans" + ".xlsx", index=False)
        ######################################################################
        df_parameters = pd.DataFrame()
        for Type_name in network_activity_feature:
            if Type_name == 'LFPRate':  ##network_activity_feature = 'LFPRate','Delay','Energy'
                type_name = 'LFP Rate(Event/min)'
            elif Type_name == 'Delay':
                type_name = 'Delay(s)'
            elif Type_name == 'Energy':
                type_name = 'Energy'
            elif Type_name == 'Frequency':
                type_name = 'Frequency'
            elif Type_name == 'Amplitude':
                type_name = 'Amplitude(uV)'
            elif Type_name == 'positive_peaks':
                type_name = 'Mean Positive Peaks(uV)'
            elif Type_name == 'negative_peaks':
                type_name = 'Mean Negative Peaks(uV)'
            elif Type_name == 'positive_peak_count':
                type_name = 'Positive Peak Count'
            elif Type_name == 'negative_peak_count':
                type_name = 'Negative Peak Count'
            elif Type_name == 'CT':
                type_name = 'CT'
            elif Type_name == 'CV2':
                type_name = 'CV2'
            else:
                type_name = 'Fano Factor'

            df_new = df.copy()
            df_new_Region = df_new.groupby(by=['Barcode'])[type_name].mean().reset_index()
            df_parameters[type_name] = [x if ~np.isnan(x) else 0 for x in list(df_new_Region[type_name])]

    #if parameter_clusters_independent == False:
        df_parameters_copy = df_parameters.copy()
        parameters_series = df_parameters_copy.to_numpy()
        scaler = sk.preprocessing.MinMaxScaler()
        dataset_scaled_para = scaler.fit_transform(parameters_series)
        #print('Check2')
        #print(dataset_scaled_para)
        # Do PCA
        pca = PCA()
        pca_result = pca.fit_transform(dataset_scaled_para)
        if k_means_auto == True:
            from sklearn.cluster import KMeans
            distortions = []
            K = range(1, 15)
            for k in K:
                kmeanModel = KMeans(n_clusters=k)
                kmeanModel.fit(pca_result)
                distortions.append(kmeanModel.inertia_)
            #################choose the best k
            interval = 0
            K = 0
            for i in range(1, len(distortions) - 1):
                if interval < abs(distortions[i] - distortions[i - 1]) / abs(
                        distortions[i + 1] - distortions[i]) and abs(distortions[i] - distortions[i - 1]) - abs(
                    distortions[i + 1] - distortions[i]) > 0:
                    interval = abs(distortions[i] - distortions[i - 1]) / abs(distortions[i + 1] - distortions[i])
                    K = i + 2
            num_clus = K
        else:
            num_clus = num_clus

        cluster_para, centers, distance = self.k_means(pca_result, num_clus)
        #else:
        #    cluster_para = cluster

        df_parameters_raw = df_parameters.copy
        df_parameters['Barcode'] = spot_uni  # {'Barcode':spot_uni}
        df_parameters['Cluster'] = cluster_para
        df_parameters.to_excel(desfilepath + gene_list_name + "_network_activity_features_clusters_based_on_kmeans" + ".xlsx",
                         index=False)
        ##############################################
        pca_gene = PCA(n_components=3, whiten=True)
        x_new_gene = pca_gene.fit_transform(dataset_scaled_gene)
        y_gene = cluster

        pca_para = PCA(n_components=3, whiten=True)
        x_new_para = pca_para.fit_transform(dataset_scaled_para)
        y_para = cluster_para

        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))
        #comp1 -comp 2

        # score = None, coeff = None, labels = None, cluster_num = None, ax = None
        self.bibiplot(score = x_new_gene[:,0:2],coeff = np.transpose(pca_gene.components_[0:2, :]),labels = gene_name_list, cluster_num = y_gene, ax = ax[0,0],xlabel = 1,ylabel = 2)
        self.bibiplot(score=x_new_gene[:, [0,2]], coeff=np.transpose(pca_gene.components_[[0,2], :]), labels=gene_name_list,
                      cluster_num=y_gene, ax= ax[1, 0],xlabel = 1,ylabel = 3)

        self.bibiplot(score=x_new_para[:, 0:2], coeff=np.transpose(pca_para.components_[0:2, :]), labels=network_activity_feature,
                      cluster_num=y_para, ax=ax[0, 1], xlabel=1, ylabel=2)
        self.bibiplot(score=x_new_para[:, [0, 2]], coeff=np.transpose(pca_para.components_[[0, 2], :]),
                      labels=network_activity_feature,cluster_num=y_para, ax=ax[1, 1], xlabel=1, ylabel=3)

        fig.tight_layout()
        colorMapTitle = gene_list_name + '_gene_expression_network_activity_features_biplot'
        fig.savefig(desfilepath + colorMapTitle + ".png", format='png', dpi=600)
        plt.close()

if __name__ == '__main__':
    srcfilepath = r'Z:/ANALYSES/SPATIOSCALES- 10X genomics/Data/'  # main path
    Analysis = MEASeqX_Project(srcfilepath)
    Analysis.network_activity_features(low=1, high=100)  # Step 1 individual
    Analysis.coordinates_for_network_activity_features() # Step 2 individual
    ################################################################# nEphys and SRT Network Activity Features Correlation Gene List
    for gene_list in column_list: # Step 3 individual and pooled
        for type_name in network_activity_feature:
            Analysis.gene_expression_network_activity_features_correlation(gene_list_name=gene_list,network_activity_feature = type_name) # individual
            Analysis.gene_expression_network_activity_features_correlation_pooled_statistics_per_cluster(gene_list_name=gene_list,network_activity_feature = type_name) # pooled condition statistics (main path should contain the condition subfolders)
        Analysis.gene_expression_network_activity_features_correlation_pooled_statistics_per_region(gene_list_name=gene_list,choose_gene = 'Bdnf') # pooled condition statistics (main path should contain the condition subfolders)
    ################################################################# nEphys and SRT Network Activity Features Correlation All Genes
    Analysis.all_gene_expression()  # Step 3 individual
    Analysis.all_gene_expression_without_filter() # Step 3 individual
    Analysis.all_gene_expression_network_activity_features_correlation() #Step 4 individual
    Analysis.all_gene_expression_network_activity_features_correlation_without_filter()  # Step 4 individual
    Analysis.all_gene_expression_network_activity_features_correlation_pooled_statistics()  # Step 5 pooled condition statistics (main path should contain the condition subfolders)
    Analysis.all_gene_expression_network_activity_features_correlation_pooled_statistics_without_filter()  # Step 5 pooled condition statistics (main path should contain the condition subfolders)

    Analysis.biplot_for_gene_expression_electrophysiological_features(gene_list_name="IEGs", choose_gene=False, k_means_auto=False, num_clus=4, parameter_clusters_independent=False, cluster_based_on_structrue=True)

'''
Filter removes genes based on gene expression level. If only a few nodes had a highly expressed gene and the rest low or no expression it would be removed. Without filter allows all gene expression in for analysis
'''
