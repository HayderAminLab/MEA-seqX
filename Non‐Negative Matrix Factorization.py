# -*- coding: utf-8 -*-
"""
Created on Aug 19 2021
@author:  Xin Hu
@company: DZNE
"""
import matplotlib.pyplot as plt
from anndata import AnnData
from scipy.signal import butter, lfilter, hilbert
from scipy import signal,stats
import scanpy as sc
import h5py
import numpy as np
import pandas as pd
import json
import os
import scipy.sparse as sp_sparse
import matplotlib.image as mpimg
from sklearn.decomposition import NMF
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

"""
The following input parameters are used for non‐negative matrix factorization.
To compare conditions put the path for datasets in the input parameters and label condition name i.e. SD and ENR and assign desired color

"""

rows = 64
cols = 64

column_list = ["IEGs"]  ### "Hipoo Signaling Pathway","Synaptic Vescicles_Adhesion","Receptors and channels","Synaptic plasticity","Hippocampal Neurogenesis","IEGs"
select_genes = ['Arc','Bdnf','Egr1','Egr3','Egr4','Fosb','Homer1','Homer3','Npas4','Nptx2','Nr4a1','Nr4a3']
gene_list_choose = ['gene1','gene2','gene3','gene4']

color = ['silver', 'dodgerblue']  # color for pooled plotting of conditions


Pixel_SRT_mm = 0.645/1000 #mm
length_nEphys = 2.67 #mm

network_activity_feature = ['LFPRate','Delay','Energy','Frequency','Amplitude','positive_peaks','negative_peaks','positive_peak_count','negative_peak_count','CT','CV2','Fano']
network_topology_metric = ['Clustering Coefficient',"Centrality","Degree",'Betweenness']

quantile_value = 0.75

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
        Get raw data
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

    def SRT_nEphys_network_activity_features_NMF_reconstruction_error(self, type_choose_name=None, min_k=1, max_k=100, fliter_gene=True, cluster='All', gene_list=None, rotate_matrix=True, plot=False,Given_gene_list=True):
        """
        Determine optimal number of factors for NMF decomposition.

            File input needed:
            -------
                - related files
                - 'SRT_nEphys_Multiscale_Coordinates_for_network_topological_metrics.xlsx'
                - 'SRT_nEphys_Multiscale_Coordinates_for_network_activity_features.xlsx'
                - 'gene_list_all.xlsx'

            Parameters
            -------

            Returns
            -------

            File output:
            -------
                - [network_activity_feature]_[cluster]_reconstruction_error.xlsx'
        """

        path = self.srcfilepath[:self.srcfilepath.rfind('/')]
        desfilepath = path + '/NMF/'
        if not os.path.exists(desfilepath):
            os.mkdir(desfilepath)
        K_list = []
        Error_list = []
        for k in range(min_k, max_k + 1):
            k = int(k)
            type = type_choose_name

            type_name = type + ' nEphys'

            # Read related information
            csv_file, tissue_lowres_scalef, features_name, matr_raw, barcodes, img, csv_file_cluster = self.read_related_files()
            barcode_cluster = np.asarray(csv_file_cluster["Barcode"])
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
            deleted_notes_cluster = [i for i in range(len(genes_per_cell)) if
                                     str(barcodes[i])[2:-1] not in barcode_cluster]
            deleted_notes.extend(deleted_notes_cluster)
            deleted_notes = list(np.unique(deleted_notes))
            # ##########################################################
            matr = np.delete(matr, deleted_notes, axis=1)

            barcodes_filter = [barcodes[i] for i in range(len(barcodes)) if i not in deleted_notes]
            # new_id = [j for i in barcode_CSV for j in range(len(barcodes)) if str(barcodes[j])[2:-1] == i]

            adata = AnnData(np.array(matr))
            sc.pp.normalize_total(adata, inplace=True)
            gene_expression = adata.X
            genes_expression_count = np.asarray((matr > 0).sum(axis=1)).squeeze()
            ###################################Choose gene 1 way
            df_gene_count = pd.DataFrame({'Gene Name': gene_name, 'Gene Count sorted': list(genes_expression_count)})
            gene_expression_series = np.asarray(gene_expression)
            if Given_gene_list == True:
                gene_add = gene_list_choose
                filter_gene_id = [i for gene in gene_add for i in range(len(gene_name)) if gene_name[i].lower() == gene]
                gene_expression_series = gene_expression_series[filter_gene_id]
                gene_expression_series = np.asarray(gene_expression_series).T
                filter_gene_expression_matrix = gene_expression_series
                remain_gene_name = np.asarray(df_gene_count['Gene Name'])
                remain_gene_name = remain_gene_name[filter_gene_id]
            else:
                if gene_list != None:
                    select_genes_add = pd.read_excel(self.srcfilepath + 'gene_list_all.xlsx')
                    gene_add = [str(i).lower() for i in select_genes_add[gene_list] if str(i) != 'nan']
                    filter_gene_id = [i for gene in gene_add for i in range(len(gene_name)) if
                                      gene_name[i].lower() == gene]
                    # id_no_expression = [i for i in range(len(gene_expression_series)) if sum(gene_expression_series[i]) == 0]
                    gene_expression_series = gene_expression_series[filter_gene_id]
                    gene_expression_series = np.asarray(gene_expression_series).T
                    filter_gene_expression_matrix = gene_expression_series
                    remain_gene_name = np.asarray(df_gene_count['Gene Name'])
                    remain_gene_name = remain_gene_name[filter_gene_id]
                else:
                    if fliter_gene == True:
                        gene_expression_list = list(gene_expression_series[gene_expression_series > 0])
                        mean_B, low_threshold_B, high_threshold_B = self.quantile_bound(gene_expression_list)
                        gene_expression_series[gene_expression_series >= high_threshold_B] = 0
                        mean_expression_nodes = np.mean(
                            [int(np.asarray((np.asarray(gene_expression_series[i]) != 0).sum())) for i in
                             range(len(gene_expression_series))])
                        std_expression_nodes = np.std(
                            [int(np.asarray((np.asarray(gene_expression_series[i]) != 0).sum())) for i in
                             range(len(gene_expression_series))])
                        # ##############################################################
                        id_no_expression = [i for i in range(len(gene_expression_series)) if
                                            sum(gene_expression_series[i]) == 0 or int(np.asarray((np.asarray(
                                                gene_expression_series[
                                                    i]) != 0).sum())) <= mean_expression_nodes + std_expression_nodes]

                        gene_expression_series = np.delete(gene_expression_series, id_no_expression, axis=0)
                        gene_expression_series = np.asarray(gene_expression_series).T
                        expression_each_gene_mean = np.asarray(gene_expression_series).mean(axis=0)
                        expression_each_gene_std = np.asarray(gene_expression_series).std(axis=0)
                        index_mean = \
                            np.where(np.array(expression_each_gene_mean) >= np.asarray(gene_expression_series).mean())[0]
                        index_std = \
                            np.where(np.array(expression_each_gene_std) >= np.asarray(gene_expression_series).std())[0]
                        keep_gene_id = set(index_mean) & set(index_std)
                        filter_gene_expression_matrix = gene_expression_series[:, list(keep_gene_id)]
                        # print('remain_gene_name', pd.DataFrame(filter_gene_expression_matrix))
                        remain_gene_name = np.asarray(df_gene_count['Gene Name'])[list(keep_gene_id)]
                        # print('remain_gene_name', remain_gene_name)
                    #
                    else:
                        id_no_expression = [i for i in range(len(gene_expression_series)) if
                                            sum(gene_expression_series[i]) == 0]
                        gene_expression_series = np.delete(gene_expression_series, id_no_expression, axis=0)
                        gene_expression_series = np.asarray(gene_expression_series).T
                        filter_gene_expression_matrix = gene_expression_series
                        remain_gene_name = np.asarray(df_gene_count['Gene Name'])
                        remain_gene_name = list(
                            [remain_gene_name[i] for i in range(len(remain_gene_name)) if i not in id_no_expression])

            gene_expression_matrix = pd.DataFrame(filter_gene_expression_matrix)
            barcodes_filter_1 = [str(barcodes_filter[i])[2:-1] for i in range(len(barcodes_filter))]

            ########################
            data_Basic_Parameters_correlation = pd.read_excel(
                self.srcfilepath + 'SRT_nEphys_Multiscale_Coordinates_for_network_activity_features.xlsx')
            data_Basic_Parameters_correlation = data_Basic_Parameters_correlation.dropna()

            data_SRT_clu = data_Basic_Parameters_correlation.copy()
            if cluster != 'All':
                data_SRT_clu = data_SRT_clu[data_SRT_clu['Cluster'] == cluster]
                s = pd.Series(range(len(data_SRT_clu)))
                data_SRT_clu = data_SRT_clu.set_index(s)

            id_keep = []
            for i in np.unique(data_SRT_clu['Barcodes']):
                id_keep.extend(np.where(np.array(barcodes_filter_1) == i)[0])
            # id_filter = [i for i in range(len(final['Gene Name'])) if final['Gene Name'][i] in np.unique(final_filter['Gene Name'])]
            id_filter = [i for i in range(len(barcodes_filter_1)) if i not in id_keep]
            gene_expression_matrix_filter = gene_expression_matrix.drop(id_filter, axis=0)
            barcodes_filter_1_filter = np.asarray(barcodes_filter_1)[id_keep]  ##related barcodes
            parameters = []
            for bar in barcodes_filter_1_filter:
                parameters.append(list(data_SRT_clu[type_name])[list(data_SRT_clu['Barcodes']).index(bar)])

            parameters = [i / max(parameters) for i in parameters]
            gene_expression_matrix_filter.columns = list(remain_gene_name)  ##related gene
            gene_expression_matrix_filter = gene_expression_matrix_filter.loc[:,
                                            ~gene_expression_matrix_filter.columns.duplicated()]
            gene_expression_matrix_filter = gene_expression_matrix_filter.reindex(
                gene_expression_matrix_filter.mean().sort_values(ascending=False).index, axis=1)
            # gene_expression_matrix_filter.insert(0, type_choose_name , parameters)
            ###########################################
            gene_expression_matrix_filter = gene_expression_matrix_filter.apply(lambda x: x / abs(x).max())
            gene_expression_matrix_filter = gene_expression_matrix_filter.mul(parameters, axis='rows')

            ###########################################
            V_Matrix = gene_expression_matrix_filter.to_numpy()
            if rotate_matrix == True:
                V_Matrix = np.rot90(V_Matrix)
            model = NMF(n_components=k)
            W_Matrix = model.fit_transform(V_Matrix)
            H_Matrix = model.components_
            Error = model.reconstruction_err_
            K_list.append(k)
            Error_list.append(Error)
        a = {'K': K_list, 'Reconstruction Error': Error_list}
        df = pd.DataFrame.from_dict(a, orient='index').T
        colorMapTitle = type_choose_name + '_' + cluster + '_reconstruction_error'
        df.to_excel(desfilepath + colorMapTitle + ".xlsx", index=False)

    def SRT_nEphys_network_activity_features_NMF(self, type_choose_name=None, k=3, fliter_gene=True, cluster='All',gene_list=None, rotate_matrix=True, plot=True, Given_gene_list=True):
        """
        Application of non-negative matrix factorization (NMF) to identify individual spatiotemporal patterns emerging from SRT and n‐Ephys networks. Input factors (k) are obtained from reconstruction error.

            File input needed:
            -------
                - related files
                - 'SRT_nEphys_Multiscale_Coordinates_for_network_activity_features.xlsx'
                - 'gene_list_all.xlsx'

            Parameters
            -------

            Returns
            -------

            File output:
            -------
                - 'SRT_nEphys_Multiscale_Coordinates_for_network_activity_features_sorted.xlsx'
                - 'V_Matrix_[network_activity_feature]_[cluster].npy'
                - 'V_Matrix_[network_activity_feature]_[cluster].png'
                - 'V_Matrix_x_y_labels_[network_activity_feature]_[cluster].txt'
                - 'W_Matrix_[network_activity_feature]_[cluster].npy'
                - 'W_Matrix_[network_activity_feature]_[cluster].png'
                - 'H_Matrix_[network_activity_feature]_[cluster].npy'
                - 'H_Matrix_[network_activity_feature]_[cluster].png'
                - 'W_Matrix_[network_activity_feature]_[cluster]_[factor].png'
        """

        path = self.srcfilepath[:self.srcfilepath.rfind('/')]
        desfilepath = path + '/NMF/'
        if not os.path.exists(desfilepath):
            os.mkdir(desfilepath)
        if os.path.exists(
                desfilepath + 'V_Matrix1_' + type_choose_name + '_' + cluster + '.npy') and os.path.exists(
            desfilepath + 'W_Matrix_' + type_choose_name + '_' + cluster + '.npy') and os.path.exists(
            desfilepath + 'H_Matrix_' + type_choose_name + '_' + cluster + '.npy'):
            V_Matrix = np.load(desfilepath + 'V_Matrix_' + type_choose_name + '_' + cluster + '.npy',
                               allow_pickle=True)
            W_Matrix = np.load(desfilepath + 'W_Matrix_' + type_choose_name + '_' + cluster + '.npy',
                               allow_pickle=True)
            H_Matrix = np.load(desfilepath + 'H_Matrix_' + type_choose_name + '_' + cluster + '.npy',
                               allow_pickle=True)
        else:
            type = type_choose_name
            type_name = type + ' nEphys'
            # Read related information
            csv_file, tissue_lowres_scalef, features_name, matr_raw, barcodes, img, csv_file_cluster = self.read_related_files()
            barcode_cluster = np.asarray(csv_file_cluster["Barcode"])
            group = np.asarray(csv_file["selection"])
            barcode_CSV = np.asarray(csv_file["barcode"])
            g = 1
            ix = np.where(group == g)
            ########################################### Filters:
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
            # delete the nodes with less then 1000 gene count
            deleted_notes = [i for i in range(len(genes_per_cell)) if genes_per_cell[i] <= 1000]
            # delete the nodes not in clusters
            deleted_notes_cluster = [i for i in range(len(genes_per_cell)) if
                                     str(barcodes[i])[2:-1] not in barcode_cluster]
            deleted_notes.extend(deleted_notes_cluster)
            deleted_notes = list(np.unique(deleted_notes))
            matr = np.delete(matr, deleted_notes, axis=1)
            barcodes_filter = [barcodes[i] for i in range(len(barcodes)) if i not in deleted_notes]
            # new_id = [j for i in barcode_CSV for j in range(len(barcodes)) if str(barcodes[j])[2:-1] == i]
            adata = AnnData(np.array(matr))
            sc.pp.normalize_total(adata, inplace=True)
            gene_expression = adata.X
            genes_expression_count = np.asarray((matr > 0).sum(axis=1)).squeeze()
            df_gene_count = pd.DataFrame({'Gene Name': gene_name, 'Gene Count sorted': list(genes_expression_count)})
            gene_expression_series = np.asarray(gene_expression)
            if Given_gene_list == True:
                gene_add = gene_list_choose
                filter_gene_id = [i for gene in gene_add for i in range(len(gene_name)) if gene_name[i].lower() == gene]
                gene_expression_series = gene_expression_series[filter_gene_id]
                gene_expression_series = np.asarray(gene_expression_series).T
                filter_gene_expression_matrix = gene_expression_series
                remain_gene_name = np.asarray(df_gene_count['Gene Name'])
                remain_gene_name = remain_gene_name[filter_gene_id]
            else:
                if gene_list != None:
                    select_genes_add = pd.read_excel(self.srcfilepath + 'gene_list_all.xlsx')
                    gene_add = [str(i).lower() for i in select_genes_add[gene_list] if str(i) != 'nan']
                    filter_gene_id = [i for gene in gene_add for i in range(len(gene_name)) if
                                      gene_name[i].lower() == gene]
                    # id_no_expression = [i for i in range(len(gene_expression_series)) if sum(gene_expression_series[i]) == 0]
                    gene_expression_series = gene_expression_series[filter_gene_id]
                    gene_expression_series = np.asarray(gene_expression_series).T
                    filter_gene_expression_matrix = gene_expression_series
                    remain_gene_name = np.asarray(df_gene_count['Gene Name'])
                    remain_gene_name = remain_gene_name[filter_gene_id]
                else:
                    if fliter_gene == True:
                        gene_expression_list = list(gene_expression_series[gene_expression_series > 0])
                        mean_B, low_threshold_B, high_threshold_B = self.quantile_bound(gene_expression_list)
                        gene_expression_series[gene_expression_series >= high_threshold_B] = 0
                        mean_expression_nodes = np.mean(
                            [int(np.asarray((np.asarray(gene_expression_series[i]) != 0).sum())) for i in
                             range(len(gene_expression_series))])
                        std_expression_nodes = np.std(
                            [int(np.asarray((np.asarray(gene_expression_series[i]) != 0).sum())) for i in
                             range(len(gene_expression_series))])
                        id_no_expression = [i for i in range(len(gene_expression_series)) if
                                            sum(gene_expression_series[i]) == 0 or int(np.asarray((np.asarray(
                                                gene_expression_series[
                                                    i]) != 0).sum())) <= mean_expression_nodes + std_expression_nodes]
                        gene_expression_series = np.delete(gene_expression_series, id_no_expression, axis=0)
                        gene_expression_series = np.asarray(gene_expression_series).T
                        expression_each_gene_mean = np.asarray(gene_expression_series).mean(axis=0)
                        expression_each_gene_std = np.asarray(gene_expression_series).std(axis=0)
                        index_mean = \
                            np.where(np.array(expression_each_gene_mean) >= np.asarray(gene_expression_series).mean())[
                                0]
                        index_std = \
                            np.where(np.array(expression_each_gene_std) >= np.asarray(gene_expression_series).std())[0]
                        keep_gene_id = set(index_mean) & set(index_std)
                        filter_gene_expression_matrix = gene_expression_series[:, list(keep_gene_id)]
                        remain_gene_name = np.asarray(df_gene_count['Gene Name'])[list(keep_gene_id)]
                    else:
                        id_no_expression = [i for i in range(len(gene_expression_series)) if
                                            sum(gene_expression_series[i]) == 0]
                        gene_expression_series = np.delete(gene_expression_series, id_no_expression, axis=0)
                        gene_expression_series = np.asarray(gene_expression_series).T
                        filter_gene_expression_matrix = gene_expression_series
                        remain_gene_name = np.asarray(df_gene_count['Gene Name'])
                        remain_gene_name = list(
                            [remain_gene_name[i] for i in range(len(remain_gene_name)) if i not in id_no_expression])

            gene_expression_matrix = pd.DataFrame(filter_gene_expression_matrix)
            barcodes_filter_1 = [str(barcodes_filter[i])[2:-1] for i in range(len(barcodes_filter))]

            ########################################### Create NPY Files
            data_Basic_Parameters_correlation = pd.read_excel(
                self.srcfilepath + 'SRT_nEphys_Multiscale_Coordinates_for_network_activity_features.xlsx')
            data_Basic_Parameters_correlation = data_Basic_Parameters_correlation.dropna()
            sorter = self.clusters  ##### sort based on clusters
            # Create the dictionary that defines the order for sorting
            sorterIndex = dict(zip(sorter, range(len(sorter))))
            data_Basic_Parameters_correlation['Cluster_Rank'] = data_Basic_Parameters_correlation['Cluster'].map(
                sorterIndex)
            data_Basic_Parameters_correlation.sort_values(['Cluster_Rank'], ascending=[True], inplace=True)
            data_Basic_Parameters_correlation.drop('Cluster_Rank', 1, inplace=True)
            data_Basic_Parameters_correlation.to_excel(
                (self.srcfilepath + 'SRT_nEphys_Multiscale_Coordinates_for_network_activity_features_sorted.xlsx'),
                index=False)

            data_SRT_clu = data_Basic_Parameters_correlation.copy()
            if cluster != 'All':
                data_SRT_clu = data_SRT_clu[data_SRT_clu['Cluster'] == cluster]
                s = pd.Series(range(len(data_SRT_clu)))
                data_SRT_clu = data_SRT_clu.set_index(s)
            id_keep = []
            for i in np.unique(data_SRT_clu['Barcodes']):
                id_keep.extend(np.where(np.array(barcodes_filter_1) == i)[0])
            # id_filter = [i for i in range(len(final['Gene Name'])) if final['Gene Name'][i] in np.unique(final_filter['Gene Name'])]
            id_filter = [i for i in range(len(barcodes_filter_1)) if i not in id_keep]
            gene_expression_matrix_filter = gene_expression_matrix.drop(id_filter, axis=0)
            barcodes_filter_1_filter = np.asarray(barcodes_filter_1)[id_keep]  ##related barcodes
            parameters = []
            for bar in barcodes_filter_1_filter:
                parameters.append(list(data_SRT_clu[type_name])[list(data_SRT_clu['Barcodes']).index(bar)])

            parameters = [i / max(parameters) for i in parameters]
            gene_expression_matrix_filter.columns = list(remain_gene_name)  ##related gene
            gene_expression_matrix_filter = gene_expression_matrix_filter.loc[:,
                                            ~gene_expression_matrix_filter.columns.duplicated()]
            gene_expression_matrix_filter = gene_expression_matrix_filter.reindex(
                gene_expression_matrix_filter.mean().sort_values(ascending=False).index, axis=1)
            # gene_expression_matrix_filter.insert(0, type_choose_name , parameters)
            gene_expression_matrix_filter = gene_expression_matrix_filter.apply(lambda x: x / abs(x).max())
            gene_expression_matrix_filter = gene_expression_matrix_filter.mul(parameters, axis='rows')
            V_Matrix = gene_expression_matrix_filter.to_numpy()
            if rotate_matrix == True:
                V_Matrix = np.rot90(V_Matrix)
            np.save(desfilepath + 'V_Matrix_' + type_choose_name + '_' + cluster, V_Matrix)
            model = NMF(n_components=k)
            W_Matrix = model.fit_transform(V_Matrix)
            np.save(desfilepath + 'W_Matrix_' + type_choose_name + '_' + cluster, W_Matrix)
            H_Matrix = model.components_
            np.save(desfilepath + 'H_Matrix_' + type_choose_name + '_' + cluster, H_Matrix)
            file_path = desfilepath + 'V_Matrix_x_y_labels_' + type_choose_name + '_' + cluster + ".txt"
            f = open(file_path, "a")
            f.seek(0)
            f.truncate()
            if rotate_matrix == True:
                msg = 'y_labels: Genes_name'
            else:
                msg = 'x_labels: Genes_name'
            f.write(msg + '\n')
            if rotate_matrix == True:
                msg = ' '.join(str(e) for e in list(gene_expression_matrix_filter.columns)[::-1])
            else:
                msg = ' '.join(str(e) for e in list(gene_expression_matrix_filter.columns))
            f.write(msg + '\n')
            if rotate_matrix == True:
                msg = 'x_labels: Barcodes'
            else:
                msg = 'y_labels: Barcodes'
            f.write(msg + '\n')
            msg = ' '.join(str(e) for e in list(barcodes_filter_1_filter))
            f.write(msg + '\n')
            f.close()

        ########################################### Plots
        gene_order = select_genes
        indices = []
        for gene in gene_order:
            indices.append(list(gene_expression_matrix_filter.columns).index(gene))
        # indices = list(sorted(range(len(H_Matrix[i])), key=lambda k: H_Matrix[i][k], reverse=True))
        gene_sort = np.asarray(gene_expression_matrix_filter.columns)[indices]
        value_sort = np.asarray(V_Matrix)[:, indices]
        fig, ax = plt.subplots()
        jet = plt.get_cmap('Greys')
        pt = ax.imshow(value_sort, cmap=jet, interpolation='None', aspect='auto')
        ##### first show the colorbar with values and select the min ,max values
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.2)
        cbar = fig.colorbar(pt, shrink=.1, cax=cax, orientation="vertical")
        ##### second based on the selected the min ,max values, set the max and min for
        # ax.set_clim(vmin=0,vmax=0.5)
        if rotate_matrix == True:
            ax.set_xlabel('# Spots')
            ax.set_ylabel('')
        else:
            ax.set_ylabel('# Spots')
            ax.set_xlabel('')
        x = [i for i in range(V_Matrix.shape[1])]
        ax.set_xticks(x)
        ax.set_xticklabels(gene_sort)
        ax.tick_params(axis="x", labelsize=8, labelrotation=30)
        labels = ax.get_xticklabels()
        [label.set_fontweight('bold') for label in labels]
        colorMapTitle = 'V_Matrix_' + type_choose_name + '_' + cluster
        fig.savefig(desfilepath + colorMapTitle + ".png", format='png', dpi=600)
        plt.close()
        #####################
        fig, ax = plt.subplots()
        ax.imshow(W_Matrix, cmap=jet, interpolation='None', aspect='auto')
        if rotate_matrix == True:
            ax.set_xlabel('# Genes')
            ax.set_ylabel('# factors')
        else:
            ax.set_ylabel('# Spots')
            ax.set_xlabel('# factors')
        ax.set_xticks(range(0, k))
        labels = [str(i + 1) for i in range(0, k)]
        ax.set_xticklabels(labels)
        colorMapTitle = 'W_Matrix_' + type_choose_name + '_' + cluster
        fig.savefig(desfilepath + colorMapTitle + ".png", format='png', dpi=600)
        plt.close()
        #####################
        fig, ax = plt.subplots(nrows=1, ncols=len(H_Matrix), figsize=(40, 10))  # , facecolor='None'
        for i in range(len(H_Matrix)):
            x = [i for i in range(len(H_Matrix[i]))]
            indices = list(sorted(range(len(H_Matrix[i])), key=lambda k: H_Matrix[i][k], reverse=True))
            gene_expression_matrix_filter = select_genes
            gene_sort = np.asarray(gene_expression_matrix_filter)[indices]
            # gene_sort = np.asarray(gene_expression_matrix_filter.columns)[indices]
            value_sort = np.asarray(H_Matrix[i])[indices]
            reference_list = [select_genes.index(l) for l in gene_sort]
            gene_sort = np.asarray([x for _, x in sorted(zip(reference_list, list(gene_sort)))])
            value_sort = np.asarray([x for _, x in sorted(zip(reference_list, list(value_sort)))])
            print('gene_sort', gene_sort)
            from scipy.interpolate import interp1d
            average_freq_smooth = np.linspace(min(x), max(x), 50)
            Cxy_smooth = interp1d(x, value_sort, kind='cubic', bounds_error=True)
            wave_smooth = Cxy_smooth(average_freq_smooth)
            wave_smooth = [0 if k < 0 else k for k in wave_smooth]
            wave_smooth = [2.5 if k > 2.5 else k for k in wave_smooth]
            ax[i].plot(average_freq_smooth, wave_smooth, linewidth=2, label='Factor ' + str(i + 1))
            ax[i].set_xticks(x)
            ax[i].set_xticklabels(gene_sort)
            ax[i].tick_params(axis="x", labelsize=8, labelrotation=30)
            labels = ax[i].get_xticklabels()
            [label.set_fontweight('bold') for label in labels]
            # ax.plot(x, H_Matrix[i], linewidth=0.5, label='Factor ' + str(i + 1))
            ax[i].spines['top'].set_visible(False)
            ax[i].spines['right'].set_visible(False)
            ax[i].set_ylabel('Inferrence (a.u.)')
            ax[i].set_xlabel('')
            ax[i].legend(loc='best', fontsize='small')
            ax[i].set_ylim(-0.5, 2.5)  ##### set y axis limits
        colorMapTitle = 'H_Matrix_' + type_choose_name + '_' + cluster
        fig.tight_layout()
        fig.savefig(desfilepath + colorMapTitle + ".png", format='png', dpi=600)
        plt.close()

        if plot == True:
            #############################################
            # find the clusters
            csv_file_cluster_name = 'Loupe Clusters.csv'
            csv_file_cluster_file, csv_file_cluster_Root = self.get_filename_path(self.srcfilepath,
                                                                                  csv_file_cluster_name)
            for i in range(len(csv_file_cluster_file)):
                if csv_file_cluster_file[i][0] != '.':
                    csv_file_cluster_root = csv_file_cluster_Root[i] + '/' + csv_file_cluster_file[i]
            #############################################
            csv_file_cluster = pd.read_csv(csv_file_cluster_root)
            barcode_cluster = np.asarray(csv_file_cluster["Barcode"])
            nEphys_cluster = np.asarray(csv_file_cluster["Loupe Clusters"])

            column_list_csv = ["barcode", "selection", "y", "x", "pixel_y", "pixel_x"]
            csv_file_name = 'tissue_positions_list.csv'
            csv_file, csv_Root = self.get_filename_path(self.srcfilepath, csv_file_name)
            for i in range(len(csv_file)):
                if csv_file[i][0] != '.':
                    csv_root = csv_Root[i] + '/' + csv_file[i]

            csv_file = pd.read_csv(csv_root, names=column_list_csv)
            csv_file.to_excel(self.srcfilepath + "tissue_positions_list.xlsx", index=False)

            scatter_x = np.asarray(csv_file["pixel_x"])
            scatter_y = np.asarray(csv_file["pixel_y"])
            barcode_raw = np.asarray(csv_file["barcode"])

            group = np.asarray(csv_file["selection"])

            mask_id = [i for i in range(len(group)) if group[i] == 1]
            extent = [min([scatter_x[i] for i in mask_id]), max([scatter_x[i] for i in mask_id]),
                      min([scatter_y[i] for i in mask_id]), max([scatter_y[i] for i in mask_id])]
            g = 1
            ix = np.where(group == g)

            Cluster_list = self.clusters
            color_map = []

            for i in ix[0]:
                bar_code = csv_file['barcode'][i]
                try:
                    clu = nEphys_cluster[list(barcode_cluster).index(bar_code)]
                    for j in range(len(Cluster_list)):
                        if Cluster_list[j] == clu:
                            color_map.append(j)
                except:
                    color_map.append(len(Cluster_list))

            x_norm = [i for i in scatter_x[ix] - extent[0]]
            y_norm = [i for i in scatter_y[ix] - extent[2]]
            barcode_norm = [i for i in barcode_raw[ix]]

            color_map_filter_id = [i for i in range(len(color_map)) if color_map[i] != len(self.clusters)]
            x_norm = np.asarray([x_norm[i] * Pixel_SRT_mm for i in color_map_filter_id])
            y_norm = np.asarray([y_norm[i] * Pixel_SRT_mm for i in color_map_filter_id])
            x_norm_raw = x_norm
            y_norm_raw = y_norm
            barcode_norm = np.asarray([barcode_norm[i] for i in color_map_filter_id])
            filter_channel = [np.where(np.asarray(barcode_norm) == bar)[0] for bar in barcodes_filter_1_filter if
                              bar in barcode_norm]
            # for bar in barcodes_filter_1_filter:
            #     if bar in barcode_norm:
            #         np.where(np.asarray(barcode_norm) == bar)[0]
            x_norm = list(x_norm[filter_channel])
            y_norm = list(y_norm[filter_channel])
            # barcode_norm = barcode_norm[filter_channel]
            W_Matrix = W_Matrix.T
            for i in range(k):
                color_map = [node * 20 for node in list(W_Matrix[i])]  ##### change for size of nodes
                fig, ax = plt.subplots()  #####figsize=(8, 10)
                ax.scatter(x_norm, y_norm, c='black', marker='o', linewidth=1, s=color_map, alpha=1, edgecolors='black')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                x_ticks_raw = np.linspace(min(x_norm_raw), max(x_norm_raw), 5, endpoint=True)
                x_ticks = [str(round(k - min(x_ticks_raw), 2)) for k in x_ticks_raw]
                ax.set_xticks(x_ticks_raw)
                ax.set_xticklabels(x_ticks)
                ax.set_xlabel('Length(mm)', fontsize=8)
                y_ticks_raw = np.linspace(min(y_norm_raw), max(y_norm_raw), 5, endpoint=True)
                y_ticks = [str(round(k - min(y_ticks_raw), 2)) for k in y_ticks_raw]
                ax.set_yticks(y_ticks_raw)
                ax.set_yticklabels(y_ticks)
                ax.set_ylabel('Length(mm)', fontsize=8)
                ax.set_title('Factor ' + str(i + 1) + '-' + type_choose_name, fontsize=8)
                ax.set_aspect('equal', 'box')
                colorMapTitle_SRT = 'W_Matrix_' + type_choose_name + '_' + cluster + '_' + 'Factor_' + str(i + 1)
                fig.savefig(desfilepath + colorMapTitle_SRT + ".png", format='png', dpi=600)
                plt.close()

    def SRT_nEphys_network_topological_metrics_NMF_reconstruction_error(self, type_choose_name=None, min_k=1, max_k=100,fliter_gene=True, cluster='All', gene_list=None,rotate_matrix=True, plot=False,Given_gene_list=True):
        """
        Determine optimal number of factors for NMF decomposition.

            File input needed:
            -------
                - related files
                - 'SRT_nEphys_Multiscale_Coordinates_for_network_topological_metrics.xlsx'
                - 'gene_list_all.xlsx'

            Parameters
            -------

            Returns
            -------

            File output:
            -------
                - [network_topological_metric]_[cluster]_reconstruction_error.xlsx'
        """

        path = self.srcfilepath[:self.srcfilepath.rfind('/')]
        desfilepath = path + '/NMF/'
        if not os.path.exists(desfilepath):
            os.mkdir(desfilepath)
        K_list = []
        Error_list = []
        for k in range(min_k, max_k + 1):
            k = int(k)
            type = type_choose_name

            type_name = type + ' nEphys'

            # Read related information
            csv_file, tissue_lowres_scalef, features_name, matr_raw, barcodes, img, csv_file_cluster = self.read_related_files()
            barcode_cluster = np.asarray(csv_file_cluster["Barcode"])
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
            deleted_notes_cluster = [i for i in range(len(genes_per_cell)) if
                                     str(barcodes[i])[2:-1] not in barcode_cluster]
            deleted_notes.extend(deleted_notes_cluster)
            deleted_notes = list(np.unique(deleted_notes))
            # ##########################################################
            matr = np.delete(matr, deleted_notes, axis=1)

            barcodes_filter = [barcodes[i] for i in range(len(barcodes)) if i not in deleted_notes]
            # new_id = [j for i in barcode_CSV for j in range(len(barcodes)) if str(barcodes[j])[2:-1] == i]

            adata = AnnData(np.array(matr))
            sc.pp.normalize_total(adata, inplace=True)
            gene_expression = adata.X
            genes_expression_count = np.asarray((matr > 0).sum(axis=1)).squeeze()
            ###################################Choose gene 1 way
            df_gene_count = pd.DataFrame({'Gene Name': gene_name, 'Gene Count sorted': list(genes_expression_count)})
            gene_expression_series = np.asarray(gene_expression)
            if Given_gene_list == True:
                gene_add = gene_list_choose
                filter_gene_id = [i for gene in gene_add for i in range(len(gene_name)) if gene_name[i].lower() == gene]
                gene_expression_series = gene_expression_series[filter_gene_id]
                gene_expression_series = np.asarray(gene_expression_series).T
                filter_gene_expression_matrix = gene_expression_series
                remain_gene_name = np.asarray(df_gene_count['Gene Name'])
                remain_gene_name = remain_gene_name[filter_gene_id]
            else:
                if gene_list != None:
                    select_genes_add = pd.read_excel(self.srcfilepath + 'gene_list_all.xlsx')
                    gene_add = [str(i).lower() for i in select_genes_add[gene_list] if str(i) != 'nan']
                    filter_gene_id = [i for gene in gene_add for i in range(len(gene_name)) if
                                      gene_name[i].lower() == gene]
                    # id_no_expression = [i for i in range(len(gene_expression_series)) if sum(gene_expression_series[i]) == 0]
                    gene_expression_series = gene_expression_series[filter_gene_id]
                    gene_expression_series = np.asarray(gene_expression_series).T
                    filter_gene_expression_matrix = gene_expression_series
                    remain_gene_name = np.asarray(df_gene_count['Gene Name'])
                    remain_gene_name = remain_gene_name[filter_gene_id]
                else:
                    if fliter_gene == True:
                        gene_expression_list = list(gene_expression_series[gene_expression_series > 0])
                        mean_B, low_threshold_B, high_threshold_B = self.quantile_bound(gene_expression_list)
                        gene_expression_series[gene_expression_series >= high_threshold_B] = 0
                        mean_expression_nodes = np.mean(
                            [int(np.asarray((np.asarray(gene_expression_series[i]) != 0).sum())) for i in
                             range(len(gene_expression_series))])
                        std_expression_nodes = np.std(
                            [int(np.asarray((np.asarray(gene_expression_series[i]) != 0).sum())) for i in
                             range(len(gene_expression_series))])
                        # ##############################################################
                        id_no_expression = [i for i in range(len(gene_expression_series)) if
                                            sum(gene_expression_series[i]) == 0 or int(np.asarray((np.asarray(
                                                gene_expression_series[
                                                    i]) != 0).sum())) <= mean_expression_nodes + std_expression_nodes]

                        gene_expression_series = np.delete(gene_expression_series, id_no_expression, axis=0)
                        gene_expression_series = np.asarray(gene_expression_series).T
                        expression_each_gene_mean = np.asarray(gene_expression_series).mean(axis=0)
                        expression_each_gene_std = np.asarray(gene_expression_series).std(axis=0)
                        index_mean = \
                            np.where(np.array(expression_each_gene_mean) >= np.asarray(gene_expression_series).mean())[
                                0]
                        index_std = \
                            np.where(np.array(expression_each_gene_std) >= np.asarray(gene_expression_series).std())[0]
                        keep_gene_id = set(index_mean) & set(index_std)
                        filter_gene_expression_matrix = gene_expression_series[:, list(keep_gene_id)]
                        # print('remain_gene_name', pd.DataFrame(filter_gene_expression_matrix))
                        remain_gene_name = np.asarray(df_gene_count['Gene Name'])[list(keep_gene_id)]
                        # print('remain_gene_name', remain_gene_name)
                    #
                    else:
                        id_no_expression = [i for i in range(len(gene_expression_series)) if
                                            sum(gene_expression_series[i]) == 0]
                        gene_expression_series = np.delete(gene_expression_series, id_no_expression, axis=0)
                        gene_expression_series = np.asarray(gene_expression_series).T
                        filter_gene_expression_matrix = gene_expression_series
                        remain_gene_name = np.asarray(df_gene_count['Gene Name'])
                        remain_gene_name = list(
                            [remain_gene_name[i] for i in range(len(remain_gene_name)) if i not in id_no_expression])

            gene_expression_matrix = pd.DataFrame(filter_gene_expression_matrix)
            barcodes_filter_1 = [str(barcodes_filter[i])[2:-1] for i in range(len(barcodes_filter))]

            ########################
            data_Basic_Parameters_correlation = pd.read_excel(
                self.srcfilepath + 'SRT_nEphys_Multiscale_Coordinates_for_network_topological_metrics.xlsx')
            data_Basic_Parameters_correlation = data_Basic_Parameters_correlation.dropna()

            data_SRT_clu = data_Basic_Parameters_correlation.copy()
            if cluster != 'All':
                data_SRT_clu = data_SRT_clu[data_SRT_clu['Cluster'] == cluster]
                s = pd.Series(range(len(data_SRT_clu)))
                data_SRT_clu = data_SRT_clu.set_index(s)

            id_keep = []
            for i in np.unique(data_SRT_clu['Barcodes']):
                id_keep.extend(np.where(np.array(barcodes_filter_1) == i)[0])
            # id_filter = [i for i in range(len(final['Gene Name'])) if final['Gene Name'][i] in np.unique(final_filter['Gene Name'])]
            id_filter = [i for i in range(len(barcodes_filter_1)) if i not in id_keep]
            gene_expression_matrix_filter = gene_expression_matrix.drop(id_filter, axis=0)
            barcodes_filter_1_filter = np.asarray(barcodes_filter_1)[id_keep]  ##related barcodes
            parameters = []
            for bar in barcodes_filter_1_filter:
                parameters.append(list(data_SRT_clu[type_name])[list(data_SRT_clu['Barcodes']).index(bar)])

            parameters = [i / max(parameters) for i in parameters]
            gene_expression_matrix_filter.columns = list(remain_gene_name)  ##related gene
            gene_expression_matrix_filter = gene_expression_matrix_filter.loc[:,
                                            ~gene_expression_matrix_filter.columns.duplicated()]
            gene_expression_matrix_filter = gene_expression_matrix_filter.reindex(
                gene_expression_matrix_filter.mean().sort_values(ascending=False).index, axis=1)
            # gene_expression_matrix_filter.insert(0, type_choose_name , parameters)
            ###########################################
            gene_expression_matrix_filter = gene_expression_matrix_filter.apply(lambda x: x / abs(x).max())
            gene_expression_matrix_filter = gene_expression_matrix_filter.mul(parameters, axis='rows')

            ###########################################
            V_Matrix = gene_expression_matrix_filter.to_numpy()
            if rotate_matrix == True:
                V_Matrix = np.rot90(V_Matrix)
            model = NMF(n_components=k)
            W_Matrix = model.fit_transform(V_Matrix)
            H_Matrix = model.components_
            Error = model.reconstruction_err_
            K_list.append(k)
            Error_list.append(Error)
        a = {'K': K_list, 'Reconstruction Error': Error_list}
        df = pd.DataFrame.from_dict(a, orient='index').T
        colorMapTitle = type_choose_name + '_' + cluster + '_reconstruction_error'
        df.to_excel(desfilepath + colorMapTitle + ".xlsx", index=False)

    def SRT_nEphys_network_topological_metrics_NMF(self, type_choose_name=None, k=3, fliter_gene=True, cluster='All',gene_list=None, rotate_matrix=True, plot=True, Given_gene_list=True):
        """
        Application of non-negative matrix factorization (NMF) to identify individual spatiotemporal patterns emerging from SRT and n‐Ephys networks. Input factors (k) are obtained from reconstruction error.

            File input needed:
            -------
                - related files
                - 'SRT_nEphys_Multiscale_Coordinates_for_network_topological_metrics.xlsx'
                - 'gene_list_all.xlsx'

            Parameters
            -------

            Returns
            -------

            File output:
            -------
                - 'SRT_nEphys_Multiscale_Coordinates_for_network_activity_features_sorted.xlsx'
                - 'V_Matrix_[network_topological_metric]_[cluster].npy'
                - 'V_Matrix_[network_topological_metric]_[cluster].png'
                - 'V_Matrix_x_y_labels_[network_topological_metric]_[cluster].txt'
                - 'W_Matrix_[network_topological_metric]_[cluster].npy'
                - 'W_Matrix_[network_topological_metric]_[cluster].png'
                - 'H_Matrix_[network_topological_metric]_[cluster].npy'
                - 'H_Matrix_[network_topological_metric]_[cluster].png'
                - 'W_Matrix_[network_topological_metric]_[cluster]_[factor].png'
        """

        path = self.srcfilepath[:self.srcfilepath.rfind('/')]
        desfilepath = path + '/NMF/'
        if not os.path.exists(desfilepath):
            os.mkdir(desfilepath)
        if os.path.exists(desfilepath + '1V_Matrix_' + type_choose_name + '_' + cluster + '.npy') and os.path.exists(
                desfilepath + 'W_Matrix_' + type_choose_name + '_' + cluster + '.npy') and os.path.exists(
            desfilepath + 'H_Matrix_' + type_choose_name + '_' + cluster + '.npy'):
            V_Matrix = np.load(desfilepath + 'V_Matrix_' + type_choose_name + '_' + cluster + '.npy',
                               allow_pickle=True)
            W_Matrix = np.load(desfilepath + 'W_Matrix_' + type_choose_name + '_' + cluster + '.npy',
                               allow_pickle=True)
            H_Matrix = np.load(desfilepath + 'H_Matrix_' + type_choose_name + '_' + cluster + '.npy',
                               allow_pickle=True)
        else:
            type = type_choose_name
            type_name = type + ' nEphys'
            # Read related information
            csv_file, tissue_lowres_scalef, features_name, matr_raw, barcodes, img, csv_file_cluster = self.read_related_files()
            barcode_cluster = np.asarray(csv_file_cluster["Barcode"])
            group = np.asarray(csv_file["selection"])
            barcode_CSV = np.asarray(csv_file["barcode"])
            g = 1
            ix = np.where(group == g)
            ########################################### Filters:
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
            # delete the nodes with less then 1000 gene count
            deleted_notes = [i for i in range(len(genes_per_cell)) if genes_per_cell[i] <= 1000]
            # delete the nodes not in clusters
            deleted_notes_cluster = [i for i in range(len(genes_per_cell)) if
                                     str(barcodes[i])[2:-1] not in barcode_cluster]
            deleted_notes.extend(deleted_notes_cluster)
            deleted_notes = list(np.unique(deleted_notes))
            # ##########################################################
            matr = np.delete(matr, deleted_notes, axis=1)
            barcodes_filter = [barcodes[i] for i in range(len(barcodes)) if i not in deleted_notes]
            # new_id = [j for i in barcode_CSV for j in range(len(barcodes)) if str(barcodes[j])[2:-1] == i]
            adata = AnnData(np.array(matr))
            sc.pp.normalize_total(adata, inplace=True)
            gene_expression = adata.X
            genes_expression_count = np.asarray((matr > 0).sum(axis=1)).squeeze()
            df_gene_count = pd.DataFrame({'Gene Name': gene_name, 'Gene Count sorted': list(genes_expression_count)})
            gene_expression_series = np.asarray(gene_expression)
            if Given_gene_list == True:
                gene_add = gene_list_choose
                filter_gene_id = [i for gene in gene_add for i in range(len(gene_name)) if gene_name[i].lower() == gene]
                gene_expression_series = gene_expression_series[filter_gene_id]
                gene_expression_series = np.asarray(gene_expression_series).T
                filter_gene_expression_matrix = gene_expression_series
                remain_gene_name = np.asarray(df_gene_count['Gene Name'])
                remain_gene_name = remain_gene_name[filter_gene_id]
            else:
                if gene_list != None:
                    select_genes_add = pd.read_excel(self.srcfilepath + 'gene_list_all.xlsx')
                    gene_add = [str(i).lower() for i in select_genes_add[gene_list] if str(i) != 'nan']
                    filter_gene_id = [i for gene in gene_add for i in range(len(gene_name)) if
                                      gene_name[i].lower() == gene]
                    # id_no_expression = [i for i in range(len(gene_expression_series)) if sum(gene_expression_series[i]) == 0]
                    gene_expression_series = gene_expression_series[filter_gene_id]
                    gene_expression_series = np.asarray(gene_expression_series).T
                    filter_gene_expression_matrix = gene_expression_series
                    remain_gene_name = np.asarray(df_gene_count['Gene Name'])
                    remain_gene_name = remain_gene_name[filter_gene_id]
                else:
                    if fliter_gene == True:
                        gene_expression_list = list(gene_expression_series[gene_expression_series > 0])
                        mean_B, low_threshold_B, high_threshold_B = self.quantile_bound(gene_expression_list)
                        gene_expression_series[gene_expression_series >= high_threshold_B] = 0
                        mean_expression_nodes = np.mean(
                            [int(np.asarray((np.asarray(gene_expression_series[i]) != 0).sum())) for i in
                             range(len(gene_expression_series))])
                        std_expression_nodes = np.std(
                            [int(np.asarray((np.asarray(gene_expression_series[i]) != 0).sum())) for i in
                             range(len(gene_expression_series))])
                        id_no_expression = [i for i in range(len(gene_expression_series)) if
                                            sum(gene_expression_series[i]) == 0 or int(np.asarray((np.asarray(
                                                gene_expression_series[
                                                    i]) != 0).sum())) <= mean_expression_nodes + std_expression_nodes]
                        gene_expression_series = np.delete(gene_expression_series, id_no_expression, axis=0)
                        gene_expression_series = np.asarray(gene_expression_series).T
                        expression_each_gene_mean = np.asarray(gene_expression_series).mean(axis=0)
                        expression_each_gene_std = np.asarray(gene_expression_series).std(axis=0)
                        index_mean = \
                            np.where(np.array(expression_each_gene_mean) >= np.asarray(gene_expression_series).mean())[
                                0]
                        index_std = \
                            np.where(np.array(expression_each_gene_std) >= np.asarray(gene_expression_series).std())[0]
                        keep_gene_id = set(index_mean) & set(index_std)
                        filter_gene_expression_matrix = gene_expression_series[:, list(keep_gene_id)]
                        remain_gene_name = np.asarray(df_gene_count['Gene Name'])[list(keep_gene_id)]
                    else:
                        id_no_expression = [i for i in range(len(gene_expression_series)) if
                                            sum(gene_expression_series[i]) == 0]
                        gene_expression_series = np.delete(gene_expression_series, id_no_expression, axis=0)
                        gene_expression_series = np.asarray(gene_expression_series).T
                        filter_gene_expression_matrix = gene_expression_series
                        remain_gene_name = np.asarray(df_gene_count['Gene Name'])
                        remain_gene_name = list(
                            [remain_gene_name[i] for i in range(len(remain_gene_name)) if i not in id_no_expression])

            gene_expression_matrix = pd.DataFrame(filter_gene_expression_matrix)
            barcodes_filter_1 = [str(barcodes_filter[i])[2:-1] for i in range(len(barcodes_filter))]

            ########################################### Create NPY Files
            data_Basic_Parameters_correlation = pd.read_excel(self.srcfilepath + 'SRT_nEphys_Multiscale_Coordinates_for_network_topological_metrics.xlsx')
            data_Basic_Parameters_correlation = data_Basic_Parameters_correlation.dropna()
            sorter = self.clusters  ##### sort based on clusters
            # Create the dictionary that defines the order for sorting
            sorterIndex = dict(zip(sorter, range(len(sorter))))
            data_Basic_Parameters_correlation['Cluster_Rank'] = data_Basic_Parameters_correlation['Cluster'].map(
                sorterIndex)
            data_Basic_Parameters_correlation.sort_values(['Cluster_Rank'], ascending=[True], inplace=True)
            data_Basic_Parameters_correlation.drop('Cluster_Rank', 1, inplace=True)
            data_Basic_Parameters_correlation.to_excel((self.srcfilepath + 'SRT_nEphys_Multiscale_Coordinates_for_network_topological_metrics_sorted.xlsx'),index=False)

            data_SRT_clu = data_Basic_Parameters_correlation.copy()
            if cluster != 'All':
                data_ST_clu = data_SRT_clu[data_SRT_clu['Cluster'] == cluster]
                s = pd.Series(range(len(data_SRT_clu)))
                data_SRT_clu = data_SRT_clu.set_index(s)
            id_keep = []
            for i in np.unique(data_SRT_clu['Barcodes']):
                id_keep.extend(np.where(np.array(barcodes_filter_1) == i)[0])
            # id_filter = [i for i in range(len(final['Gene Name'])) if final['Gene Name'][i] in np.unique(final_filter['Gene Name'])]
            id_filter = [i for i in range(len(barcodes_filter_1)) if i not in id_keep]
            gene_expression_matrix_filter = gene_expression_matrix.drop(id_filter, axis=0)
            barcodes_filter_1_filter = np.asarray(barcodes_filter_1)[id_keep]  ##related barcodes
            parameters = []
            for bar in barcodes_filter_1_filter:
                parameters.append(data_SRT_clu[type_name][list(data_SRT_clu['Barcodes']).index(bar)])

            parameters = [i / max(parameters) for i in parameters]
            gene_expression_matrix_filter.columns = list(remain_gene_name)  ##related gene
            gene_expression_matrix_filter = gene_expression_matrix_filter.loc[:,
                                            ~gene_expression_matrix_filter.columns.duplicated()]
            gene_expression_matrix_filter = gene_expression_matrix_filter.reindex(
                gene_expression_matrix_filter.mean().sort_values(ascending=False).index, axis=1)
            # gene_expression_matrix_filter.insert(0, type_choose_name , parameters)
            gene_expression_matrix_filter = gene_expression_matrix_filter.apply(lambda x: x / abs(x).max())
            gene_expression_matrix_filter = gene_expression_matrix_filter.mul(parameters, axis='rows')
            V_Matrix = gene_expression_matrix_filter.to_numpy()
            if rotate_matrix == True:
                V_Matrix = np.rot90(V_Matrix)
            np.save(desfilepath + 'V_Matrix_' + type_choose_name + '_' + cluster, V_Matrix)
            model = NMF(n_components=k)
            W_Matrix = model.fit_transform(V_Matrix)
            np.save(desfilepath + 'W_Matrix_' + type_choose_name + '_' + cluster, W_Matrix)
            H_Matrix = model.components_
            np.save(desfilepath + 'H_Matrix_' + type_choose_name + '_' + cluster, H_Matrix)
            file_path = desfilepath + 'V_Matrix_x_y_labels_' + type_choose_name + '_' + cluster + ".txt"
            f = open(file_path, "a")
            f.seek(0)
            f.truncate()
            if rotate_matrix == True:
                msg = 'y_labels: Genes_name'
            else:
                msg = 'x_labels: Genes_name'
            f.write(msg + '\n')
            if rotate_matrix == True:
                msg = ' '.join(str(e) for e in list(gene_expression_matrix_filter.columns)[::-1])
            else:
                msg = ' '.join(str(e) for e in list(gene_expression_matrix_filter.columns))
            f.write(msg + '\n')
            if rotate_matrix == True:
                msg = 'x_labels: Barcodes'
            else:
                msg = 'y_labels: Barcodes'
            f.write(msg + '\n')
            msg = ' '.join(str(e) for e in list(barcodes_filter_1_filter))
            f.write(msg + '\n')
            f.close()

        ########################################### Plots
        gene_order = select_genes
        indices = []
        for gene in gene_order:
            indices.append(list(gene_expression_matrix_filter.columns).index(gene))
        # indices = list(sorted(range(len(H_Matrix[i])), key=lambda k: H_Matrix[i][k], reverse=True))
        gene_sort = np.asarray(gene_expression_matrix_filter.columns)[indices]
        value_sort = np.asarray(V_Matrix)[:, indices]
        fig, ax = plt.subplots()
        jet = plt.get_cmap('Greys')
        pt = ax.imshow(value_sort, cmap=jet, interpolation='None', aspect='auto')
        ##### first show the colorbar with values and select the min ,max values
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.2)
        cbar = fig.colorbar(pt, shrink=.1, cax=cax, orientation="vertical")
        ##### second based on the selected the min ,max values, set the max and min for
        # ax.set_clim(vmin=0,vmax=0.5)
        if rotate_matrix == True:
            ax.set_xlabel('# Spots')
            ax.set_ylabel('')
        else:
            ax.set_ylabel('# Spots')
            ax.set_xlabel('')
        x = [i for i in range(V_Matrix.shape[1])]
        ax.set_xticks(x)
        ax.set_xticklabels(gene_sort)
        ax.tick_params(axis="x", labelsize=8, labelrotation=30)
        labels = ax.get_xticklabels()
        [label.set_fontweight('bold') for label in labels]
        colorMapTitle = 'V_Matrix_' + type_choose_name + '_' + cluster
        fig.savefig(desfilepath + colorMapTitle + ".png", format='png', dpi=600)
        plt.close()
        #####################
        fig, ax = plt.subplots()
        ax.imshow(W_Matrix, cmap=jet, interpolation='None', aspect='auto')
        if rotate_matrix == True:
            ax.set_xlabel('# Genes')
            ax.set_ylabel('# factors')
        else:
            ax.set_ylabel('# Spots')
            ax.set_xlabel('# factors')
        ax.set_xticks(range(0, k))
        labels = [str(i + 1) for i in range(0, k)]
        ax.set_xticklabels(labels)
        colorMapTitle = 'W_Matrix_' + type_choose_name + '_' + cluster
        fig.savefig(desfilepath + colorMapTitle + ".png", format='png', dpi=600)
        plt.close()
        #############################
        fig, ax = plt.subplots(nrows=1, ncols=len(H_Matrix), figsize=(40, 10))  # , facecolor='None'
        for i in range(len(H_Matrix)):
            x = [i for i in range(len(H_Matrix[i]))]
            indices = list(sorted(range(len(H_Matrix[i])), key=lambda k: H_Matrix[i][k], reverse=True))
            gene_expression_matrix_filter = select_genes
            gene_sort = np.asarray(gene_expression_matrix_filter)[indices]
            # gene_sort = np.asarray(gene_expression_matrix_filter.columns)[indices]
            value_sort = np.asarray(H_Matrix[i])[indices]
            reference_list = [select_genes.index(l) for l in gene_sort]
            gene_sort = np.asarray([x for _, x in sorted(zip(reference_list, list(gene_sort)))])
            value_sort = np.asarray([x for _, x in sorted(zip(reference_list, list(value_sort)))])
            print('gene_sort', gene_sort)
            from scipy.interpolate import interp1d
            average_freq_smooth = np.linspace(min(x), max(x), 50)
            Cxy_smooth = interp1d(x, value_sort, kind='cubic', bounds_error=True)
            wave_smooth = Cxy_smooth(average_freq_smooth)
            wave_smooth = [0 if k < 0 else k for k in wave_smooth]
            wave_smooth = [2.5 if k > 2.5 else k for k in wave_smooth]
            ax[i].plot(average_freq_smooth, wave_smooth, linewidth=2, label='Factor ' + str(i + 1))
            ax[i].set_xticks(x)
            ax[i].set_xticklabels(gene_sort)
            ax[i].tick_params(axis="x", labelsize=8, labelrotation=30)
            labels = ax[i].get_xticklabels()
            [label.set_fontweight('bold') for label in labels]
            # ax.plot(x, H_Matrix[i], linewidth=0.5, label='Factor ' + str(i + 1))
            ax[i].spines['top'].set_visible(False)
            ax[i].spines['right'].set_visible(False)
            ax[i].set_ylabel('Inferrence (a.u.)')
            ax[i].set_xlabel('')
            ax[i].legend(loc='best', fontsize='small')
            ax[i].set_ylim(-0.5, 2.5)
        colorMapTitle = 'H_Matrix_' + type_choose_name + '_' + cluster
        fig.tight_layout()
        fig.savefig(desfilepath + colorMapTitle + ".png", format='png', dpi=600)
        plt.close()

        if plot == True:
            #############################################
            # fine the clusters
            csv_file_cluster_name = 'Loupe Clusters.csv'
            csv_file_cluster_file, csv_file_cluster_Root = self.get_filename_path(self.srcfilepath,
                                                                                  csv_file_cluster_name)
            for i in range(len(csv_file_cluster_file)):
                if csv_file_cluster_file[i][0] != '.':
                    csv_file_cluster_root = csv_file_cluster_Root[i] + '/' + csv_file_cluster_file[i]
            #############################################
            csv_file_cluster = pd.read_csv(csv_file_cluster_root)
            barcode_cluster = np.asarray(csv_file_cluster["Barcode"])
            nEphys_cluster = np.asarray(csv_file_cluster["Loupe Clusters"])

            column_list_csv = ["barcode", "selection", "y", "x", "pixel_y", "pixel_x"]
            csv_file_name = 'tissue_positions_list.csv'
            csv_file, csv_Root = self.get_filename_path(self.srcfilepath, csv_file_name)
            for i in range(len(csv_file)):
                if csv_file[i][0] != '.':
                    csv_root = csv_Root[i] + '/' + csv_file[i]

            csv_file = pd.read_csv(csv_root, names=column_list_csv)
            csv_file.to_excel(self.srcfilepath + "tissue_positions_list.xlsx", index=False)

            scatter_x = np.asarray(csv_file["pixel_x"])
            scatter_y = np.asarray(csv_file["pixel_y"])
            barcode_raw = np.asarray(csv_file["barcode"])

            group = np.asarray(csv_file["selection"])

            mask_id = [i for i in range(len(group)) if group[i] == 1]
            extent = [min([scatter_x[i] for i in mask_id]), max([scatter_x[i] for i in mask_id]),
                      min([scatter_y[i] for i in mask_id]), max([scatter_y[i] for i in mask_id])]
            g = 1
            ix = np.where(group == g)

            Cluster_list = self.clusters
            color_map = []

            for i in ix[0]:
                bar_code = csv_file['barcode'][i]
                try:
                    clu = nEphys_cluster[list(barcode_cluster).index(bar_code)]
                    for j in range(len(Cluster_list)):
                        if Cluster_list[j] == clu:
                            color_map.append(j)
                except:
                    color_map.append(len(Cluster_list))

            x_norm = [i for i in scatter_x[ix] - extent[0]]
            y_norm = [i for i in scatter_y[ix] - extent[2]]
            barcode_norm = [i for i in barcode_raw[ix]]

            color_map_filter_id = [i for i in range(len(color_map)) if color_map[i] != len(self.clusters)]
            x_norm = np.asarray([x_norm[i] * Pixel_SRT_mm for i in color_map_filter_id])
            y_norm = np.asarray([y_norm[i] * Pixel_SRT_mm for i in color_map_filter_id])
            x_norm_raw = x_norm
            y_norm_raw = y_norm
            barcode_norm = np.asarray([barcode_norm[i] for i in color_map_filter_id])
            filter_channel = [np.where(np.asarray(barcode_norm) == bar)[0] for bar in barcodes_filter_1_filter if
                              bar in barcode_norm]
            # for bar in barcodes_filter_1_filter:
            #     if bar in barcode_norm:
            #         np.where(np.asarray(barcode_norm) == bar)[0]
            x_norm = list(x_norm[filter_channel])
            y_norm = list(y_norm[filter_channel])
            # barcode_norm = barcode_norm[filter_channel]
            W_Matrix = W_Matrix.T
            for i in range(k):
                color_map = [node * 10 for node in list(W_Matrix[i])]
                fig, ax = plt.subplots()
                ax.scatter(x_norm, y_norm, c='black', marker='o', linewidth=1, s=color_map, alpha=1, edgecolors='black')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                x_ticks_raw = np.linspace(min(x_norm_raw), max(x_norm_raw), 5, endpoint=True)
                x_ticks = [str(round(k - min(x_ticks_raw), 2)) for k in x_ticks_raw]
                ax.set_xticks(x_ticks_raw)
                ax.set_xticklabels(x_ticks)
                ax.set_xlabel('Length(mm)', fontsize=8)
                y_ticks_raw = np.linspace(min(y_norm_raw), max(y_norm_raw), 5, endpoint=True)
                y_ticks = [str(round(k - min(y_ticks_raw), 2)) for k in y_ticks_raw]
                ax.set_yticks(y_ticks_raw)
                ax.set_yticklabels(y_ticks)
                ax.set_ylabel('Length(mm)', fontsize=8)
                ax.set_title('Factor ' + str(i + 1) + '-' + type_choose_name, fontsize=8)
                ax.set_aspect('equal', 'box')
                colorMapTitle_ST = 'W_Matrix_' + type_choose_name + '_' + cluster + '_' + 'Factor_' + str(i + 1)
                fig.savefig(desfilepath + colorMapTitle_ST + ".png", format='png', dpi=600)
                plt.close()


if __name__ == '__main__':
    srcfilepath = r'Z:/ANALYSES/SPATIOSCALES- 10X genomics/Data/'  # main path
    ############Basic Statistic####################################
    Analysis = MEASeqX_Project(srcfilepath)
    ################################################################# Network Activity Features NMF
    Analysis.SRT_nEphys_network_activity_features_NMF_reconstruction_error(type_choose_name='LFPRate', min_k=1, max_k=100, cluster='All',Given_gene_list=False, gene_list='IEGs', rotate_matrix=False) # Step 1 individual
    Analysis.SRT_nEphys_network_activity_features_NMF(type_choose_name='LFPRate', k=12, cluster='All', Given_gene_list=False, gene_list='IEGs', rotate_matrix=False) # Step 2 individual
    ################################################################# Network Topological Metrics NMF
    Analysis.SRT_nEphys_network_topological_metrics_NMF_reconstruction_error(type_choose_name='Degree', min_k=1, max_k=100, cluster='All',Given_gene_list=False, gene_list='IEGs', rotate_matrix=False) # Step 1 individual
    Analysis.SRT_nEphys_network_topological_metrics_NMF(type_choose_name='Degree', k=12, cluster='All', Given_gene_list=False, gene_list='IEGs', rotate_matrix=False)  # Step 1 individual # Step 2 individual

