# -*- coding: utf-8 -*-
"""
Created on Sept 07 2021
@author:  BIONICS_LAB
@company: DZNE
"""
import matplotlib.pyplot as plt
from anndata import AnnData
from sklearn.decomposition import PCA
import scanpy as sc
from scipy import stats
import h5py
import numpy as np
import pandas as pd
import json
import os
import seaborn as sns
import scipy.sparse as sp_sparse
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

"""
The following input parameters are used for specific gene list and select gene plotting. 
To compare conditions put the path for datasets in the input parameters and label condition name i.e. SD and ENR and assign desired color

"""

rows = 64
cols = 64

column_list = ["IEGs"]  ### "Hippo Signaling Pathway","Synaptic Vescicles_Adhesion","Receptors and channels","Synaptic plasticity","Hippocampal Neurogenesis","IEGs"
select_genes = ['Arc', 'Bdnf', 'Egr1', 'Egr3', 'Egr4', 'Fosb']  ### >5 genes needed

conditions = ['SD', 'ENR']
condition1_path = r'Z:/ANALYSES/SPATIOSCALES- 10X genomics/Data/SD/'
condition2_path = r'Z:/ANALYSES/SPATIOSCALES- 10X genomics/Data/ENR/'

color = ['silver', 'dodgerblue']  # color for pooled plotting of conditions

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

        return csv_file, tissue_lowres_scalef, features_name, matr_raw, barcodes, img, csv_file_cluster

    def UMIs_Gene_plot(self, img_cut=None, extent=None, x_filter=None, y_filter=None, genes_per_cell=None, umis_per_cell=None, cdict=None, g=2, label=None,path=None):
        fig, (ax, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 8))

        ax.imshow(img_cut)
        ax.scatter(x_filter - extent[0], y_filter - extent[2], c=cdict[g], label=label[g], s=2, alpha=1)
        ######################
        ax.legend(fontsize='small')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.set_xticks([])
        ax.set_yticks([])

        #######################
        ax1.imshow(img_cut, alpha=0.7)
        ################
        pt = ax1.scatter(x_filter - extent[0], y_filter - extent[2], c=umis_per_cell, cmap='jet', s=2, alpha=1)
        ######################
        pt.set_clim(vmax=80000)
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("bottom", size="3%", pad=0.2)
        cbar = fig.colorbar(pt, shrink=.1, label='UMIs per cell', cax=cax, orientation="horizontal")

        # plt.colorbar(pt,shrink=.3,label = 'UMIs per cell',ax=ax1)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax1.get_yticklabels(), visible=False)
        ax1.set_xticks([])
        ax1.set_yticks([])

        #################################
        ax2.imshow(img_cut, alpha=0.7)
        pt1 = ax2.scatter(x_filter - extent[0], y_filter - extent[2], c=genes_per_cell, cmap='jet', s=2, alpha=1)
        pt1.set_clim(vmax=8000)
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("bottom", size="3%", pad=0.2)
        cbar = fig.colorbar(pt1, shrink=.1, label='Genes per cell', cax=cax, orientation="horizontal")
        # plt.colorbar(pt1,shrink=.3,label = 'Genes per cell',ax=ax2)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax2.get_yticklabels(), visible=False)
        ax2.set_xticks([])
        ax2.set_yticks([])

        ###########################################################
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.5, hspace=0)
        plt.savefig(path + 'all_gene_' + 'UMI' + ".png", format='png', dpi=600)

    def PCA_Clustering(self, gene_expression_series=None, x_filter=None, extent=None, y_filter=None, new_id_filter=None,img_cut=None, path=None):
        # PCA & clustering
        # Do PCA
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(gene_expression_series)
        # Store results of PCA in a data frame
        result = pd.DataFrame(pca_result, columns=['PCA%i' % i for i in range(3)])
        ########################
        from sklearn.cluster import KMeans
        distortions = []
        K = range(1, 15)
        for k in K:
            kmeanModel = KMeans(n_clusters=k)
            kmeanModel.fit(pca_result)
            distortions.append(kmeanModel.inertia_)
        # choose the best k
        interval = 0
        K = 0
        for i in range(1, len(distortions) - 1):
            if interval < abs(distortions[i] - distortions[i - 1]) / abs(
                    distortions[i + 1] - distortions[i]) and abs(distortions[i] - distortions[i - 1]) - abs(
                distortions[i + 1] - distortions[i]) > 0:
                interval = abs(distortions[i] - distortions[i - 1]) / abs(distortions[i + 1] - distortions[i])
                K = i + 2
        if K == 0:
            num_clus = 3
        else:
            num_clus = K
        cluster, centers, distance = self.k_means(pca_result, num_clus)
        x_coordinate, y_coordinate = x_filter - extent[0], y_filter - extent[2]
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        ax[0].scatter(pca_result[:, 0], pca_result[:, 1], c=cluster, cmap='jet', marker='o', s=2, alpha=1)
        ax[0].set_xlabel('1st principal component', fontsize=20)
        ax[0].set_ylabel('2nd principal component', fontsize=20)
        ax[0].set_title('PCA', fontsize=23)
        ax[0].grid(False)

        ################
        cluster = [cluster[i] for i in new_id_filter]
        ax[1].imshow(img_cut, alpha=1)
        ax[1].scatter(x_coordinate, y_coordinate, c='black', s=2, alpha=0.7)
        bul = ax[1].scatter(x_coordinate, y_coordinate, marker='o', c=cluster, cmap='jet', s=2, alpha=1)
        cbar = fig.colorbar(bul, ticks=np.arange(np.min(cluster), np.max(cluster) + 1), shrink=.7)
        cbar.set_label('Cluster')
        ax[1].set_xlabel('Pixel')
        ax[1].set_ylabel('Pixel')
        ax[1].set_ylim(max(y_coordinate), 0)
        ax[1].set_aspect('equal', 'box')
        ax[1].grid(False)
        ax[1].set_title('Clusters for electrodes', fontsize=23)
        colorMapTitle = 'PCA_Clustering'
        fig.savefig(path + colorMapTitle + ".png", format='png', dpi=600)
        plt.close()

    def k_means(self, data, num_clus=3, steps=200):

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

    def gene_expression(self, plot_UMIs=True, select_plot='select_genes', top_gene_show=20, plot_gene_expression=True, plot_mutual_information=True, top_common_gene_show=300, PCA_Clustering=True, gene_list_name=None, value=0.9):
        """
        Plot specific gene lists and select genes for analysis.

            File input needed:
            -------
                - related files
                - 'gene_list_all.xlsx'

            Parameters
            -------

            Returns
            -------

            File output:
            -------
            if plot_gene_expression = True AND select_plot = 'gene_list'
                - '[gene_list]_gene_expression.png'
                - '[gene_list]_gene_expression_per_cluster.xlsx'
            if plot_gene_expression = True AND select_plot = 'select genes'
                - 'select_gene_expression.png'
                - 'select_gene_expression_per_cluster.xlsx'
            if plot_gene_expression = True AND select_plot = 'top_expressed_genes'
                - 'top_expressed_gene_expression.png'
                - 'top_expressed_gene_expression_per_cluster.xlsx'
            if plot_gene_expression = True AND select_plot = 'top_expressed_common_genes'
                - 'top_expressed_common_gene_expression.png'
                - 'top_expressed_common_gene_expression_per_cluster.xlsx'
            if plot_mutual_information = True
                - 'mutual_information.txt'
                - '[gene_list]_mutual_information_paired_cluster.xlsx'
                - '[gene_list]_mutual_information_paired_cluster.png'
                - '[gene_list]_SRT_functional_connectivity.xlsx'
                - '[gene_list]_SRT_functional_connectivity.png'
            if plot_UMIs = True
                - 'all_gene_UMI.png'
            if PCA_clustering = True
                - 'PCA_Clustering.png'
        """

        path = self.srcfilepath[:self.srcfilepath.rfind('/')]
        desfilepath = path + '/Gene_Expression_Plots/'
        if not os.path.exists(desfilepath):
            os.mkdir(desfilepath)

        # Read related information
        csv_file, tissue_lowres_scalef, features_name, matr_raw, barcodes, img, csv_file_cluster = self.read_related_files()
        barcode_cluster = np.asarray(csv_file_cluster["Barcode"])
        Loupe_cluster = np.asarray(csv_file_cluster["Loupe Clusters"])

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

        # print(matr.sum(axis=0))# calculate UMIs and genes per cell
        # calculate UMIs and genes per cell
        umis_per_cell = np.asarray(matr.sum(axis=0)).squeeze()  # Or matr.sum(axis=0)
        genes_per_cell = np.asarray((matr > 0).sum(axis=0)).squeeze()

        ###############delete the nodes with less then 1000 gene count
        deleted_notes = [i for i in range(len(genes_per_cell)) if genes_per_cell[i] <= 1000]
        ###############delete the nodes not in clusters
        deleted_notes_cluster = [i for i in range(len(genes_per_cell)) if str(barcodes[i])[2:-1] not in barcode_cluster]
        deleted_notes.extend(deleted_notes_cluster)
        deleted_notes = list(np.unique(deleted_notes))
        ##########################################################
        matr = np.delete(matr, deleted_notes, axis=1)

        barcodes_filter = [barcodes[i] for i in range(len(barcodes)) if i not in deleted_notes]
        new_id_filter = [j for i in barcode_CSV for j in range(len(barcodes_filter)) if
                         str(barcodes_filter[j])[2:-1] == i]

        genes_per_cell_raw = [i if i > 1000 else 0 for i in genes_per_cell]
        umis_per_cell_raw = [umis_per_cell[i] if genes_per_cell[i] > 1000 else 0 for i in range(len(umis_per_cell))]

        new_id = [j for i in barcode_CSV for j in range(len(barcodes)) if str(barcodes[j])[2:-1] == i]

        umis_per_cell = [umis_per_cell_raw[new_id[i]] for i in range(len(new_id)) if i not in deleted_notes]
        genes_per_cell = [genes_per_cell_raw[new_id[i]] for i in range(len(new_id)) if i not in deleted_notes]
        x_filter = [scatter_x[ix][i] for i in range(len(scatter_x[ix])) if new_id[i] not in deleted_notes]
        y_filter = [scatter_y[ix][i] for i in range(len(scatter_y[ix])) if new_id[i] not in deleted_notes]
        ix_filter = [ix[0][i] for i in range(len(scatter_x[ix])) if new_id[i] not in deleted_notes]
        # print(len(new_id_filter),len(x_filter),max(new_id_filter))
        Barcodes_deleted_filter = barcodes_filter

        mask_id = [i for i in range(len(group)) if group[i] == 1]
        extent = [min([scatter_x[i] for i in mask_id]), max([scatter_x[i] for i in mask_id]),
                  min([scatter_y[i] for i in mask_id]), max([scatter_y[i] for i in mask_id])]

        img_cut = img[int(extent[2]):int(extent[3]) + 2, int(extent[0]):int(extent[1]) + 3,
                  :]  # x and y value set to cut the areas interested

        if plot_UMIs == True:
            self.UMIs_Gene_plot(img_cut=img_cut, extent=extent, x_filter=x_filter, y_filter=y_filter,
                                genes_per_cell=genes_per_cell, umis_per_cell=umis_per_cell, cdict=cdict, g=g,
                                label=label,path=desfilepath)

        adata = AnnData(np.array(matr))
        sc.pp.normalize_total(adata, inplace=True)
        gene_expression = adata.X
        genes_expression_count = np.asarray((matr > 0).sum(axis=1)).squeeze()
        ##########################cross correlation based on gene expression level
        gene_expression_filter = [i for i in np.asarray(gene_expression) if any(i)]
        ############################ choose top gene
        from operator import itemgetter
        indices, genes_expression_count_sorted = zip(
            *sorted(enumerate(genes_expression_count), key=itemgetter(1), reverse=True))
        ###################################Choose gene 1 way
        if select_plot == 'top_expressed_common_genes':
            Selected_select_genes = self.find_common_gene(condition1_path=condition1_path, condition2_path=condition2_path,
                                                          top_common_gene_show=top_common_gene_show)
            gene_name_list = [gene_name[i] for i in indices]
            top_gene_indices = [i for name in Selected_select_genes for i in range(len(gene_name_list)) if
                                len(re.findall(gene_name_list[i], name, flags=re.IGNORECASE)) > 0]
            top_gene_name = Selected_select_genes
        ###################################Choose gene way 2
        elif select_plot == 'gene_list':
            filetype_gene = 'gene_list_all.xlsx'
            filename_gene, Root = self.get_filename_path(self.srcfilepath, filetype_gene)
            for i in range(len(filename_gene)):
                if filename_gene[i][0] != '.':
                    gene_root = Root[i] + '/' + filename_gene[i]

            Selected_select_genes = list(pd.read_excel(gene_root)[gene_list_name])
            Selected_select_genes = [i for i in Selected_select_genes if type(i) == str]
            gene_name_list = [gene_name[i] for i in indices]
            top_gene_indices = [i for name in Selected_select_genes for i in range(len(gene_name_list)) if
                                len(re.findall(gene_name_list[i], name, flags=re.IGNORECASE)) > 0]
            top_gene_name = Selected_select_genes
        elif select_plot == 'select_genes':
            Selected_select_genes = select_genes
            gene_name_list = [gene_name[i] for i in indices]
            top_gene_indices = [i for name in Selected_select_genes for i in range(len(gene_name_list)) if
                                len(re.findall(gene_name_list[i], name, flags=re.IGNORECASE)) > 0]
            top_gene_name = Selected_select_genes
        else:
            top_gene_indices = indices[:top_gene_show]  # choose the top 300 expression genes
            df_gene_count = pd.DataFrame({'Gene Name': [gene_name[i] for i in indices],
                                          'Gene Count sorted': list(genes_expression_count_sorted)})
            top_gene_name = list(df_gene_count['Gene Name'][:top_gene_show])

        print('check2', top_gene_name)
        #####################################
        gene_expression_filter = [np.asarray(gene_expression)[i] for i in top_gene_indices]
        gene_expression_series = np.asarray(gene_expression_filter).T
        id_no_expression = [i for i in range(len(gene_expression_series)) if sum(gene_expression_series[i]) == 0]
        gene_expression_series = np.delete(gene_expression_series, id_no_expression, axis=0)
        # Channel_ID = [i for i in range(len(genes_per_cell)) if i not in id_no_expression]
        Channel_ID = [new_id_filter[i] for i in range(len(new_id_filter)) if i not in id_no_expression]
        # x_filter = [x_filter[i] for i in range(len(x_filter)) if i not in id_no_expression]
        # y_filter = [y_filter[i] for i in range(len(y_filter)) if i not in id_no_expression]
        barcodes_filter_1 = [barcodes_filter[i] for i in range(len(barcodes_filter)) if i not in id_no_expression]
        new_id_filter_1 = [j for i in barcode_CSV for j in range(len(barcodes_filter_1)) if
                           str(barcodes_filter_1[j])[2:-1] == i]


        # choose the highest top expression gene
        if plot_gene_expression == True:
            Cluster_list = list(self.clusters)
            Cluster_list.append('Not in Cluster')

            color_map = []
            barcode_list = []
            for i in ix_filter:
                bar_code = csv_file['barcode'][i]
                barcode_list.append(bar_code)
                try:
                    clu = Loupe_cluster[list(barcode_cluster).index(bar_code)]
                    for j in range(len(Cluster_list)):
                        if Cluster_list[j] == clu:
                            color_map.append(j)
                except:
                    color_map.append(len(Cluster_list) - 1)

            cluster_list = [Cluster_list[i] for i in color_map]
            print(top_gene_name)

            self.plot_top_expressed_gene(top_gene_name=top_gene_name, gene_name=gene_name,
                                         gene_expression=gene_expression,
                                         new_id_filter=new_id_filter, img_cut=img_cut, x_filter=x_filter,
                                         y_filter=y_filter, extent=extent, cluster_list=cluster_list,
                                         barcode_list=barcode_list, gene_list_name=gene_list_name,
                                         gene_from_option=select_plot, path=desfilepath)

        if plot_mutual_information == True:
            self.plot_mutual_information(gene_expression_series=gene_expression_series,
                                                                   x_filter=x_filter, extent=extent, y_filter=y_filter,
                                                                   img_cut=img_cut, Channel_ID=Channel_ID,
                                                                   gene_list_name=gene_list_name,
                                                                   Barcodes=Barcodes_deleted_filter,
                                                                   new_id_filter=new_id_filter_1, value=value,
                                                                   barcode_filter_for_cluster=barcodes_filter_1,
                                                                   barcode_cluster=barcode_cluster,
                                                                   Loupe_cluster=Loupe_cluster)
        if PCA_Clustering == True:
            self.PCA_Clustering(gene_expression_series=gene_expression, x_filter=x_filter, extent=extent,
                                y_filter=y_filter, new_id_filter=new_id_filter, img_cut=img_cut, path=desfilepath)
        print('Done')

    def find_common_gene(self, condition1_path=None, condition2_path=None, top_common_gene_show=300):
        ##########################
        h5_file_name_condition1 = 'filtered_feature_bc_matrix.h5'
        h5_file_condition1, h5_file_Root_condition1 = self.get_filename_path(condition1_path, h5_file_name_condition1)
        for i in range(len(h5_file_condition1)):
            if h5_file_condition1[i][0] != '.':
                h5_root_condition1 = h5_file_Root_condition1[i] + '/' + h5_file_condition1[i]
        #############################################
        filehdf5_10x = h5py.File(h5_root_condition1, 'r')
        features_name = np.asarray(filehdf5_10x["matrix"]["features"]['name'])
        gene_name_condition1 = [str(features_name[i])[2:-1] for i in range(len(features_name))]

        shape = np.asarray(filehdf5_10x["matrix"]['shape'])
        indices = np.asarray(filehdf5_10x["matrix"]["indices"])
        indptr = np.asarray(filehdf5_10x["matrix"]["indptr"])
        data = np.asarray(filehdf5_10x["matrix"]["data"])
        matr_raw = sp_sparse.csc_matrix((data, indices, indptr), shape=shape).toarray()

        adata = AnnData(np.array(matr_raw))
        sc.pp.normalize_total(adata, inplace=True)
        genes_expression_count = np.asarray((adata.X > 0).sum(axis=1)).squeeze()
        from operator import itemgetter
        indices, genes_expression_count_sorted = zip(
            *sorted(enumerate(genes_expression_count), key=itemgetter(1), reverse=True))
        df_gene_count_condition1 = pd.DataFrame({'Gene Name': [gene_name_condition1[i] for i in indices],
                                          'Gene Count sorted': list(genes_expression_count_sorted)})
        ##########################
        ##########################
        h5_file_name_condition2 = 'filtered_feature_bc_matrix.h5'
        h5_file_condition2, h5_file_Root_condition2 = self.get_filename_path(condition2_path, h5_file_name_condition2)
        for i in range(len(h5_file_condition2)):
            if h5_file_condition2[i][0] != '.':
                h5_root_condition2 = h5_file_Root_condition2[i] + '/' + h5_file_condition2[i]
        #############################################
        filehdf5_10x = h5py.File(h5_root_condition2, 'r')
        features_name = np.asarray(filehdf5_10x["matrix"]["features"]['name'])
        gene_name_condition2 = [str(features_name[i])[2:-1] for i in range(len(features_name))]

        shape = np.asarray(filehdf5_10x["matrix"]['shape'])
        indices = np.asarray(filehdf5_10x["matrix"]["indices"])
        indptr = np.asarray(filehdf5_10x["matrix"]["indptr"])
        data = np.asarray(filehdf5_10x["matrix"]["data"])
        matr_raw = sp_sparse.csc_matrix((data, indices, indptr), shape=shape).toarray()
        adata = AnnData(np.array(matr_raw))
        sc.pp.normalize_total(adata, inplace=True)
        genes_expression_count = np.asarray((adata.X > 0).sum(axis=1)).squeeze()
        from operator import itemgetter
        indices, genes_expression_count_sorted = zip(
            *sorted(enumerate(genes_expression_count), key=itemgetter(1), reverse=True))
        df_gene_count_condition2 = pd.DataFrame({'Gene Name': [gene_name_condition1[i] for i in indices],
                                          'Gene Count sorted': list(genes_expression_count_sorted)})

        common_gene = list(set(gene_name_condition1) & set(gene_name_condition2))

        import re
        filter_gene_id = [i for i in range(len(common_gene)) if
                          len(re.findall(r'^SRS', common_gene[i], flags=re.IGNORECASE)) > 0 or len(
                              re.findall(r'^Mrp', common_gene[i], flags=re.IGNORECASE)) > 0 or len(
                              re.findall(r'^Rp', common_gene[i], flags=re.IGNORECASE)) > 0 or len(
                              re.findall(r'^mt', common_gene[i], flags=re.IGNORECASE)) > 0 or len(
                              re.findall(r'^Ptbp', common_gene[i], flags=re.IGNORECASE)) > 0]
        common_gene = [common_gene[i] for i in range(len(common_gene)) if i not in filter_gene_id]
        ##############
        ############################ choose top gene
        set_gene_condition1 = [i for i in df_gene_count_condition1['Gene Name'] if i in common_gene]
        set_gene_condition2 = [i for i in df_gene_count_condition2['Gene Name'] if i in common_gene]
        for k in range(100, len(common_gene), 100):
            if len(list(set(set_gene_condition1[:top_common_gene_show + k]) & set(
                    set_gene_condition2[:top_common_gene_show + k]))) >= top_common_gene_show:
                common_gene_top = list(
                    set(set_gene_condition1[:top_common_gene_show + k]) & set(set_gene_condition2[:top_common_gene_show + k]))[
                                  :top_common_gene_show]
                break
            else:
                continue
        return common_gene_top

    def plot_top_expressed_gene(self, top_gene_name=None, gene_name=None, gene_from_option=None, gene_expression=None,
                                new_id_filter=None, img_cut=None, x_filter=None, y_filter=None, extent=None,
                                cluster_list=None, barcode_list=None, gene_list_name=None, path=None):

        # from operator import itemgetter
        # indices, genes_expression_count_sorted = zip(*sorted(enumerate(genes_expression_count), key=itemgetter(1), reverse=True))
        # df_gene_count = pd.DataFrame({'Gene Name': [gene_name[i] for i in indices], 'Gene Count sorted': list(genes_expression_count_sorted)})
        # top_gene_name = df_gene_count['Gene Name'][:top_gene_show]
        # print(top_gene_name)
        #########################################################
        cluster_all = []
        gene_expression_values_all = []
        gene_name_all = []
        channel_position = []
        barcode_list_all = []
        # print(list(top_gene_name))
        fig, ax = plt.subplots(nrows=int(len(list(top_gene_name)) / 5) + 1, ncols=5,
                               figsize=(20, 20))  # , facecolor='None'
        k = 0
        import re
        for gene_name_choose in list(top_gene_name):
            id_find = -1
            for i in range(len(gene_name)):
                if len(re.findall(gene_name_choose, gene_name[i], flags=re.IGNORECASE)) > 0:
                    # if gene_name_choose == gene_name[i]:
                    id_find = i
            if id_find != -1:
                gene_expression_find = gene_expression[id_find]
                gene_expression_find = [gene_expression_find[i] for i in new_id_filter]
                gene_expression_values_all.extend(gene_expression_find)
                gene_name_all.extend([gene_name_choose] * len(gene_expression_find))
                cluster_all.extend(cluster_list)
                barcode_list_all.extend(barcode_list)
                channel_position.extend(
                    [((x_filter - extent[0])[i], (y_filter - extent[2])[i]) for i in range(len(x_filter))])
                #######################
                # print('k',k)
                ax[int(k / 5), int(k % 5)].imshow(img_cut, alpha=1)
                ################
                pt = ax[int(k / 5), int(k % 5)].scatter(x_filter - extent[0], y_filter - extent[2],
                                                        c=gene_expression_find, cmap='jet', s=2, alpha=0.7)
                ######################
                # pt.set_clim(vmax=80000)
                divider = make_axes_locatable(ax[int(k / 5), int(k % 5)])
                cax = divider.append_axes("bottom", size="3%", pad=0.2)
                cbar = fig.colorbar(pt, shrink=.1, label='Gene Expression', cax=cax, orientation='horizontal')

                ax[int(k / 5), int(k % 5)].spines['top'].set_visible(False)
                ax[int(k / 5), int(k % 5)].spines['right'].set_visible(False)
                ax[int(k / 5), int(k % 5)].spines['bottom'].set_visible(False)
                ax[int(k / 5), int(k % 5)].spines['left'].set_visible(False)
                plt.setp(ax[int(k / 5), int(k % 5)].get_xticklabels(), visible=False)
                plt.setp(ax[int(k / 5), int(k % 5)].get_yticklabels(), visible=False)
                ax[int(k / 5), int(k % 5)].set_xticks([])
                ax[int(k / 5), int(k % 5)].set_yticks([])
                ax[int(k / 5), int(k % 5)].set_title(gene_name_choose)
                # ax[int(k / 5), int(k % 5)].set_box_aspect(1)

                k += 1
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.1, hspace=1)
        # select_plot = 'select_genes', 'top_expressed_common_genes','gene_list','top_expressed_genes'
        if gene_from_option == 'gene_list':
            colorMapTitle = gene_list_name
        elif gene_from_option == 'top_expressed_common_genes':
            colorMapTitle = 'top_expressed_common'
        elif gene_from_option == 'select_genes':
            colorMapTitle = 'select'
        else:
            colorMapTitle = 'top_expressed'
        fig.savefig(path + colorMapTitle + '_gene_expression' + ".png", format='png', dpi=600)
        plt.close()
        ###########################
        # cluster_all = []
        # gene_expression_values_all = []
        # gene_name_all = []
        df = pd.DataFrame(
            {'Barcode': barcode_list_all, 'Channel Position': channel_position, 'gene Name': gene_name_all,
             'Gene Expression Level': gene_expression_values_all, 'Cluster': cluster_all})
        df.to_excel(path + colorMapTitle + "_gene_expression_per_cluster" + ".xlsx", index=False)

    def plot_mutual_information(self, gene_expression_series=None, x_filter=None, extent=None, y_filter=None, img_cut=None, Channel_ID=None,gene_list_name=None, Barcodes=None, new_id_filter=None,value=0.9, barcode_filter_for_cluster=None, barcode_cluster=None, Loupe_cluster=None):
        ################################
        desfilepath = self.srcfilepath + 'Mutual_Information/'
        if not os.path.exists(desfilepath):
            os.mkdir(desfilepath)
        barcode_filter_for_cluster = [str(i)[2:-1] for i in barcode_filter_for_cluster]
        cluster_gene_expression = \
            [Loupe_cluster[list(barcode_cluster).index(bar)] if bar in barcode_cluster else 'Not in Cluster' for bar in
             barcode_filter_for_cluster]
        gene_expression_series_sorted = []
        cluster_gene_expression_sorted = []
        for clu in self.clusters:
            gene_expression_series_sorted.extend(
                [gene_expression_series[i] for i in range(len(cluster_gene_expression)) if
                 cluster_gene_expression[i] == clu])
            cluster_gene_expression_sorted.extend(
                [cluster_gene_expression[i] for i in range(len(cluster_gene_expression)) if
                 cluster_gene_expression[i] == clu])
        ###################################
        # [2: -1]
        file_path = desfilepath + 'mutual_information' + ".txt"
        f = open(file_path, "a")
        f.seek(0)
        f.truncate()
        cluster_gene_expression_sorted_uni, cluster_index = np.unique(cluster_gene_expression_sorted, return_index=True)
        cluster_group = [(cluster_gene_expression_sorted_uni[i], cluster_index[i]) for i in range(len(cluster_index))]

        def takeSecond(elem):
            return elem[1]

        cluster_group.sort(key=takeSecond)
        cluster_gene_expression_sorted_uni = [cluster_group[i][0] for i in range(len(cluster_group))]
        cluster_index = [cluster_group[i][1] for i in range(len(cluster_group))]
        for i in range(len(cluster_gene_expression_sorted_uni)):
            if i == len(cluster_index) - 1:
                msg = cluster_gene_expression_sorted_uni[i] + ' goes from ' + str(cluster_index[i]) + ' to ' + str(
                    len(cluster_gene_expression_sorted))
                f.write(msg + '\n')
            else:
                msg = cluster_gene_expression_sorted_uni[i] + ' goes from ' + str(cluster_index[i]) + ' to ' + str(
                    cluster_index[i + 1] - 1)
                f.write(msg + '\n')

        f.close()

        # else:
        import math
        import sklearn.metrics as sm

        paired_cluster = []
        mi_values = []
        Per_cor = np.zeros(
            (len(np.asarray(gene_expression_series_sorted)), len(np.asarray(gene_expression_series_sorted))),
            dtype=np.float)
        for i in range(len(np.asarray(gene_expression_series_sorted))):
            clu_1 = str(cluster_gene_expression_sorted[i])
            for j in range(i, len(np.asarray(gene_expression_series_sorted))):
                clu_2 = str(cluster_gene_expression_sorted[j])
                Per_cor[i, j] = sm.normalized_mutual_info_score(np.asarray(gene_expression_series_sorted)[i],
                                                                np.asarray(gene_expression_series_sorted)[j])
                Per_cor[j, i] = Per_cor[i, j]
                paired_cluster.append(clu_1 + '_' + clu_2)
                mi_values.append(Per_cor[i, j])
        # Per_cor = np.reshape(Per_cor, (len(gene_expression_series), len(gene_expression_series)))
        where_are_NaNs = np.isnan(Per_cor)
        Per_cor[where_are_NaNs] = 0
        n = Per_cor.shape[0]
        Per_cor[range(n), range(n)] = 0
        # print(pd.DataFrame(Per_cor))
        fig = plt.figure()
        jet = plt.get_cmap('cool')
        ax = fig.add_subplot(111)
        cax = ax.imshow(Per_cor, cmap=jet)
        cax.set_clim(0, 1.0)
        ax.set_xlabel('Channel ID Sorted')
        ax.set_ylabel('Channel ID Sorted')
        ax.grid(False)
        fig.colorbar(cax, label='Mutual Information', ax=ax, shrink=.3)
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ##############################
        colorMapTitle = gene_list_name + "_mutual_information_paired_cluster"
        fig.savefig(desfilepath + colorMapTitle + ".png", format='png', dpi=600)

        d = {'Paired Cluster': paired_cluster, 'MI values': mi_values}
        df = pd.DataFrame(data=d)
        df.to_excel(desfilepath + colorMapTitle + ".xlsx", index=False)

        ####################################################
        Channel_ID = [i for i in range(len(Per_cor))]
        Channel_ID_new = []
        Barcodes_Channel_ID = []

        Corr_id = []
        Barcodes_Corr_id = []
        coor_Value = []
        ###############################
        for i in range(len(Per_cor)):
            Corr_id.extend([Channel_ID[new_id_filter[j]] for j in range(len(Per_cor[i, :])) if Per_cor[i, j] >= value])
            Barcodes_Corr_id.extend(
                [Barcodes[new_id_filter[j]] for j in range(len(Per_cor[i, :])) if Per_cor[i, j] >= value])
            coor_Value.extend([j for j in Per_cor[i, :] if j >= value])
            Channel_ID_new.extend(
                [Channel_ID[new_id_filter[i]]] * len([j for j in range(len(Per_cor[i, :])) if Per_cor[i, j] >= value]))
            Barcodes_Channel_ID.extend(
                [Barcodes[new_id_filter[i]]] * len([j for j in range(len(Per_cor[i, :])) if Per_cor[i, j] >= value]))

        x_coordinate, y_coordinate = x_filter - extent[0], y_filter - extent[2]
        dataframe = pd.DataFrame(
            {'Channel ID': Channel_ID_new, 'Barcodes_Channel_ID': Barcodes_Channel_ID, 'Corr_id': Corr_id,
             'Barcodes_Corr_id': Barcodes_Corr_id, 'coor_Per_data': coor_Value})
        dataframe.to_excel(self.srcfilepath + gene_list_name + '_SRT_functional_connectivity' + ".xlsx", index=False)
        # print(dataframe)
        fig, ax = plt.subplots()
        ax.imshow(img_cut, alpha=1)
        ################
        ax.scatter(x_coordinate, y_coordinate, c='black', s=2, alpha=0.7)
        ax.set_xlabel('Pixel')
        ax.set_ylabel('Pixel')
        count = 0
        for i in range(len(Channel_ID_new)):
            if count % 100 == 1:  # choose 50% correlation to plot
                df = pd.DataFrame({'x': (x_coordinate[Channel_ID_new[i]], x_coordinate[Corr_id[i]]),
                                   'y': (y_coordinate[Channel_ID_new[i]], y_coordinate[Corr_id[i]])})
                ax.plot('x', 'y', data=df, color='blue', linewidth=0.5, alpha=0.3)
            count += 1
        ax.set_ylim(max(y_coordinate), 0)
        ax.set_aspect('equal', 'box')
        ax.grid(False)
        colorMapTitle = gene_list_name + "_SRT_functional_connectivity"
        fig.savefig(self.srcfilepath + colorMapTitle + ".png", format='png', dpi=600)

    def mutual_information_statistics(self, gene_list_choose='IEGs'):
        """
        Prepare mutual information distance score statistics

            File input needed:
            -------
                - '[gene_list]_mutual_information_paired_cluster.xlsx'

            Parameters
            -------

            Returns
            -------

            File output:
            -------
                - '[gene_list]_mutual_information_paired_cluster_statistics.xlsx
        """

        desfilepath = self.srcfilepath + 'Mutual_Information/'
        if not os.path.exists(desfilepath):
            os.mkdir(desfilepath)
        filetype_excel = gene_list_choose + '_mutual_information_paired_cluster.xlsx'
        filename_excel, Root = self.get_filename_path(self.srcfilepath, filetype_excel)

        for i in range(len(filename_excel)):
            if filename_excel[i][0] != '.':
                Cluster_con = []
                Mean_for_clusters = []
                Sem_for_clusters = []
                data = pd.read_excel(Root[i] + '/' + filename_excel[i])
                paired_cluster_uni = np.unique(data['Paired Cluster'])
                for clu in paired_cluster_uni:
                    Cluster_con.append(clu)
                    data_clu = data.copy()
                    data_clu = data_clu[data_clu['Paired Cluster'] == clu]
                    s = pd.Series(range(len(data_clu)))
                    data_clu = data_clu.set_index(s)
                    Mean_for_clusters.append(np.mean(data_clu['MI values']))
                    Sem_for_clusters.append(stats.sem(data_clu['MI values']))

                df = pd.DataFrame(
                    {'Clusters': Cluster_con, 'Mean for MI values': Mean_for_clusters,
                     'SEM for MI values': Sem_for_clusters})
                df.to_excel(desfilepath + filename_excel[i][:-5] + "_statistics" + ".xlsx", index=False)

    def mutual_information_pooled_statistics(self, gene_list_choose='IEGs'):
        """
        Compare mutual information distance scores between input conditions i.e. SD and ENR

            File input needed:
            -------
                - '[gene_list]_mutual_information_paired_cluster.xlsx'

            Parameters
            -------

            Returns
            -------

            File output:
            -------
                - '[gene_list]_mutual_information_paired_cluster_pooled_statistics.xlsx'
                - '[gene_list]_mutual_information_paired_cluster_pooled_statistics_p_values.xlsx'
        """


        path = self.srcfilepath[:self.srcfilepath.rfind('/')]
        desfilepath = path + '/Mutual_Information_Pooled_Statistics/'
        if not os.path.exists(desfilepath):
            os.mkdir(desfilepath)
        filetype_excel = gene_list_choose + '_mutual_information_paired_cluster.xlsx'
        filename_excel, Root = self.get_filename_path(self.srcfilepath, filetype_excel)
        Cluster_con = []
        final_Gene_expression_clusters = pd.DataFrame()

        for i in range(len(filename_excel)):
            if filename_excel[i][0] != '.':
                data = pd.read_excel(Root[i] + '/' + filename_excel[i])
                final_Gene_expression_clusters = pd.concat([final_Gene_expression_clusters, data], axis=0).reset_index()
                for con_name in conditions:
                    if con_name in Root[i]:
                        Cluster_con.extend([con_name] * len(data))

        final_Gene_expression_clusters['Condition'] = Cluster_con
        final_Gene_expression_clusters = final_Gene_expression_clusters.drop(['level_0', 'index'], axis=1)
        final_Gene_expression_clusters.to_excel(
            desfilepath + gene_list_choose + '_mutual_information_paired_cluster_pooled_statistics.xlsx', index=False)
        p_values = []
        Paired_Cluster = []
        paired_cluster_uni = np.unique(final_Gene_expression_clusters['Paired Cluster'])
        for clu in paired_cluster_uni:
            Cluster_con.append(clu)
            data_clu = final_Gene_expression_clusters.copy()
            data_clu = data_clu[data_clu['Paired Cluster'] == clu]
            s = pd.Series(range(len(data_clu)))
            data_clu = data_clu.set_index(s)
            #####################################
            value_con = data_clu.copy()
            value_con = value_con[value_con['Condition'] == conditions[1]]
            s = pd.Series(range(len(value_con)))
            value_con = value_con.set_index(s)
            condition2_Values = list(value_con['MI values'])
            value_con = data_clu.copy()
            value_con = value_con[value_con['Condition'] == conditions[0]]
            s = pd.Series(range(len(value_con)))
            value_con = value_con.set_index(s)
            condition1_Values = list(value_con['MI values'])

            if len(condition2_Values) > 0 and len(condition1_Values) > 0:
                p_values.append(float(stats.ttest_ind(condition2_Values, condition1_Values, equal_var=False, nan_policy='omit')[1]))
                Paired_Cluster.append(clu)

        df = pd.DataFrame({'Clusters': Paired_Cluster, 'p_values': p_values})
        df.to_excel(desfilepath + gene_list_choose + '_mutual_information_paired_cluster_pooled_statistics_p_values.xlsx',
                    index=False)

    def gene_expression_pooled_statistics(self, gene_list_choose='IEGs'):
        """
        Compare gene expression values within a given gene list between conditions i.e. SD and ENR

            File input needed:
            -------
                - '[gene_list]_gene_expression_per_cluster.xlsx'

            Parameters
            -------

            Returns
            -------

            File output:
            -------
                - '[gene_list]_gene_expression_per_cluster_pooled.xlsx'
                - '[gene_list]_gene_expression_pooled_statistics.png'
                - '[gene_list]_gene_expression_pooled_statistics_p_values.xlsx'
        """
        path = self.srcfilepath[:self.srcfilepath.rfind('/')]
        desfilepath = path + '/Gene_Expression_Pooled_Statistics/'
        if not os.path.exists(desfilepath):
            os.mkdir(desfilepath)
        if os.path.exists(desfilepath + gene_list_choose + "_gene_expression_per_cluster_pooled" + ".xlsx"):
            print('Find ' + gene_list_choose + "_gene_expression_per_cluster_pooled" + ".xlsx")
        else:
            filetype_excel = gene_list_choose + '_gene_expression_per_cluster.xlsx'
            filename_excel, Root = self.get_filename_path(self.srcfilepath, filetype_excel)
            Barcode = []
            Channel_Position = []
            gene_Name = []
            Gene_Expression_Level = []
            Cluster = []
            Condition_all = []
            file_name_all = []
            for i in range(len(filename_excel)):
                if filename_excel[i][0] != '.':
                    data = pd.read_excel(Root[i] + '/' + filename_excel[i])
                    gene_Name.extend(data['gene Name'])
                    Barcode.extend(data['Barcode'])
                    Channel_Position.extend(data['Channel Position'])
                    Cluster.extend(data['Cluster'])
                    data['Gene Expression Level'] = data['Gene Expression Level'].fillna(0)
                    Gene_Expression_Level.extend(data['Gene Expression Level'])
                    filetype_bxr = '.bxr'
                    root_raw = Root[i][:Root[i].rfind('/')]
                    filename_bxr, Root_bxr = self.get_filename_path(root_raw + '/', filetype_bxr)
                    file_name_all.extend([filename_bxr[0][:-4]] * len(data['gene Name']))
                    name = 'No condition'
                    for con_name in conditions:
                        if con_name in Root[i]:
                            name = con_name

                    Condition_all.extend([name] * len(data['gene Name']))
            df_Gene_expression_clusters_all = pd.DataFrame(
                {'Barcode': Barcode, 'Channel Position': Channel_Position, 'Gene Name': gene_Name,
                 'Gene Expression Level': Gene_Expression_Level, 'Cluster': Cluster,
                 'File Name': file_name_all, 'Condition': Condition_all})
            df_Gene_expression_clusters_all.to_excel(
                desfilepath + gene_list_choose + "_gene_expression_per_cluster_pooled" + ".xlsx",
                index=False)

        filetype_excel = gene_list_choose + "_gene_expression_per_cluster_pooled" + ".xlsx"
        filename_excel, Root = self.get_filename_path(path, filetype_excel)
        for i in range(len(filename_excel)):
            if filename_excel[i][0] != '.':
                data = pd.read_excel(desfilepath + filename_excel[i])
                Gene_Name = data['Gene Name']
                expression_level = data['Gene Expression Level']
                Gene_Name_uni = np.unique(Gene_Name)
                mean_gene_expression = [
                    np.mean([expression_level[i] for i in range(len(Gene_Name)) if Gene_Name[i] == gene]) for gene in
                    Gene_Name_uni]
                conbin = [(mean_gene_expression[i], Gene_Name_uni[i]) for i in range(len(Gene_Name_uni))]

                def takeOne(elem):
                    return elem[0]

                conbin.sort(key=takeOne, reverse=True)
                top_high_expression_gene = [i[1] for i in conbin]
                print(top_high_expression_gene)
                fig, ax = plt.subplots(ncols=1, nrows=int(len(top_high_expression_gene)), figsize=(10, 50))
                count = 0
                writer = pd.ExcelWriter(desfilepath + gene_list_choose + "_gene_expression_pooled_statistics_p_values" + '.xlsx',
                                        engine='xlsxwriter')
                for gene in top_high_expression_gene:
                    data_new = data.copy()
                    data_new = data_new[data_new['Gene Name'] == gene]
                    s = pd.Series(range(len(data_new)))
                    data_new = data_new.set_index(s)
                    data_new['Gene Expression Level'] = data_new['Gene Expression Level'].fillna(0)
                    ##################################################################################
                    cluster_gene = []
                    p_values = []
                    clusters_unique = np.unique(list(data_new['Cluster']))
                    for clu in clusters_unique:
                        data_new_copy = data_new.copy()
                        data_new_clu = data_new_copy[data_new_copy['Cluster'] == clu]
                        condition2_values = list(
                            data_new_clu[data_new_clu['Condition'] == conditions[1]]['Gene Expression Level'])
                        condition1_values = list(
                            data_new_clu[data_new_clu['Condition'] == conditions[0]]['Gene Expression Level'])
                        p_values.append(stats.ttest_ind(condition2_values, condition1_values)[1])
                        cluster_gene.append(clu)
                    final = pd.DataFrame({'Cluster': cluster_gene, 'p_value': p_values})
                    try:
                        final.to_excel(writer, sheet_name=gene, index=False)
                    except:
                        continue
                    ###################################################################################
                    sns.barplot(x='Cluster', y='Gene Expression Level', data=data_new, order=self.clusters,
                                ax=ax[count], hue_order=conditions, hue='Condition', ci=60,palette=color)
                    ax[count].legend([], [], frameon=False)
                    # ax[count].legend(loc='upper right', borderaxespad=0., fontsize='xx-small')
                    ax[count].grid(False)
                    # for tick in ax[count].xaxis.get_major_ticks():
                    #     tick.label.set_fontsize(8)
                    # for tick in ax[count].yaxis.get_major_ticks():
                    #     tick.label.set_fontsize(8)
                    ax[count].spines['top'].set_visible(False)
                    ax[count].spines['right'].set_visible(False)
                    ax[count].set_facecolor('none')
                    ax[count].set_xlabel('')
                    ax[count].set_ylabel(gene, fontdict=dict(weight='bold'))
                    labels = ax[count].get_xticklabels() + ax[count].get_yticklabels()
                    [label.set_fontweight('bold') for label in labels]
                    count += 1

                plt.tight_layout()
                fig.savefig(desfilepath + gene_list_choose + "_gene_expression_pooled_statistics" + ".png", format='png',
                            dpi=600)
                plt.close()
                writer.save()


if __name__ == '__main__':
    srcfilepath = r'Z:/ANALYSES/SPATIOSCALES- 10X genomics/Data/'  # main path
    Analysis = MEASeqX_Project(srcfilepath)
    for gene_list in column_list:
        Analysis.gene_expression(gene_list_name=gene_list,
                                 plot_gene_expression=True,
                                 select_plot='gene_list',
                                 top_gene_show=20,
                                 top_common_gene_show=20,
                                 plot_mutual_information=True,
                                 plot_UMIs=False,
                                 PCA_Clustering=False,
                                 value=0.9)  # select_plot = 'gene_list','select_genes','top_expressed_genes','top_expressed_common_genes'
    for gene_list in column_list:
        Analysis.mutual_information_statistics(gene_list_choose=gene_list) # individual statistics
        Analysis.mutual_information_pooled_statistics(gene_list_choose=gene_list) # pooled condition statistics (main path should contain the condition subfolders)
    Analysis.gene_expression_pooled_statistics(gene_list_choose='IEGs') # pooled condition statistics (main path should contain the condition subfolders)
