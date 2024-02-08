# -*- coding: utf-8 -*-
"""
Created on Aug 19 2021
@author:  BIONICS_LAB
@company: DZNE
"""
import matplotlib.pyplot as plt
import h5py
import numpy as np
import pandas as pd
import json
from ast import literal_eval
import os
import scipy.sparse as sp_sparse
import matplotlib.image as mpimg
import cv2
import help_functions.LFP_denosing as LFP_denosing
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

"""
Images for SRT and nEphys were captured on different microscopes at different resolutions.

The following input(s) are used to rescale the SRT H&E stained microscope slice image to the respective nEphys microscope slice image so that the regional strutucal clustering input is overlaid.

n‐Ephys electrode‐SRT spot matching is not one‐to‐one due to the difference in technology resolution so each SRT spot is assigned to the related n‐Ephys electrode(s) based on overlay.

"""

Pixel_SRT_mm = 0.645/1000 #H&E image was captured at a resolution of 1.550 pixel/1um -> 0.645um per pixel ->0.000645 mm (this value will change based on individual imaging settings)
length_nEphys = 2.67 #nEphys MEA active area in mm

SRT_diameter = 55 #SRT spot in um
nEphys_diameter = 21 # nEphys electrode diamater in um

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
        csv_file_cluster_name = 'Loupe Clusters.csv'
        csv_file_cluster_file, csv_file_cluster_Root = self.get_filename_path(self.srcfilepath, csv_file_cluster_name)
        for i in range(len(csv_file_cluster_file)):
            if csv_file_cluster_file[i][0] != '.':
                csv_file_cluster_root = csv_file_cluster_Root[i] + '/' + csv_file_cluster_file[i]
        #############################################
        csv_file_cluster = pd.read_csv(csv_file_cluster_root)

        return csv_file,tissue_lowres_scalef,features_name,matr_raw,barcodes,img,csv_file_cluster

    def Multiscale_Spatial_Alignment(self,plot_all =False,move_reference_name = 'Distal CA1',rotate_reference_name ='Proximal CA3',Pre_processing=True,Zooming = False,get_coordinates_relation = True):
        """
        Provide the transcriptomic and electrophysiologic profiles of the same cell assembly with spatial context. Performs automatic slice alignment using image resizing and rotation based on reference points.

            File input needed:
            -------
                - related files
                - 'SRT Reference Points.csv'
                - 'nEphys Reference Points.csv'

            Parameters
            ----------
            plot_all : Boolean
                The folder path.
            move_reference_name: string
                The file type(e.g. .bxr, .xlsx).
            plot_all : string
                The folder path.
            move_reference_name: string
                reference_name in 'nEphys Reference Points.csv'.
            rotate_reference_name : string
                rotate_reference_name in 'nEphys Reference Points.csv'.
            Pre_processing: Boolean
                Choose to flip or rotate the SRT iamge.
            Zooming: Boolean
                automatic zoom the ST iamge to match nEphys iamge.
            get_coordinates_relation: Boolean
                Choose to generate the final 'SRT_nEphys_Coordinates.xlsx".

            Returns
            -------

            File output:
            -------
                - 'nEphys_SRT_overlay.png'
                - 'SRT_nEphys_Coordinates.xlsx"
        """
        def clockwise_angle(v1, v2):
            x1, y1 = v1
            x2, y2 = v2
            dot = x1 * x2 + y1 * y2
            det = x1 * y2 - y1 * x2
            theta = np.arctan2(det, dot)
            theta = theta if theta > 0 else 2 * np.pi + theta
            if theta * 180 / np.pi > 180:
                angle = 360 - theta * 180 / np.pi
            else:
                angle = theta * 180 / np.pi
            return angle

        path = self.srcfilepath[:self.srcfilepath.rfind('/')]
        desfilepath = path + '/Multiscale_Overlay/'
        if not os.path.exists(desfilepath):
            os.mkdir(desfilepath)
        #############################
        filetype_bxr = '.bxr'
        filename_bxr, Root = self.get_filename_path(self.srcfilepath, filetype_bxr)
        for expFile in filename_bxr:
            if expFile[0] != '.':
                filehdf5_bxr = h5py.File(self.srcfilepath + expFile, 'r')  # read LFPs bxr files
                ChsGroups = np.asarray(filehdf5_bxr["3BUserInfo"]["ChsGroups"])
                if type(ChsGroups['Name'][0]) != str:
                    ChsGroups['Name'] = [i.decode("utf-8") for i in ChsGroups['Name']]
                MeaChs2ChIDsVector = np.asarray(filehdf5_bxr["3BResults"]["3BInfo"]["MeaChs2ChIDsVector"])
                ########################
                if os.path.exists(self.srcfilepath + expFile[:-4] + '_denosed_LfpChIDs' + '.npy') and os.path.exists(
                        self.srcfilepath + expFile[:-4] + '_denosed_LfpTimes' + '.npy'):
                    lfpChId_raw = np.load(self.srcfilepath + expFile[:-4] + '_denosed_LfpChIDs' + '.npy')
                    lfpTimes_raw = np.load(self.srcfilepath + expFile[:-4] + '_denosed_LfpTimes' + '.npy')
                else:
                    Analysis = LFP_denosing.LFPAnalysis_Function(self.srcfilepath,condition_choose='BS')  # condition_choose ='OB' or 'BS'
                    lfpChId_raw, lfpTimes_raw, LfpForms = Analysis.AnalyzeExp(expFile=expFile)


        colorMap = np.ones(4096)
        colorMap = [-i for i in colorMap]
        colorMap = np.array(colorMap)
        indexColor = 1
        clusters = []

        for Clu in self.clusters:
            for i in range(len(ChsGroups['Name'])):
                if ChsGroups['Name'][i] == Clu:
                    cluster_id = []
                    for j in range(len(ChsGroups['Chs'][i])):
                        # if ChsGroups['Chs'][i][j][1] - 1 + (ChsGroups['Chs'][i][j][0] - 1) * 64 in np.unique(lfpChId_raw):
                        colorMap[ChsGroups['Chs'][i][j][1] - 1 + (ChsGroups['Chs'][i][j][0] - 1) * 64] = indexColor
                        cluster_id.append(ChsGroups['Chs'][i][j][1] - 1 + (ChsGroups['Chs'][i][j][0] - 1) * 64)
                    clusters.append(ChsGroups['Name'][i])
                    indexColor += 1
        #########################
        Id = [i for i in range(len(colorMap)) if colorMap[i] > 0]

        new_Row = MeaChs2ChIDsVector["Col"] - 1
        new_Col = MeaChs2ChIDsVector["Row"] - 1
        coordinates = np.zeros((len(Id), 2), dtype=np.int)
        j = 0
        for i in Id:
            coordinates[j, :] = (new_Row[i], new_Col[i])
            j = j + 1
        denoisedEventRateMap_temp = [i for i in colorMap if i > 0]
        ################################################
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

        # fine the clusters
        csv_file_cluster_name = 'Loupe Clusters.csv'
        csv_file_cluster_file, csv_file_cluster_Root = self.get_filename_path(self.srcfilepath, csv_file_cluster_name)
        for i in range(len(csv_file_cluster_file)):
            if csv_file_cluster_file[i][0] != '.':
                csv_file_cluster_root = csv_file_cluster_Root[i] + '/' + csv_file_cluster_file[i]
        #############################################
        csv_file_cluster = pd.read_csv(csv_file_cluster_root)
        # print(list(csv_file_cluster.columns))
        barcode_cluster = np.asarray(csv_file_cluster["Barcode"])
        Loupe_cluster = np.asarray(csv_file_cluster["Loupe Clusters"])

        Cluster_list = self.clusters
        color_map = []

        for i in ix[0]:
            bar_code = csv_file['barcode'][i]
            try:
                clu = Loupe_cluster[list(barcode_cluster).index(bar_code)]
                for j in range(len(Cluster_list)):
                    if Cluster_list[j] == clu:
                        color_map.append(j)
            except:
                color_map.append(len(Cluster_list))

        ####################################################
        # fine the reference point
        nEphys_Reference_Points_name = 'nEphys Reference Points.csv'
        nEphys_Reference_Points_file, nEphys_Reference_Points_Root = self.get_filename_path(self.srcfilepath,nEphys_Reference_Points_name)
        for i in range(len(nEphys_Reference_Points_file)):
            if nEphys_Reference_Points_file[i][0] != '.':
                nEphys_Reference_Points_root = nEphys_Reference_Points_Root[i] + '/' + nEphys_Reference_Points_file[i]
        csv_nEphys_Reference_Points = pd.read_csv(nEphys_Reference_Points_root)
        Reference_Points_name_nEphys = csv_nEphys_Reference_Points['Reference Points']


        SRT_Reference_Points_name = 'SRT Reference Points.csv'
        SRT_Reference_Points_file, SRT_Reference_Points_Root = self.get_filename_path(self.srcfilepath,SRT_Reference_Points_name)
        for i in range(len(SRT_Reference_Points_file)):
            if SRT_Reference_Points_file[i][0] != '.':
                SRT_Reference_Points_root = SRT_Reference_Points_Root[i] + '/' + SRT_Reference_Points_file[i]
        csv_SRT_Reference_Points = pd.read_csv(SRT_Reference_Points_root)
        Reference_Points_name_SRT = csv_SRT_Reference_Points['Reference Points']

        reference_coordinate_SRT_x = []
        reference_coordinate_SRT_y = []
        for bar in csv_SRT_Reference_Points['Barcode']:
            indeces = list(csv_file['barcode']).index(bar)
            reference_coordinate_SRT_x.append((scatter_x[indeces] - extent[0])*Pixel_SRT_mm)
            reference_coordinate_SRT_y.append((scatter_y[indeces] - extent[2])*Pixel_SRT_mm)

        reference_coordinate_nEphys_x = []
        reference_coordinate_nEphys_y = []
        for bar in csv_nEphys_Reference_Points['Barcode']:
            related_cor = literal_eval(bar)
            reference_coordinate_nEphys_x.append((related_cor[1]-1)*length_nEphys/64)
            reference_coordinate_nEphys_y.append((related_cor[0]-1)*length_nEphys/64)

        x_raw = [i[0] * length_nEphys / 64 for i in coordinates]

        y_raw = [i[1] * length_nEphys / 64 for i in coordinates]
        x_norm = [i for i in scatter_x[ix] - extent[0]]
        y_norm = [i for i in scatter_y[ix] - extent[2]]
        csv_file, tissue_lowres_scalef, features_name, matr_raw, barcodes, img, csv_file_cluster = self.read_related_files()
        x_norm_raw = [i for i in np.asarray(csv_file["pixel_x"])[ix]* tissue_lowres_scalef - extent[0]]
        y_norm_raw = [i for i in np.asarray(csv_file["pixel_y"])[ix]* tissue_lowres_scalef - extent[2]]

        cm = plt.cm.get_cmap('Set1', len(self.clusters))
        color_map_filter_id = [i for i in range(len(color_map)) if color_map[i] != len(self.clusters)]
        x_norm = [x_norm[i] * Pixel_SRT_mm for i in color_map_filter_id]
        y_norm = [y_norm[i] * Pixel_SRT_mm for i in color_map_filter_id]

        color_map = [color_map[i] for i in color_map_filter_id]
        if Pre_processing==True:
            # y_norm = [max(y_norm) - i for i in y_norm]
            # reference_coordinate_SRT_y = [max(y_norm) - i for i in reference_coordinate_SRT_y]
            # #
            # x_norm = [max(x_norm) - i for i in x_norm]
            # reference_coordinate_SRT_x = [max(x_norm) - i for i in reference_coordinate_SRT_x]

            coordinate_flip = [[x_norm[i], y_norm[i]] for i in range(len(x_norm))]
            M_inv = cv2.getRotationMatrix2D(((max(x_norm)-min(x_norm))/2, (max(y_norm)-min(y_norm))/2), 90, 1) ###Change Degree
            ones = np.ones(shape=(len(coordinate_flip), 1))
            points_ones = np.hstack([coordinate_flip, ones])
            coordinate_flip = np.asarray(M_inv.dot(points_ones.T).T)
            x_norm = [i[0] for i in coordinate_flip]
            y_norm = [i[1] for i in coordinate_flip]


            reference_coordinate_flip = [[reference_coordinate_SRT_x[i], reference_coordinate_SRT_y[i]] for i in range(len(reference_coordinate_SRT_x))]
            ones = np.ones(shape=(len(reference_coordinate_flip), 1))
            points_ones = np.hstack([reference_coordinate_flip, ones])
            reference_coordinate_flip = np.asarray(M_inv.dot(points_ones.T).T)
            reference_coordinate_SRT_x = [i[0] for i in reference_coordinate_flip]
            reference_coordinate_SRT_y = [i[1] for i in reference_coordinate_flip]

        index_nEphys_rotate = list(Reference_Points_name_nEphys).index(rotate_reference_name)
        index_SRT_rotate = list(Reference_Points_name_SRT).index(rotate_reference_name)

        index_nEphys = list(Reference_Points_name_nEphys).index(move_reference_name)
        index_SRT = list(Reference_Points_name_SRT).index(move_reference_name)

        v1_SRT = [reference_coordinate_SRT_x[index_SRT], reference_coordinate_SRT_y[index_SRT]]
        v2_SRT = [reference_coordinate_SRT_x[index_SRT_rotate], reference_coordinate_SRT_y[index_SRT_rotate]]

        v1_nEphys = [reference_coordinate_nEphys_x[index_nEphys], reference_coordinate_nEphys_y[index_nEphys]]
        v2_nEphys = [reference_coordinate_nEphys_x[index_nEphys_rotate],reference_coordinate_nEphys_y[index_nEphys_rotate]]
        if Zooming == True:
            Zoom_percentage = np.sqrt(
                np.square(v1_SRT[0] - v2_SRT[0]) + np.square(v1_SRT[1] - v2_SRT[1]))/np.sqrt(
                np.square(v1_nEphys[0] - v2_nEphys[0]) + np.square(v1_nEphys[1] - v2_nEphys[1]))
            # Zoom_percentage =1/Zoom_percentage
            print('Zoom_percentage', Zoom_percentage)

            x_move = [i - ((max(x_norm) - min(x_norm)) / 2) for i in x_norm]
            y_move = [i - ((max(y_norm) - min(y_norm)) / 2) for i in y_norm]
            x_zoom = [i / Zoom_percentage for i in x_move]
            y_zoom = [i / Zoom_percentage for i in y_move]
            x_norm = [i + ((max(x_norm) - min(x_norm)) / 2) for i in x_zoom]
            y_norm = [i + ((max(y_norm) - min(y_norm)) / 2) for i in y_zoom]

            x_move = [i - ((max(x_norm) - min(x_norm)) / 2) for i in reference_coordinate_SRT_x]
            y_move = [i - ((max(y_norm) - min(y_norm)) / 2) for i in reference_coordinate_SRT_y]
            x_zoom = [i / Zoom_percentage for i in x_move]
            y_zoom = [i / Zoom_percentage for i in y_move]
            reference_coordinate_SRT_x = [i + ((max(x_norm) - min(x_norm)) / 2) for i in x_zoom]
            reference_coordinate_SRT_y = [i + ((max(y_norm) - min(y_norm)) / 2) for i in y_zoom]

        #############################
        ###################################move nEphys
        x_move_distance = reference_coordinate_SRT_x[index_SRT] - reference_coordinate_nEphys_x[index_nEphys]
        y_move_distance = reference_coordinate_SRT_y[index_SRT] - reference_coordinate_nEphys_y[index_nEphys]


        x_raw_move = np.asarray(x_raw) + x_move_distance
        y_raw_move = np.asarray(y_raw) + y_move_distance
        reference_coordinate_nEphys_x_move = np.asarray(reference_coordinate_nEphys_x) + x_move_distance
        reference_coordinate_nEphys_y_move = np.asarray(reference_coordinate_nEphys_y) + y_move_distance

        v1_SRT = [reference_coordinate_SRT_x[index_SRT], reference_coordinate_SRT_y[index_SRT]]
        v2_SRT = [reference_coordinate_SRT_x[index_SRT_rotate], reference_coordinate_SRT_y[index_SRT_rotate]]
        angle_SRT = clockwise_angle(v1_SRT, v2_SRT)
        v1_nEphys = [reference_coordinate_nEphys_x_move[index_nEphys], reference_coordinate_nEphys_y_move[index_nEphys]]
        v2_nEphys = [reference_coordinate_nEphys_x_move[index_nEphys_rotate],
                   reference_coordinate_nEphys_y_move[index_nEphys_rotate]]
        angle_nEphys = clockwise_angle(v1_nEphys, v2_nEphys)

        rotate_angle = angle_nEphys - angle_SRT

        reference_coordinate_nEphys_move = [[reference_coordinate_nEphys_x_move[i], reference_coordinate_nEphys_y_move[i]]
                                          for i in range(len(reference_coordinate_nEphys_x_move))]
        coordinate_move = [[x_raw_move[i], y_raw_move[i]] for i in range(len(x_raw_move))]


        M_inv = cv2.getRotationMatrix2D((v1_nEphys[0], v1_nEphys[1]), rotate_angle, 1)
        # add ones
        ones = np.ones(shape=(len(coordinate_move), 1))

        points_ones = np.hstack([coordinate_move, ones])
        # transform points
        coordinate_move_rotate = np.asarray(M_inv.dot(points_ones.T).T)

        ones_reference = np.ones(shape=(len(reference_coordinate_nEphys_move), 1))
        points_ones_reference = np.hstack([reference_coordinate_nEphys_move, ones_reference])
        # transform points
        reference_coordinate_nEphys_move_rotate = np.asarray(M_inv.dot(points_ones_reference.T).T)
        if plot_all == True:
            #############find related coordinate
            fig = plt.figure()
            ax = fig.add_subplot(221)
            ax.scatter(x_norm, y_norm, c=color_map, cmap=cm, marker='o', s=5/2, alpha=1)
            ax.scatter(reference_coordinate_SRT_x, reference_coordinate_SRT_y, c='black', marker='s', s=5/2, alpha=1)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            x_ticks_raw = np.linspace(min(x_norm), max(x_norm), 5, endpoint=True)
            x_ticks = [str(round(i-min(x_ticks_raw), 2)) for i in x_ticks_raw]
            ax.set_xticks(x_ticks_raw)
            ax.set_xticklabels(x_ticks)
            ax.set_xlabel('Length(mm)', fontsize=8)
            y_ticks_raw = np.linspace(min(y_norm), max(y_norm), 5, endpoint=True)
            y_ticks = [str(round(i-min(y_ticks_raw), 2)) for i in y_ticks_raw]
            ax.set_yticks(y_ticks_raw)
            ax.set_yticklabels(y_ticks)
            ax.set_ylabel('Length(mm)', fontsize=8)
            ax.set_title('SRT', fontsize=8)
            ax.set_aspect('equal', 'box')

            ax = fig.add_subplot(223)
            ax.scatter(x_norm, y_norm, c=color_map, cmap=cm, marker='o', s=5/2, alpha=1)
            ax.scatter(reference_coordinate_SRT_x, reference_coordinate_SRT_y, c='black', marker='s', s=5/2, alpha=1)
            # ax.scatter(x_raw, y_raw, marker='o', alpha=0.5, c=denoisedEventRateMap_temp, s=2/2, cmap=cm)
            # ax.scatter(reference_coordinate_nEphys_x, reference_coordinate_nEphys_y, marker='s', alpha=0.5, c='black', s=2/2)

            ax.scatter(x_raw_move, y_raw_move, marker='o', alpha=0.5, c=denoisedEventRateMap_temp, s=2 / 2, cmap=cm)
            ax.scatter(reference_coordinate_nEphys_x_move, reference_coordinate_nEphys_y_move, marker='s', alpha=0.5, c='black',s=2 / 2)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ############################
            x_ticks_raw = np.linspace(min(x_norm), max(x_norm), 5, endpoint=True)
            x_ticks = [str(round(i-min(x_ticks_raw), 2)) for i in x_ticks_raw]
            ax.set_xticks(x_ticks_raw)
            ax.set_xticklabels(x_ticks)
            ax.set_xlabel('Length(mm)', fontsize=8)
            y_ticks_raw = np.linspace(min(y_norm), max(y_norm), 5, endpoint=True)
            y_ticks = [str(round(i-min(y_ticks_raw), 2)) for i in y_ticks_raw]
            ax.set_yticks(y_ticks_raw)
            ax.set_yticklabels(y_ticks)
            ax.set_ylabel('Length(mm)', fontsize=8)
            ax.set_title('Step 1:Moving coordinates based on '+ move_reference_name, fontsize=8)
            ax.set_aspect('equal', 'box')

            ax = fig.add_subplot(222)
            ax.scatter(x_raw, y_raw, marker='o', alpha=1, c=denoisedEventRateMap_temp, s=2/2, cmap=cm)
            ax.scatter(reference_coordinate_nEphys_x, reference_coordinate_nEphys_y, marker='s', alpha=1, c='black', s=2/2)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            #############################
            x_ticks_raw = np.linspace(min(x_norm), max(x_norm), 5, endpoint=True)
            x_ticks = [str(round(i-min(x_ticks_raw), 2)) for i in x_ticks_raw]
            ax.set_xticks(x_ticks_raw)
            ax.set_xticklabels(x_ticks)
            ax.set_xlabel('Length(mm)', fontsize=8)
            y_ticks_raw = np.linspace(min(y_norm), max(y_norm), 5, endpoint=True)
            y_ticks = [str(round(i-min(y_ticks_raw), 2)) for i in y_ticks_raw]
            ax.set_yticks(y_ticks_raw)
            ax.set_yticklabels(y_ticks)
            ax.set_ylabel('Length(mm)', fontsize=8)


            ###################################
            # x_ticks_raw = np.linspace(0,length_nEphys, 5, endpoint=True)
            # x_ticks = [str(round(i, 2)) for i in x_ticks_raw]
            # ax.set_xticks(x_ticks_raw)
            # ax.set_xticklabels(x_ticks)
            # ax.set_xlabel('Length(mm)')
            # y_ticks_raw = np.linspace(0,length_nEphys, 5, endpoint=True)
            # y_ticks = [str(round(i, 2)) for i in y_ticks_raw]
            # ax.set_yticks(y_ticks_raw)
            # ax.set_yticklabels(y_ticks)
            # ax.set_ylabel('Length(mm)')
            ax.set_title('nEphys', fontsize=8)
            ax.set_aspect('equal', 'box')

            ax = fig.add_subplot(224)
            ax.scatter(x_norm, y_norm, c=color_map, cmap=cm, marker='o', s=5 / 2, alpha=1)
            ax.scatter(reference_coordinate_SRT_x, reference_coordinate_SRT_y, c='black', marker='s', s=5 / 2, alpha=1)
            # ax.scatter(x_raw_move, y_raw_move, marker='o', alpha=0.5, c=denoisedEventRateMap_temp, s=2 / 2, cmap=cm)
            # ax.scatter(reference_coordinate_nEphys_x_move, reference_coordinate_nEphys_y_move, marker='s', alpha=0.5, c='black',s=2 / 2)

            ax.scatter(coordinate_move_rotate[:,0], coordinate_move_rotate[:,1], marker='o', alpha=0.5, c=denoisedEventRateMap_temp, s=2 / 2, cmap=cm)
            ax.scatter(reference_coordinate_nEphys_move_rotate[:,0], reference_coordinate_nEphys_move_rotate[:,1], marker='s', alpha=0.5,c='black', s=2 / 2)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            x_ticks_raw = np.linspace(min(min(x_norm),min(x_raw_move)), max(max(x_norm),max(x_raw_move)), 5, endpoint=True)
            x_ticks = [str(round(i, 1)) for i in x_ticks_raw]
            ax.set_xticks(x_ticks_raw)
            ax.set_xticklabels(x_ticks)
            ax.set_xlabel('Length(mm)', fontsize=8)
            y_ticks_raw = np.linspace(min(min(y_norm),min(y_raw_move)), max(max(y_norm),max(y_raw_move)), 5, endpoint=True)
            y_ticks = [str(round(i, 1)) for i in y_ticks_raw]
            ax.set_yticks(y_ticks_raw)
            ax.set_yticklabels(y_ticks)
            ax.set_ylabel('Length(mm)', fontsize=8)
            ax.set_title('Step 2:Rotation angle based on '+ rotate_reference_name, fontsize=8)
            ax.set_aspect('equal', 'box')

            fig.tight_layout()
            colorMapTitle_SRT = 'nEphys_SRT_overlay'
            fig.savefig(desfilepath + colorMapTitle_SRT + ".png", format='png', dpi=600)
            plt.close()


        if get_coordinates_relation == True:
            #x_norm, y_norm, color_map,x_norm_raw,y_norm_raw, barcode_raw[ix]
            #coordinate_move_rotate[:,0], coordinate_move_rotate[:,1],coordinates
            barcodes_all = []
            gene_coordinates = []
            related_coordinates_COMS = []
            cluster_all = []
            search_radius = (SRT_diameter / 2)   # mm
            nEphys_reach_radius = (nEphys_diameter/ 2)
            for i in range(len(x_norm)):
                related_nEphys_coordinate = []
                ST_area_x = [x_norm[i]*1000 - search_radius, x_norm[i]*1000 + search_radius]
                ST_area_y = [y_norm[i]*1000 - search_radius, y_norm[i]*1000 + search_radius]

                for j in range(len(coordinate_move_rotate)):

                    nEphys_area_x = [coordinate_move_rotate[j][0]*1000 - nEphys_reach_radius, coordinate_move_rotate[j][0]*1000 + nEphys_reach_radius]
                    nEphys_area_y = [coordinate_move_rotate[j][1]*1000 - nEphys_reach_radius, coordinate_move_rotate[j][1]*1000 + nEphys_reach_radius]

                    if nEphys_area_x[1]<ST_area_x[0] or nEphys_area_x[0]>ST_area_x[1] or nEphys_area_y[1]<ST_area_y[0] or nEphys_area_y[0]> ST_area_y[1]:
                        pass
                    else:
                        related_nEphys_coordinate.append(list(coordinates[j]))

                # if len(related_nEphys_coordinate)>0:
                barcodes_all.append(barcode_raw[ix][i])
                gene_coordinates.append([x_norm_raw[i],y_norm_raw[i]])
                related_coordinates_COMS.append(related_nEphys_coordinate)
                cluster_all.append(self.clusters[color_map[i]])
            a = {'Barcodes': barcodes_all, 'Coordinates in SRT': gene_coordinates,
                 "Coordinates in nEphys": related_coordinates_COMS, "Cluster": cluster_all}
            df = pd.DataFrame.from_dict(a, orient='index').T

            df.to_excel(self.srcfilepath + 'SRT_nEphys_Coordinates' + ".xlsx", index=False)
            # pass


if __name__ == '__main__':
    srcfilepath = r'Z:/ANALYSES/SPATIOSCALES- 10X genomics/Nature Communications Data Report/Data/Test2/'  # main path
    Analysis = MEASeqX_Project(srcfilepath)

    Analysis.Multiscale_Spatial_Alignment(plot_all=True, move_reference_name='Top DG', rotate_reference_name='Right DG',
                                Pre_processing=True,
                                get_coordinates_relation=True)  # 'Proximal CA3','Left DG','Distal CA1','Top DG','Right DG'#########Step 2

