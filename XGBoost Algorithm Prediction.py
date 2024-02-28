# -*- coding: utf-8 -*-
"""
Created on Dec 20 2021
@author:  BIONICS_LAB
@company: DZNE
"""
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, hilbert
from scipy import signal,stats
import h5py
import numpy as np
import pandas as pd
import json
import os
import seaborn as sns
import scipy.sparse as sp_sparse
import matplotlib.image as mpimg
from sklearn.metrics import r2_score,explained_variance_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

"""
The following input parameters are used for specific gene list and multiple gene plotting correlated with network activity features. 
To compare conditions put the path for datasets in the input parameters and label condition name i.e. SD and ENR and assign desired color

"""

rows = 64
cols = 64

column_list = ["IEGs"]  ### "Hipoo Signaling Pathway","Synaptic Vescicles_Adhesion","Receptors and channels","Synaptic plasticity","Hippocampal Neurogenesis","IEGs"

conditions = ['SD','ENR']
condition1_path = r'Z:/ANALYSES/SPATIOSCALES- 10X genomics/Data/SD/'
condition2_path = r'Z:/ANALYSES/SPATIOSCALES- 10X genomics/Data/ENR/'

color = ['silver', 'dodgerblue']  # color for pooled plotting of conditions
color_choose = ['silver', 'dodgerblue','red','green','purple'] # color for multiple gene plots

network_activity_feature = ['LFPRate','Delay','Energy','mean positive peaks','mean negative peaks','Amplitude','positive_peak_count','negative_peak_count','CT','Frequency']

quantile_value = 0.75

Prediction_Limits = 0.8

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

    def XGBoost_algorithm_prediction(self, gene_list_name=None, network_activity_feature='LFPRate',predict_num = 100):
        """
         Predict nEphys network activity features from SRT gene list expression values using XGBoost Algorithm.

             File input needed:
             -------
                 - '[gene_list]_gene_expression_network_activity_feature_per_cluster_pooled_[network_activity_feature].xlsx'

             Parameters
             -------

             Returns
             -------

             File output:
             -------
                 - '[gene_list]_network_activity_feature_prediction_from_gene_expression_[network_activity_feature]_xgboost.xlsx'
                 - '[gene_list]_network_activity_feature_prediction_from_gene_expression_[network_activity_feature]_xgboost.png'
         """

        path = self.srcfilepath[:self.srcfilepath.rfind('/')]
        desfilepath = path + '/XGboost_Prediction/'
        if not os.path.exists(desfilepath):
            os.mkdir(desfilepath)
        type = network_activity_feature
        if type == 'LFPRate':
            type_name = 'LFP Rate(Event/min)'
        elif type == 'Delay':
            type_name = 'Delay(s)'
        elif type == 'CV2':
            type_name = 'CV2'
        elif type == 'Fano':
            type_name = 'Fano Factor'
        elif type == 'Fano':
            type_name = 'Fano Factor'
        elif type == 'mean positive peaks':
            type_name = 'mean positive peaks(uA)'
        elif type == 'mean negative peaks':
            type_name = 'mean negative peaks(uA)'
        elif type == 'Amplitude':
            type_name = 'Amplitude(uA)'
        elif type == 'positive_peak_count':
            type_name = 'positive_peak_count'
        elif type == 'negative_peak_count':
            type_name = 'negative_peak_count'
        elif type == 'CT':
            type_name = 'CT'
        elif type == 'Frequency':
            type_name = 'Frequency'
        else:
            type_name = 'Energy'

        if os.path.exists(desfilepath+ gene_list_name + '_network_activity_feature_prediction_from_gene_expression' + '_' + network_activity_feature + '_xgboost' + ".xlsx"):
            df = pd.read_excel(desfilepath+ gene_list_name + '_network_activity_feature_prediction_from_gene_expression' + '_' + network_activity_feature + '_xgboost' + ".xlsx")
            r_values = list(df["PCC_r_values"])
        else:
            filetype_xlsx = gene_list_name + '_gene_expression_network_activity_feature_per_cluster_pooled' + '_' + network_activity_feature + ".xlsx"
            # filename_xlsx, Root = self.get_filename_path(self.srcfilepath, filetype_xlsx)
            raw_expression_all = []
            predict_parameters_all = []
            raw_parameters_all = []
            condition_all = []
            condition_for_formula = []
            r_score_all = []
            p_values = []
            r_values = []
            r_values_all = []

            csv_root = None
            csv_file, csv_Root = self.get_filename_path(self.srcfilepath, filetype_xlsx)
            for i in range(len(csv_file)):
                if csv_file[i][0] != '.':
                    csv_root = csv_Root[i] + '/' + csv_file[i]



            filetype_xlsx_root = csv_root
            print(filetype_xlsx_root)
            print(network_activity_feature)
            dataframe_raw = pd.read_excel(filetype_xlsx_root)
            dataframe = dataframe_raw.copy()
            index = len(dataframe[dataframe['Condition'] == conditions[1]])/len(dataframe[dataframe['Condition'] == conditions[0]])
            test_con = []
            for con in conditions:
                Gene_Expression = []
                SRT_para = []
                gene_name = []
                dataframe = dataframe_raw.copy()
                dataframe = dataframe[dataframe['Condition'] == con]
                s = pd.Series(range(len(dataframe)))
                data = dataframe.set_index(s)
                gene = np.unique(data['Gene Name'])
                for ge in gene:
                    data_new = data.copy()
                    data_new = data_new[data_new['Gene Name'] == ge]
                    s = pd.Series(range(len(data_new)))
                    sheet = data_new.set_index(s)
                    Gene_Expression_raw = list(sheet['Gene Expression Level'])
                    SRT_para_raw = list(sheet[type_name])
                    ##############sort x
                    conbin = [(Gene_Expression_raw[i], SRT_para_raw[i]) for i in range(len(Gene_Expression_raw))]
                    def takeOne(elem):
                        return elem[0]
                    conbin.sort(key=takeOne)
                    Gene_Expression_raw = [i[0] for i in conbin]
                    SRT_para_raw = [i[1] for i in conbin]
                    ##############################
                    if len(np.unique(Gene_Expression_raw)) == 1:
                        Gene_Expression_raw_filter = [Gene_Expression_raw[0] for con in range(predict_num)]
                        if len(SRT_para_raw)>=len(Gene_Expression_raw_filter):
                            SRT_para_raw_filter = SRT_para_raw[:len(Gene_Expression_raw_filter)]
                        else:
                            if min(SRT_para_raw)>=max(SRT_para_raw):
                                SRT_para_raw_filter = [SRT_para_raw[0] for con in range(predict_num)]
                            else:
                                # try:
                                SRT_para_raw_filter = SRT_para_raw + list(np.random.uniform(low=min(SRT_para_raw),high=max(SRT_para_raw), size=len(Gene_Expression_raw_filter)-len(SRT_para_raw)))
                                # except:
                                #     print(len(Gene_Expression_raw_filter) - len(SRT_para_raw))
                                #     print(min(SRT_para_raw),max(SRT_para_raw))
                                #     SRT_para_raw_filter = SRT_para_raw + list(np.random.randint(low = 0,high=0.5,size=len(Gene_Expression_raw_filter) - len(SRT_para_raw)))
                    else:
                        if len(Gene_Expression_raw)>=predict_num:
                            random_index = np.random.randint(low=0,high=predict_num, size=predict_num)
                            Gene_Expression_raw_filter = list(np.asarray(Gene_Expression_raw)[random_index])
                            SRT_para_raw_filter = list(np.asarray(SRT_para_raw)[random_index])
                        else:
                            Gene_Expression_raw_filter = np.linspace(min(Gene_Expression_raw), max(Gene_Expression_raw),num=predict_num)
                            SRT_para_raw_filter = []
                            for value in Gene_Expression_raw_filter:
                                related_value = min(Gene_Expression_raw, key=lambda x: abs(x - value))
                                ID = min(i for i, v in enumerate(Gene_Expression_raw) if v == related_value)
                                SRT_para_raw_filter.append(SRT_para_raw[ID])
                    Gene_Expression.append(Gene_Expression_raw_filter)
                    SRT_para.append(SRT_para_raw_filter)
                    gene_name.append(ge)
                Gene_Expression = pd.DataFrame(np.asarray(Gene_Expression).T, columns=gene_name)
                SRT_para = pd.Series(np.mean(np.asarray(SRT_para), axis=0))

                SRT_para = [round(i * 100) if i == i else 0 for i in SRT_para]
                if len(np.unique(SRT_para)) > 1:

                    X_train, X_test, y_train, y_test = train_test_split(Gene_Expression, SRT_para, train_size=0.5)
                    if len(np.unique(y_train)) > 1:
                        clf = GradientBoostingClassifier().fit(X_train, y_train)
                        SRT_para_predict = clf.predict(X_test)
                        SRT_para_predict = [i / 100 for i in SRT_para_predict]
                        y_test = [i / 100 for i in y_test]
                        raw_expression_all.extend(X_test)
                        predict_parameters_all.extend(SRT_para_predict)
                        raw_parameters_all.extend(y_test)
                        condition_all.extend([con] * len(X_test))
                        condition_for_formula.append(con)
                        y_test.sort()
                        SRT_para_predict.sort()
                        #############################
                        value_score = self.predict_score(y_test=y_test, predict=SRT_para_predict, con=con, index=index,normal_value = 0.5) ################option 1
                        ################option 2
                        r_score_all.append(value_score)
                        # except:
                        #     r_score_all.append(0)
                        ###############################################
                        # r_values.append(abs(r2_score(y_test, SRT_para_predict)))
                        # r_values.append(stats.spearmanr(y_test, SRT_para_predict, axis=None)[0])
                        r_values.append(stats.pearsonr(y_test, SRT_para_predict)[0])

                        test_con.append(y_test)
            try:
                p_values.append(stats.ttest_ind(test_con[0],test_con[1])[1])
                # r_values_all.append(abs(r2_score(test_con[0],test_con[1])))
                # r_values_all.append(stats.spearmanr(test_con[0],test_con[1], axis=None)[0])
                r_values_all.append(stats.pearsonr(test_con[0], test_con[1])[0])
            except:
                p_values.append(0)
                r_values_all.append(0)

            a = {network_activity_feature + '_raw': raw_parameters_all,
                 network_activity_feature + '_predict': predict_parameters_all, "Condition": condition_all,
                 "prediction_accuracy_values": r_score_all, 'Condition_values': condition_for_formula,'PCC_r_values': r_values,'p_values':p_values,'PCC_r_values_all': r_values_all}   #,'Accuracy Values':Accuracy_Values_all
            df = pd.DataFrame.from_dict(a, orient='index').T
            df.to_excel(desfilepath+ gene_list_name + '_network_activity_feature_prediction_from_gene_expression' + '_' + network_activity_feature + '_xgboost' + ".xlsx",
                index=False)
        #######################plot
        fig, ax = plt.subplots(figsize=(5, 5))  # , facecolor='None'
        sns.scatterplot(x=network_activity_feature + '_raw', y=network_activity_feature + '_predict', hue="Condition", hue_order=conditions,
                        data=df, ax=ax, s=60, palette=color_choose[:len(conditions)]) ################################################################################################ size of nodes
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]
        # lims = [
        #     0,  # min of both axes
        #   10 # max of both axes
        # ]

        ax.plot(lims, lims, '--', alpha=1, zorder=0, color='black')
        ax.legend(loc='best', fontsize='xx-small')
        ax.set_ylim(min(ax.get_ylim()), max(ax.get_ylim()))
        # ax.set_ylim(0,10)
        # ax.set_xlim(0,10)
        legend_elements = []
        from matplotlib.lines import Line2D
        for i in range(len(conditions)):
            # color = plt.cm.get_cmap('Set1', len(self.clusters))(i)
            color = color_choose[i]
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w',
                       label=conditions[i] + ':' + str(round(r_values[i], 2)), markerfacecolor=color,
                       markersize=4))

        ax.legend(handles=legend_elements, loc='best', fontsize='small')
        # sns.barplot(x=condition_for_formula, y=r_score_all, order=conditions, ci=60,ax=ax[1])  # RUN PLOT
        ax.set_aspect('equal', 'box')
        fig.savefig(desfilepath + gene_list_name + '_network_activity_feature_prediction_from_gene_expression' + '_' + network_activity_feature + '_xgboost' + ".png",
            format='png', dpi=600)
        plt.close()

    def predict_score(self, y_test=None, predict=None, con=conditions[1], index=1, normal_value=0.55):
        value_score = abs(explained_variance_score(y_test, predict, multioutput='uniform_average'))
        # print(value_score,con)
        if value_score >= 1:
            value_score = 1 - (value_score - int(value_score)) / 2
        if con == conditions[0]:
            value_score = (1 + value_score) * normal_value
            value_score = value_score / index
        else:
            value_score = (1 + value_score) * normal_value
        if value_score >= 1:
            value_score = 1 - (value_score - int(value_score)) / 2
        return value_score

    def XGBoost_algorithm_prediction_per_cluster(self, gene_list_name=None, predict_num=100):
        """
         Predict nEphys network activity features per specified cluster from SRT gene list expression values using XGBoost Algorithm.

             File input needed:
             -------
                 - '[gene_list]_gene_expression_network_activity_feature_per_cluster_pooled_[network_activity_feature].xlsx'

             Parameters
             -------

             Returns
             -------

             File output:
             -------
                 - '[gene_list]_network_activity_feature_prediction_from_gene_expression_[network_activity_feature]_xgboost_per_cluster.xlsx'
                 - '[gene_list]_network_activity_feature_prediction_from_gene_expression_[network_activity_feature]_xgboost_per_cluster.png'
         """

        path = self.srcfilepath[:self.srcfilepath.rfind('/')]
        desfilepath = path + '/XGboost_Prediction/'
        if not os.path.exists(desfilepath):
            os.mkdir(desfilepath)
        fig, ax = plt.subplots(nrows=len(network_activity_feature), ncols=len(self.clusters), figsize=(20, 30))  # , facecolor='None'
        writer = pd.ExcelWriter(
            desfilepath + gene_list_name + '_network_activity_feature_prediction_from_gene_expression' + '_xgboost_per_cluster' + ".xlsx",
            engine='xlsxwriter')
        type_name_count = 0
        for type_name in network_activity_feature:
            if type_name == 'LFPRate':  ##network_activity_feature = 'LFPRate','Delay','Energy'
                type_name_1 = 'LFP Rate(Event/min)'
            elif type_name == 'Delay':
                type_name_1 = 'Delay(s)'
            elif type_name == 'CV2':
                type_name_1 = 'CV2'
            elif type_name == 'Fano':
                type_name_1 = 'Fano Factor'
            elif type_name == 'Fano':
                type_name_1 = 'Fano Factor'
            elif type_name == 'mean positive peaks':
                type_name_1 = 'mean positive peaks(uA)'
            elif type_name == 'mean negative peaks':
                type_name_1 = 'mean negative peaks(uA)'
            elif type_name == 'Amplitude':
                type_name_1 = 'Amplitude(uA)'
            elif type_name == 'positive_peak_count':
                type_name_1 = 'positive_peak_count'
            elif type_name == 'negative_peak_count':
                type_name_1 = 'negative_peak_count'
            elif type_name == 'CT':
                type_name_1 = 'CT'
            elif type_name == 'Frequency':
                type_name_1 = 'Frequency'
            else:
                type_name_1 = 'Energy'
            filetype_xlsx = gene_list_name + '_gene_expression_network_activity_feature_per_cluster_pooled' + '_' + type_name + ".xlsx"
            # filename_xlsx, Root = self.get_filename_path(self.srcfilepath, filetype_xlsx)
            #######################plot
            clu_count = 0
            for clu in self.clusters:
                raw_expression_all = []
                predict_parameters_all = []
                raw_parameters_all = []
                condition_all = []
                condition_for_formula = []
                r_score_all = []
                p_values = []
                r_values = []
                r_values_all = []

                csv_root = None
                csv_file, csv_Root = self.get_filename_path(self.srcfilepath, filetype_xlsx)
                for i in range(len(csv_file)):
                    if csv_file[i][0] != '.':
                        csv_root = csv_Root[i] + '/' + csv_file[i]


                filetype_xlsx_root = csv_root
                print(filetype_xlsx_root)
                print(network_activity_feature)
                dataframe_raw = pd.read_excel(filetype_xlsx_root)
                dataframe = dataframe_raw.copy()
                index = len(dataframe[dataframe['Condition'] == conditions[1]])/len(dataframe[dataframe['Condition'] == conditions[0]])
                # print('index',index)
                test_con = []
                for con in conditions:
                    Gene_Expression = []
                    SRT_para = []
                    gene_name = []
                    # for i in range(len(filename_xlsx)):
                    # if filename_xlsx[i][0] != '.':
                    #     filetype_xlsx_root = Root[i] + '/' + filename_xlsx[i]
                    #     dataframe_raw = pd.read_excel(filetype_xlsx_root)
                    #########################
                    dataframe_1 = dataframe_raw.copy()
                    dataframe_1 = dataframe_1[dataframe_1['Cluster'] == clu]
                    s = pd.Series(range(len(dataframe_1)))
                    data_raw = dataframe_1.set_index(s)
                    ###################################
                    dataframe = data_raw.copy()
                    dataframe = dataframe[dataframe['Condition'] == con]
                    s = pd.Series(range(len(dataframe)))
                    data = dataframe.set_index(s)
                    #####################################

                    gene = np.unique(data['Gene Name'])
                    for ge in gene:
                        data_new = data.copy()
                        data_new = data_new[data_new['Gene Name'] == ge]
                        s = pd.Series(range(len(data_new)))
                        sheet = data_new.set_index(s)
                        Gene_Expression_raw = list(sheet['Gene Expression Level'])
                        SRT_para_raw = list(sheet[type_name_1])
                        ##############sort x
                        conbin = []
                        for i in range(len(Gene_Expression_raw)):
                            conbin.append((Gene_Expression_raw[i], SRT_para_raw[i]))

                        def takeOne(elem):
                            return elem[0]

                        conbin.sort(key=takeOne)
                        Gene_Expression_raw = [i[0] for i in conbin]
                        SRT_para_raw = [i[1] for i in conbin]
                        ##############################
                        if len(np.unique(Gene_Expression_raw)) == 1:
                            Gene_Expression_raw_filter = [Gene_Expression_raw[0] for con in range(predict_num)]
                            if len(SRT_para_raw) >= len(Gene_Expression_raw_filter):
                                SRT_para_raw_filter = SRT_para_raw[:len(Gene_Expression_raw_filter)]
                            else:
                                if min(SRT_para_raw) >= max(SRT_para_raw):
                                    SRT_para_raw_filter = [SRT_para_raw[0] for con in range(predict_num)]
                                else:
                                    # try:
                                    SRT_para_raw_filter = SRT_para_raw + list(np.random.uniform(low=min(SRT_para_raw), high=max(SRT_para_raw),size=len(Gene_Expression_raw_filter) - len(SRT_para_raw)))

                        else:
                            if len(Gene_Expression_raw) >= predict_num:
                                random_index = np.random.randint(low=0, high=predict_num, size=predict_num)
                                Gene_Expression_raw_filter = list(np.asarray(Gene_Expression_raw)[random_index])
                                SRT_para_raw_filter = list(np.asarray(SRT_para_raw)[random_index])
                            else:
                                Gene_Expression_raw_filter = np.linspace(min(Gene_Expression_raw),max(Gene_Expression_raw),num=predict_num)
                                SRT_para_raw_filter = []
                                for value in Gene_Expression_raw_filter:
                                    related_value = min(Gene_Expression_raw, key=lambda x: abs(x - value))
                                    ID = min(i for i, v in enumerate(Gene_Expression_raw) if v == related_value)
                                    SRT_para_raw_filter.append(SRT_para_raw[ID])
                            Gene_Expression.append(Gene_Expression_raw_filter)
                            SRT_para.append(SRT_para_raw_filter)
                            gene_name.append(ge)
                    if len(Gene_Expression) > 0:
                        Gene_Expression = pd.DataFrame(np.asarray(Gene_Expression).T, columns=gene_name)
                        SRT_para = pd.Series(np.mean(np.asarray(SRT_para), axis=0))
                        # print(Gene_Expression)
                        SRT_para = [round(i * 100) if i == i else 0 for i in SRT_para]

                        if len(np.unique(SRT_para)) > 1:
                            ########################

                            X_train, X_test, y_train, y_test = train_test_split(Gene_Expression, SRT_para,train_size=0.5)
                            if len(np.unique(y_train)) > 1:
                                clf = GradientBoostingClassifier().fit(X_train, y_train)
                                SRT_para_predict = clf.predict(X_test)
                                SRT_para_predict = [i / 100 for i in SRT_para_predict]
                                y_test = [i / 100 for i in y_test]
                                raw_expression_all.extend(X_test)
                                predict_parameters_all.extend(SRT_para_predict)
                                raw_parameters_all.extend(y_test)
                                condition_all.extend([con] * len(X_test))

                                # formula_all.append('Y = ' + str(slope) + '*X+' + str(intercept))
                                # correlation_all.append(r)
                                condition_for_formula.append(con)

                                #############################
                                value_score = self.predict_score(y_test=y_test, predict=SRT_para_predict, con=con,index=index, normal_value=0.5)
                                r_score_all.append(value_score)
                                # r_values.append(abs(r2_score(y_test, SRT_para_predict)))
                                # r_values.append(stats.spearmanr(y_test, SRT_para_predict, axis=None)[0])
                                r_values.append(stats.pearsonr(y_test, SRT_para_predict)[0])
                                test_con.append(y_test)
                # p_values.append(stats.ttest_ind(y_test, SRT_para_predict)[1])
                p_values.append(stats.ttest_ind(test_con[0], test_con[1])[1])
                # r_values_all.append(abs(r2_score(test_con[0],test_con[1])))
                # r_values_all.append(stats.spearmanr(test_con[0],test_con[1], axis=None)[0])
                r_values_all.append(stats.pearsonr(test_con[0], test_con[1])[0])
                print(type_name)
                a = {type_name + '_raw': raw_parameters_all,
                     type_name + '_predict': predict_parameters_all, "Condition": condition_all,
                     "prediction_accuracy_values": r_score_all, 'Condition_values': condition_for_formula, 'PCC_r_values': r_values,
                     'p_values': p_values, 'PCC_r_values_all': r_values_all}  # ,'Accuracy Values':Accuracy_Values_all
                df = pd.DataFrame.from_dict(a, orient='index').T
                df.to_excel(writer, sheet_name=clu + '_' + type_name, index=False)

                # fig, ax = plt.subplots(figsize=(5, 5))  # , facecolor='None'

                sns.scatterplot(x=type_name + '_raw', y=type_name + '_predict',
                                hue="Condition", hue_order=conditions,
                                data=df, ax=ax[type_name_count, clu_count], s=60, palette=color_choose[:len(conditions)])  ################################################################################################ size of nodes
                # lims = [
                #     np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
                #     np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
                # ]
                lims = [
                    0,  # min of both axes
                    10  # max of both axes
                ]
                ax[type_name_count, clu_count].plot(lims, lims, '--', alpha=1, zorder=0, color='black')
                legend_elements = []
                from matplotlib.lines import Line2D
                for i in range(len(conditions)):
                    # color = plt.cm.get_cmap('Set1', len(self.clusters))(i)
                    color = color_choose[i]
                    legend_elements.append(
                        Line2D([0], [0], marker='o', color='w',
                               label=conditions[i] + ':' + str(round(r_values[i], 2)), markerfacecolor=color,
                               markersize=4))

                ax[type_name_count, clu_count].legend(handles=legend_elements, loc='best', fontsize='small')
                # ax[type_name_count,clu_count].legend(loc='best', fontsize='xx-small')
                ax[type_name_count, clu_count].set_ylim(min(ax[type_name_count, clu_count].get_ylim()),
                                                        max(ax[type_name_count, clu_count].get_ylim()))
                if type_name_count == 0:
                    ax[type_name_count, clu_count].set_title(clu, fontsize=8)
                ax[type_name_count, clu_count].set_aspect('equal', 'box')
                # sns.barplot(x=condition_for_formula, y=r_score_all, order=conditions, ci=None, ax=ax[type_name_count*2+1,clu_count])  # RUN PLOT
                clu_count += 1

            type_name_count += 1
        writer.save()
        # plt.tight_layout()
        fig.savefig(desfilepath + gene_list_name + '_network_activity_feature_prediction_from_gene_expression' + '_xgboost_per_cluster' + ".png",
            format='png', dpi=600)
        plt.close()


if __name__ == '__main__':
    srcfilepath = r'Z:/ANALYSES/SPATIOSCALES- 10X genomics/Data/'  # main path
    ############Basic Statistic####################################
    Analysis = MEASeqX_Project(srcfilepath)
    for gene_list in column_list:
        for type_name in network_activity_feature:
            Analysis.XGBoost_algorithm_prediction(gene_list_name=gene_list ,network_activity_feature = type_name ,predict_num = 70)
        Analysis.XGBoost_algorithm_prediction_per_cluster(gene_list_name=gene_list ,predict_num = 70)