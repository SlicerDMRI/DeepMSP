import numpy as np
import statistics
from itertools import combinations
import math
import cv2
import copy
import json
import io
import time
import os
from argparse import ArgumentParser
from glob import glob
import csv
import whitematteranalysis as wma
import sys
from vanilla_gradient import VanillaGradient as SaliencyMap
from guided_backprop import GuidedBackprop

import torch
import torch.nn as nn

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from sklearn.metrics import davies_bouldin_score as DB
from sklearn.metrics import silhouette_score as SS
from sklearn.metrics import calinski_harabasz_score as CH
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, HDBSCAN, AgglomerativeClustering

torch.manual_seed(0)
np.random.seed(0)

def plot_saliency_grid(avg_over_subjects, metric_names, category_names, output_dir):
    mpl.use('TkAgg'), plt.close('all'), plt.clf()
    if not os.path.exists(OUTPUT_BASE):
        os.makedirs(OUTPUT_BASE)

    # Compute the category means, so that we can sort by their average
    category_averages = np.swapaxes(np.array([np.mean(avg_over_subjects[:,:2,:],1), np.mean(avg_over_subjects[:,2:,:],1)]),0,1)
    column_means = np.mean(category_averages,axis=(1,2))
    sorted_indices = np.argsort(column_means)

    for task_id in range(avg_over_subjects.shape[1]):
        plt.imshow(avg_over_subjects[:,task_id,:].T[:,sorted_indices])
        plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        plt.xlabel('Cluster')
        plt.ylabel('dMRI Feature')
        plt.title(metric_names[task_id])
        plt.savefig(output_dir + '/saliency_task_'+metric_names[task_id]+'.png', bbox_inches='tight', dpi=300)
        plt.clf()


    for i, category_name in enumerate(['Motor', 'Cognitive']):
        plt.imshow(category_averages[:,i,:].T[:,sorted_indices])
        plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        plt.xlabel('Cluster')
        plt.ylabel('dMRI Feature')
        plt.title(category_name)
        plt.savefig(output_dir + '/saliency_category_'+category_name+'.png', bbox_inches='tight', dpi=300)
        plt.clf()

    # Compute difference
    diffs = np.abs(category_averages[:,0,:] - category_averages[:,1,:]).T

    # Sort by mean of motor/cognitive for each feature
    diffs = diffs[:, sorted_indices]

    plt.imshow(diffs, cmap='gray')
    plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    plt.xlabel('Cluster')
    plt.ylabel('dMRI Feature')
    plt.title('Absolute Difference (Motor - Cognitive)')
    plt.savefig(output_dir + '/saliency_category_diffs.png', bbox_inches='tight', dpi=300)
    plt.clf()

def plot_per_category_saliency(all_scores, tract_names, categories, cluster_names, output_dir, tract_order=['ICP', 'MCP', 'SCP', 'IP', 'PF'], ymax=None):
    # num_folds x num_metrics x num_subjects_per_fold x num_clusters x num_features
    category_scores = {'motor': [], 'cognitive': []}
    task_scores = np.mean(all_scores, axis=(2,4)) # num_folds x num_metrics x num_clusters
    for i, category in enumerate(categories):
        task_score = task_scores[:,i,:]
        category_scores[category].append(task_score) # num_folds x num_clusters
    # now category_scores['motor'] is of shape: num_metrics_of_motor x num_folds x num_clusters

    averaged_category_scores = {}
    for category in category_scores.keys():
        averaged_category_scores[category] = np.mean(np.array(category_scores[category]),0)
    category_scores = averaged_category_scores # now category_scores['motor'] is: num_folds x num_clusters

    # tract -> cluster scores
    tract_scores = {tract: [[], []] for tract in set(tract_names)}
    for i, tract_name in enumerate(tract_names):
        motor, cognitive = category_scores['motor'][:,i], category_scores['cognitive'][:,i]
        tract_scores[tract_name][0].append(motor)
        tract_scores[tract_name][1].append(cognitive)
    # now: tract_scores['MCP'][0] is of shape: num_clusters x num_folds

    averaged_tract_scores = {tract: [[], []] for tract in tract_scores.keys()}
    for tract in tract_scores.keys():
        averaged_tract_scores[tract][0] = np.mean(np.array(tract_scores[tract][0]),0)
        averaged_tract_scores[tract][1] = np.mean(np.array(tract_scores[tract][1]),0)
    tract_scores = averaged_tract_scores
    # now: tract_scores[tract][0] is of shape: num_folds

    xlabels = tract_order
    series_labels = ['Motor', 'Cognitive']
    series_data = np.array([[np.mean(np.array(tract_scores[tract][0])), np.mean(np.array(tract_scores[tract][1]))] for tract in xlabels]).T
    series_errors = np.array([[np.std(np.array(tract_scores[tract][0])), np.std(np.array(tract_scores[tract][1]))] for tract in xlabels]).T

    bar_graph(xlabels, series_labels, series_data, series_errors, 'Saliency', output_dir=output_dir, fn='tract_categories', title=None, ymax=ymax)


# xlabels = ['x1', 'x2', 'x3']
# series_labels = [seriesA, seriesB, ...]
# series_data = [[x1,x2,x3], [x1,x2,x3], ...]
def bar_graph(xlabels, series_labels, series_data, errors, ylabel, output_dir, fn, title=None, ymax=None):
    mpl.use('TkAgg'), plt.close('all'), plt.clf()
    n_groups, n_series = len(series_data[0]), len(xlabels)

    barWidth = 0.4
    index = np.arange(n_groups)

    # Plotting
    for i, series in enumerate(series_data):
        label = series_labels[i]
        plt.bar(index + i*barWidth, series, barWidth, label=label, yerr=errors[i])

    # Add labels and title
    plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)

    # Add legend
    if ymax is not None:
        plt.ylim(0,ymax)
    plt.xticks(index+barWidth/2, xlabels)
    plt.legend()

    # Show the plot
    plt.savefig(output_dir + '/' + fn + '.png', bbox_inches='tight', dpi=300)
    plt.clf()

def parcel_analysis(labels, scores, task_names, cluster_names, feature_names, tract_names, categories, output_dir, tract_order=['ICP', 'MCP', 'SCP', 'IP', 'PF'], print_parcels=False):
    saliency_values = scores.copy()
    mpl.use('TkAgg'), plt.close('all'), plt.clf()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    bar_colors = ['purple', 'cyan', 'gold', 'red']

    # Create the dictionaries
    parcels = {label: [] for label in set(labels)}
    parcel_keys = list(sorted(parcels.keys()))
    for i, label in enumerate(labels):
        parcels[label].append([scores[i], tract_names[i], cluster_names[i]])

    # 1. For each parcel, we want the mean saliency for each feature
    feature_means = []
    feature_stdevs = []
    for parcel in parcel_keys:
        curr_scores = np.mean(np.array([item[0] for item in parcels[parcel]]),axis=1)
        feature_means.append(np.mean(curr_scores,0))
        feature_stdevs.append(np.std(curr_scores,0))
    ymax = math.ceil(max([max(feature_means[i])+max(feature_stdevs[i]) for i in range(len(feature_means))]) *10) / 10.0
    for i in range(len(feature_means)):
        curr_parcel = feature_means[i]
        curr_err = feature_stdevs[i]
        plt.rcParams['font.size'] = 16
        plt.title('Parcel ' + str(i))
        plt.bar([feature_num for feature_num in range(len(curr_parcel))], curr_parcel, yerr=curr_err)
        plt.xticks([feature_num for feature_num in range(len(curr_parcel))], feature_names, rotation=90)
        plt.ylabel('Mean Saliency')
        plt.ylim(0,ymax)
        plt.savefig(output_dir + '/feature_means_parcel_'+str(i)+'.png', bbox_inches='tight', dpi=300)
        plt.rcParams['font.size'] = 12
        plt.clf()
    # we want a 20 x num_items array
    feature_data = np.array(feature_means).T
    plot_violin(list(feature_data), output_dir, 'feature_means_violin', None, 'Saliency', 0, ymax, feature_names, hbar=False)
    plt.boxplot(feature_data.T)
    plt.xticks([feature_num for feature_num in range(len(curr_parcel))], feature_names, rotation=90)
    plt.ylabel('Saliency')
    plt.ylim(0,0.4)
    plt.savefig(output_dir + '/feature_means_boxplot.png', bbox_inches='tight', dpi=300)
    plt.clf()

    feature_means_means = [sum(item)/len(item) for item in feature_means]

    # 2. For a given parcel, we want the mean overall saliency
    overall_means = []
    overall_stdevs = []
    for parcel in parcel_keys:
        curr_scores = np.mean(np.array([item[0] for item in parcels[parcel]]), axis=(1,2))
        overall_means.append(np.mean(curr_scores))
        overall_stdevs.append(np.std(curr_scores))
    plt.rcParams['font.size'] = 16
    bar = plt.bar([i for i in range(len(overall_means))], overall_means, yerr=overall_stdevs)
    plt.xticks([i for i in range(len(overall_means))], ['Parcel ' + str(i+1) for i in range(len(overall_means))])
    plt.ylabel('Saliency')
    plt.savefig(output_dir + '/overall_means_parcels.png', bbox_inches='tight', dpi=300)
    plt.rcParams['font.size'] = 12
    plt.clf()

    # 3. For a given parcel, we want the mean saliency for each task
    task_means = []
    task_stdevs = []
    for parcel in parcel_keys:
        curr_scores = np.mean(np.array([item[0] for item in parcels[parcel]]),axis=2)
        task_means.append(np.mean(curr_scores,0))
        task_stdevs.append(np.std(curr_scores,0))
    ymax = math.ceil(max([max(task_means[i])+max(task_stdevs[i]) for i in range(len(task_means))]) *10) / 10.0
    for i in range(len(task_means)):
        curr_parcel = task_means[i]
        curr_err = task_stdevs[i]
        plt.rcParams['font.size'] = 16
        plt.title('Parcel ' + str(i))
        plt.bar([task_num for task_num in range(len(curr_parcel))], curr_parcel, yerr=curr_err)
        plt.xticks([task_num for task_num in range(len(curr_parcel))], task_names)
        plt.ylim(0,ymax)
        plt.ylabel('Saliency')
        plt.savefig(output_dir + '/task_means_parcel_'+str(i)+'.png', bbox_inches='tight', dpi=300)
        plt.rcParams['font.size'] = 12
        plt.clf()

    # 4. For a given parcel, we want the mean saliency for each category (motor/cognitive)
    # parcels[parcel] = [scores, tract_names, cluster_names]
    category_means = []
    category_stdevs = []
    for parcel in parcel_keys:
        task_scores = np.mean(np.array([item[0] for item in parcels[parcel]]),axis=2)
        motor_scores = np.mean(task_scores[:,:2],1)
        cognitive_scores = np.mean(task_scores[:,2:],1)
        curr_scores = np.array([motor_scores, cognitive_scores]).T
        category_means.append(np.mean(curr_scores,0))
        category_stdevs.append(np.std(curr_scores,0))
    ymax = math.ceil(max([max(category_means[i])+max(category_stdevs[i]) for i in range(len(category_means))]) *10) / 10.0
    for i in range(len(category_means)):
        plt.rcParams['font.size'] = 18
        curr_parcel = category_means[i]
        curr_err = category_stdevs[i]
        plt.title('Parcel ' + str(i))
        bar = plt.bar([category_num for category_num in range(len(curr_parcel))], curr_parcel, yerr=curr_err)
        plt.xticks([category_num for category_num in range(len(curr_parcel))], ['Motor', 'Cognitive'])
        plt.ylim(0,ymax)
        plt.ylabel('Saliency')
        plt.savefig(output_dir + '/category_means_parcel_'+str(i)+'.png', bbox_inches='tight', dpi=300)
        plt.rcParams['font.size'] = 12
        plt.clf()
    
    # 4.1 Plot each category as a series
    xlabels = ['Parcel '+str(k+1) for k in parcel_keys]
    series_labels = ['Motor', 'Cognitive']
    series_data = [[category_means[i][0] for i in range(len(category_means))], [category_means[i][1] for i in range(len(category_means))]]
    series_errors = [[category_stdevs[i][0] for i in range(len(category_stdevs))], [category_stdevs[i][1] for i in range(len(category_stdevs))]]
    bar_graph(xlabels, series_labels, series_data, series_errors, 'Saliency', output_dir=output_dir, fn='category_means', title=None, ymax=0.35)

    # 5. For a given parcel, we want a count of how many of each white matter tract belong to it
    parcel_freqs = []
    for parcel in parcel_keys:
        tracts = [item[1] for item in parcels[parcel]]
        freq = {tract: 0 for tract in tract_names}
        total = 0
        for tract in tracts:
            freq[tract] += 1
            total += 1
        distr = {tract: 100*freq[tract]/total for tract in freq.keys()}
        parcel_freqs.append(distr)
    ymax = math.ceil(max([max(distr.values()) for distr in parcel_freqs]) / 10) * 10
    for parcel_id in range(len(parcel_keys)):
        distr = parcel_freqs[parcel_id]
        plt.title('Parcel ' + str(parcel_id))
        plt.bar([i for i in range(len(distr.keys()))], [distr[tract] for tract in tract_order])
        plt.xticks([i for i in range(len(distr.keys()))], tract_order)
        plt.ylim(0,ymax)
        plt.ylabel('Percentage of parcel')
        plt.savefig(output_dir + '/tract_distr_parcel_'+str(parcel_id)+'.png', bbox_inches='tight', dpi=300)
        plt.clf()

    # 6. For a given tract, plot its parcel distribution
    tract_clusters = {tract: [] for tract in set(tract_names)}
    cluster_parcels = {}
    for i in range(len(labels)):
        tract_clusters[tract_names[i]].append(cluster_names[i])
        cluster_parcels[cluster_names[i]] = labels[i]

    tract_freqs = []
    for tract in tract_order:
        parcels_within = [cluster_parcels[cluster] for cluster in tract_clusters[tract]]
        counts = {parcel: 0 for parcel in parcel_keys}
        for parcel in parcels_within:
            counts[parcel] += 1
        distr = {parcel: 100*counts[parcel]/len(parcels_within) for parcel in counts}
        tract_freqs.append(distr)
    ymax = math.ceil(max([max(distr.values()) for distr in tract_freqs]) / 10) * 10
    for tract_id in range(len(tract_order)):
        distr = tract_freqs[tract_id]
        plt.title(tract_order[tract_id])
        plt.bar([i for i in range(len(distr.keys()))], [distr[parcel] for parcel in parcel_keys])
        plt.xticks([i for i in range(len(distr.keys()))], [str(k) for k in parcel_keys])
        plt.ylim(0,ymax)
        plt.ylabel('Percentage of tract')
        plt.xlabel('Parcel')
        plt.savefig(output_dir + '/parcel_distr_tract_'+tract_order[tract_id]+'.png', bbox_inches='tight', dpi=300)
        plt.clf()

    if print_parcels:
        for tract in tract_clusters:
            print(tract)
            for parcel in parcel_keys:
                print('Parcel ' + str(parcel))
                for cluster_name in tract_clusters[tract]:
                    if cluster_parcels[cluster_name] == parcel:
                        print(cluster_name)
            print('--')
    
    # 7. For a given tract, we want the average motor/cognitive score for it's consistuent parcels.
    # i.e. For all parcels, store the mean motor/cognitve scores. Then assign this value to all 
    #      clusters within that parcel. Then find the averages of these vectors for each tract.
    cluster_parcel_vectors = {}
    for i, cluster in enumerate(cluster_names):
        parcel = cluster_parcels[cluster]
        cluster_parcel_vectors[cluster] = category_means[parcel]
    tract_mean_vectors = []
    tract_std_vectors = []
    all_cat_scores = []
    for tract in tract_order:
        clusters_in_tract = tract_clusters[tract]
        scores = [cluster_parcel_vectors[cluster] for cluster in clusters_in_tract]
        all_cat_scores.append(scores)
        tract_mean_vectors.append(np.mean(np.array(scores),0))
        tract_std_vectors.append(np.std(np.array(scores),0))
    ymax = math.ceil(max([max(tract_mean_vectors[i])+max(tract_std_vectors[i]) for i in range(len(tract_mean_vectors))]) *10) / 10.0
    for tract_id in range(len(tract_order)):
        score, error = tract_mean_vectors[tract_id], tract_std_vectors[tract_id]
        plt.title(tract_order[tract_id])
        plt.bar([i for i in range(len(score))], score, yerr=error)
        plt.xticks([i for i in range(len(score))], ['Motor', 'Cognitive'])
        plt.ylim(0,ymax)
        plt.ylabel('Saliency')
        plt.savefig(output_dir + '/parcel_mean_category_'+tract_order[tract_id]+'.png', bbox_inches='tight', dpi=300)
        plt.clf()

    parcel_tracts = {} # parcel_tracts[parcel][tract] = [clusterA, clusterB, ...]
    tract_clusters = {tract_name: [] for tract_name in tract_order}
    cluster_parcel = {cluster_names[i]: labels[i] for i in range(len(cluster_names))}
    cluster_scores = {cluster_names[i]: np.mean(saliency_values[i],axis=1) for i in range(len(cluster_names))}
    for parcel in parcel_keys:
        if parcel not in parcel_tracts:
            parcel_tracts[parcel] = {}
        for tract in sorted(list(set(tract_names))):
            if tract not in parcel_tracts[parcel]:
                parcel_tracts[parcel][tract] = []
    for i in range(len(cluster_names)):
        parcel = labels[i]
        cluster_name = cluster_names[i]
        tract = tract_names[i]
        parcel_tracts[parcel][tract].append(cluster_name)
        tract_clusters[tract].append(cluster_name)

    parcel_tract_scores = {}
    for parcel in parcel_tracts.keys():
        if parcel not in parcel_tract_scores:
            parcel_tract_scores[parcel] = {}
        for tract in parcel_tracts[parcel].keys():
            if len(parcel_tracts[parcel][tract]) == 0:
                continue
            mean_score = np.mean(np.array([[(cluster_scores[cluster_name][0]+cluster_scores[cluster_name][1])/2, (cluster_scores[cluster_name][2]+cluster_scores[cluster_name][3])/2] for cluster_name in parcel_tracts[parcel][tract]]),axis=0)
            parcel_tract_scores[parcel][tract] = mean_score

    for tract in tract_order:
        scores = []
        clusters_included = []
        for cluster_name in tract_clusters[tract]:
            parcel = cluster_parcels[cluster_name]
            score = parcel_tract_scores[parcel][tract] # get the score for this parcel/tract combo
            scores.append(score)
            clusters_included.append(cluster_name)

        motor_cog_score = np.array([(x[0]-x[1])/(x[0]+x[1]) for x in scores])
        motor_cog_score = motor_cog_score / np.max(np.abs(motor_cog_score))
        motor_cog_score = motor_cog_score/2+0.5

        gen_model(motor_cog_score, 'tract', tract+'_mixed', clusters_included, 'bwr', opacity=0.1, output_dir=output_dir+'/vis')

# Expects shape: 5 x 53 x 20
def plot_tract_saliencies(scores, tract_names, output_dir, output_fn, tract_order=['ICP', 'MCP', 'SCP', 'IP', 'PF']):
    mpl.use('TkAgg'), plt.close('all'), plt.clf()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Divide into tracts
    tract_scores = {tract_name: [] for tract_name in set(tract_names)}
    for i, tract in enumerate(tract_names):
        tract_scores[tract].append(scores[:,i,:])

    # Convert mean/stdev for each tract
    means = {}
    stdevs = {}
    for tract in tract_scores.keys():
        # average over all the individual clusters, and the 20 features
        base_score = np.mean(np.array(tract_scores[tract]),axis=(0,2)) 
        
        # average over the folds
        mean = np.mean(base_score,axis=0)
        stdev = np.std(base_score,axis=0)

        means[tract] = mean
        stdevs[tract] = stdev

    means_plot, stdevs_plot, names_plot = [], [], []
    for tract in tract_order:
    #for tract in sorted(tract_scores.keys()):
        names_plot.append(tract)
        means_plot.append(means[tract])
        stdevs_plot.append(stdevs[tract])

    print(np.array(means_plot).shape)
    plt.bar([i for i in range(len(names_plot))], means_plot, yerr=stdevs_plot)
    plt.xticks([i for i in range(len(names_plot))],names_plot)
    plt.ylabel('Saliency')
    plt.savefig(output_dir + '/'+output_fn+'.png', bbox_inches='tight', dpi=300)
    plt.clf()

# Expects a N x num_features vector, to which it will calculate the
# mean and stadard deviation over N
def plot_feature_saliencies(data, names, output_dir, output_fn, title=None, ymax=None):
    mpl.use('TkAgg'), plt.close('all')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    feature_colors = []
    base_names = {'FA1': 'red', 'FA2': 'orange', 'NoS': 'green', 'NoP': 'purple', 'Trace1': 'blue', 'Trace2': 'gold'}
    for feature_name in names:
        feature_colors.append(base_names[feature_name.split('.')[0]])

    stat_colors = []
    stat_type = {'Min': 'red', 'Max': 'orange', 'Median': 'green', 'Mean': 'purple', 'Variance': 'blue'}
    for feature_name in names:
        if feature_name == 'NoS':
            stat_colors.append('gold')
        elif feature_name == 'NoP':
            stat_colors.append('pink')
        else:
            stat_colors.append(stat_type[feature_name.split('.')[-1]])

    means, stdevs = np.mean(data,0), np.std(data,0)

    print('Feature Means ' + title)
    print(np.mean(means[:5]), np.mean(means[5:10]), means[10], means[11], np.mean(means[12:16]), np.mean(means[16:]))

    print('Stat Means ' + title)
    print('Min, Max, Median, Mean, Variance')
    print(np.mean(means[np.array([0,5,12,16])]), np.mean(means[np.array([1,6,13,17])]), np.mean(means[np.array([2,7,14,18])]), np.mean(means[np.array([3,8,15,19])]), np.mean(means[np.array([4,9])]))
    if ymax is not None:
        plt.ylim(0,ymax)
    bar = plt.bar([i for i in range(len(names))], means, yerr=stdevs)
    plt.xticks([i for i in range(len(names))],names,rotation=90)
    plt.ylabel('Saliency')
    if title is not None:
        plt.title(title)
    plt.savefig(output_dir + '/'+output_fn+'.png', bbox_inches='tight', dpi=300)

    # Plot with feature colors
    for i in range(len(bar)):
        bar[i].set_color(feature_colors[i])
    plt.savefig(output_dir + '/'+output_fn+'_feature_colors.png', bbox_inches='tight', dpi=300)

    # Plot with stat colours
    for i in range(len(bar)):
        bar[i].set_color(stat_colors[i])
    plt.savefig(output_dir + '/'+output_fn+'_stat_colors.png', bbox_inches='tight', dpi=300)

    plt.clf()

def concatenate_images(fn1, fn2, output_dir, output_fn):
    im1, im2 = cv2.imread(fn1), cv2.imread(fn2)
    concat = np.hstack((im1, im2))
    cv2.imwrite(output_dir + '/' + output_fn + '.png', concat)

def average_hemispheres(scores, cluster_names, tract_names):
    unique_cluster_ids = sorted(list(set([x.split('.')[-1] for x in cluster_names])))
    cluster_ids = [x.split('.')[-1] for x in cluster_names]

    # Create a list of cluster types
    cluster_types = []
    for cluster_id in unique_cluster_ids:
        index = cluster_ids.index(cluster_id) # find the first match, this is enough for now
        if 'commissural.' in cluster_names[index]:
            cluster_types.append('commissural')
        else:
            cluster_types.append('hemisphere')

    new_scores = []
    new_tract_names = []
    for i, cluster_id in enumerate(unique_cluster_ids):
        # Determine all indexes of the current cluster_id
        all_locations = []
        for index, cluster_name in enumerate(cluster_names):
            if cluster_id in cluster_name:
                all_locations.append(index)

        # Either take directly or average the hemispheres, and store the result
        if cluster_types[i] == 'commissural':
            new_scores.append(scores[all_locations[0]])
            new_tract_names.append(tract_names[all_locations[0]])
            if len(all_locations) != 1:
                print('ERROR: Invalid number of cluster names.')
                sys.exit()
        else:
            mean = (scores[all_locations[0]] + scores[all_locations[1]])/2
            new_scores.append(mean)
            scores[all_locations[1]] = mean
            new_tract_names.append(tract_names[all_locations[0]])
            if len(all_locations) != 2:
                print('ERROR: Invalid number of cluster names.')
                sys.exit()

    return new_scores, unique_cluster_ids, new_tract_names

def write_to_file(s, output_dir, fn):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_dir + '/' + fn, 'w') as f:
        f.write(s)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout_rate=0):
        super(TransformerBlock, self).__init__()
        self.multi_head_attention = nn.MultiheadAttention(d_model, nhead)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_feedforward, d_model)
        )
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Multi-Head Attention
        attn_output, _ = self.multi_head_attention(self.layer_norm1(x), self.layer_norm1(x), self.layer_norm1(x))
        x = x + self.dropout1(attn_output)
        
        # Feed Forward
        ff_output = self.feed_forward(self.layer_norm2(x))
        x = x + self.dropout2(ff_output)
        
        return x

class FC(nn.Module):
    def __init__(self, input_channels=1, num_classes=1, dropout=90):
        super(FC, self).__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(input_channels, 4096),
            nn.ReLU(),
            nn.Dropout(dropout/100),

            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(dropout/100),

            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(dropout/100),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout/100),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout/100),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout/100),

            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.decoder(x)

        return x


class CNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=1, dropout=90):
        super(CNN, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout/100),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout/100),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout/100),
            nn.Conv1d(in_channels=64, out_channels=100, kernel_size=5, stride=1, padding=2),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(100 * input_channels, 100),
            nn.Dropout(dropout/100),
            nn.Linear(100, 64),
            nn.Dropout(dropout/100),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # Input shape: batch_size x 500
        # Reshape for 1D-CNN: batch_size x 1 x 500
        x = x.unsqueeze(1)

        x = self.encoder(x)
        x = x.view(x.size(0), -1) # flatten
        x = self.decoder(x)
        
        return x

class TransformerModel(nn.Module):
    def __init__(self, input_channels=1940, num_classes=1, dropout=90, d_model=512, nhead=8, num_layers=6,dim_feedforward=2048):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_channels, d_model)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model, nhead, num_layers, dim_feedforward, dropout/100) for _ in range(num_layers)]
        )
        
        self.dense_layers = nn.Sequential(
            nn.Linear(d_model, d_model),
            # Add more dense layers if necessary
            nn.Dropout(dropout/100),
            nn.Linear(d_model, 11)
        )
        self.dropout = nn.Dropout(dropout/100)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(0)  # Add a batch dimension
        for block in self.transformer_blocks:
            x = block(x)

        x = self.dense_layers(x)
        return x.squeeze(0)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, folds, dataset_dir='/Users/aritchetchenian/Desktop/graph_cnn', metric='TPVT', mean=None, stdev=None):
        # Data structure for storing all items
        self.subjects = []

        all_names = ['Endurance_AgeAdj', 'GaitSpeed_Comp', 'Dexterity_AgeAdj', 'Strength_AgeAdj', 'PicSeq_AgeAdj', 'CardSort_AgeAdj', 'Flanker_AgeAdj', 'ReadEng_AgeAdj', 'PicVocab_AgeAdj', 'ProcSpeed_AgeAdj', 'ListSort_AgeAdj']
        self.labels = [get_labels_mapping('./csvs/' + name + '.csv') for name in all_names]

        # Gather subject IDs
        subject_ids = []
        for fold in folds:
            subject_ids += ['/'.join(x.replace('\\', '/').split('/')[-2:]) for x in sorted(glob(dataset_dir + '/fold'+str(fold)+'/*.npy'))]
        subject_ids.sort()

        # Calculate the mean/stdev
        if mean is None:
            data = np.array([np.load(dataset_dir + '/' + subject) for subject in subject_ids])
            self.mean, self.stdev = np.mean(data,axis=0), np.std(data,axis=0)
        else:
            self.mean, self.stdev = mean, stdev

        for subject in subject_ids:
            total_label = np.array([self.labels[i][subject.split('/')[-1].split('.')[0]] for i in range(len(self.labels))])

            input_vector = np.load(dataset_dir + '/' + subject)

            # standardisation
            input_vector = (input_vector - self.mean) / (self.stdev + 0.00001)

            self.subjects.append([torch.from_numpy(input_vector).float(), torch.tensor(total_label).float()])

    def __getitem__(self, idx):
        return self.subjects[idx]

    def get_stats(self):
        return [self.mean, self.stdev]

    def __len__(self):
        return len(self.subjects)

def gen_parcel_stats(parcel_labels, input_vectors, metrics, tract_names, categories, output_dir):
    unique_labels = sorted(list(set(parcel_labels)))
    unique_categories = sorted(list(set(categories)))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    mpl.use('TkAgg')
    plt.close('all')

    # Calculate tract ownerships (e.g. parcel 0 is 20% SCP, 30% ICP, etc.)
    for label in unique_labels:
        counts = {}
        for tract_name in sorted(list(set(tract_names))):
            counts[tract_name] = 0
        total_count = 0
        for i in range(len(parcel_labels)):
            if parcel_labels[i] == label:
                counts[tract_names[i]] += 1
                total_count += 1
        plt.clf()
        plt.figure(figsize=(10,6))
        plt.bar([tract_name for tract_name in sorted(list(set(tract_names)))], [100*counts[tract_name]/total_count for tract_name in sorted(list(set(tract_names)))])
        plt.xticks(fontsize=45)
        plt.savefig(output_dir + '/'+str(label)+'_tract_distr.png', bbox_inches='tight')


    # Calculate mean saliency of each task for each label (e.g. parcel 0 has 0.1 saliency for strength, etc.)
    for label in unique_labels:
        vectors = []
        for i in range(len(parcel_labels)):
            curr_label = parcel_labels[i]
            input_vector = input_vectors[i]
            if curr_label == label:
                vectors.append(input_vector)
        vectors_mean = np.mean(np.array(vectors),0)
        vectors_stdev = np.std(np.array(vectors),0)

        # Plot the mean saliency for each task
        plt.close('all')
        plt.clf()
        plt.figure(figsize=(10,6))
        plt.bar([metric for metric in metrics], vectors_mean, yerr=vectors_stdev, capsize=1)
        plt.xticks(rotation=10, fontsize=20)
        plt.xlabel('Metric')
        plt.ylabel('Mean Saliency')
        plt.ylim(0,1)
        plt.title('Mean Saliencies Per Cluster')
        plt.savefig(output_dir + '/mean_task_saliencies_'+str(label)+'.png', bbox_inches='tight')

        # Plot the mean saliency for each category
        category_means = [[] for category in unique_categories]
        for i, category in enumerate(categories):
            category_index = unique_categories.index(category)
            category_means[category_index].append(vectors_mean[i])
        category_means = [sum(x)/len(x) for x in category_means]

        plt.clf()
        plt.figure(figsize=(10,6))
        plt.bar([category for category in unique_categories], category_means)
        plt.xticks(fontsize=34)
        plt.ylabel('Mean Saliency')
        plt.ylim(0,1)
        plt.title('Mean Saliencies Per Cluster')
        plt.savefig(output_dir + '/mean_category_saliencies_'+str(label)+'.png', bbox_inches='tight')

        plt.figure(figsize=(10,6))
        plt.bar([category for category in unique_categories], category_means)
        plt.xticks(fontsize=34)
        plt.xlabel('Category')
        plt.ylabel('Mean Saliency')
        plt.title('Mean Saliencies Per Cluster')
        plt.savefig(output_dir + '/mean_category_saliencies_yaxis_'+str(label)+'.png', bbox_inches='tight')

def cosine(a,b):
    dot = np.dot(a,b)
    normA = np.linalg.norm(a)
    normB = np.linalg.norm(b)

    return dot / (normA * normB)

def mean_cosine_similarity(all_folds):
    all_sims = []
    for a,b in combinations(list(all_folds),2):
        sim = cosine(a,b)
        all_sims.append(sim)

    return np.mean(np.array(all_sims)), np.std(np.array(all_sims))

# Given a set of scores (num_clusters x num_categories), and an assignment of categories
# Return the mean score for each cluster for each category
# i.e. input is 97 cluster scores for each of strength, dexterity, TPVT, Flanker
#      then output would be 97 cluster scores for motor and cognitive
def get_category_scores(scores, categories):
    # Initialise
    category_scores = {}
    for category in set(categories):
        category_scores[category] = []

    # Aggregate
    for i in range(len(categories)):
        category = categories[i]
        mean_scores = scores[:,i]
        category_scores[category].append(mean_scores)

    # Summarise
    for category in set(categories):
        category_scores[category] = sum(category_scores[category]) / len(category_scores[category])

    return category_scores

def split_into_tracts(labels, cluster_names, tract_names):
    new_labels = [[] for i in range(len(set(cluster_names)))]
    norm_new_labels = [[] for i in range(len(set(cluster_names)))]
    new_clusters = [[] for i in range(len(set(cluster_names)))]
    tract_order = sorted([x for x in set(tract_names)])

    # Globally normalise the labels
    norm_labels = norm(labels)

    for i in range(len(labels)):
        label = labels[i]
        norm_label = norm_labels[i]
        name = cluster_names[i]
        tract = tract_names[i]

        pos = tract_order.index(tract)

        norm_new_labels[pos].append(norm_label)
        new_labels[pos].append(label)
        new_clusters[pos].append(name)

    return norm_new_labels, new_labels, new_clusters, tract_order

def get_freq(items, possible_labels):
    freq = {}
    for item in sorted(possible_labels):
        freq[item] = 0

    for item in items:
        freq[item] += 1

    return freq

def vis_tracts(labels, cluster_names, tract_names, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    norm_new_labels, new_labels, new_clusters, tract_order = split_into_tracts(labels, cluster_names, tract_names)

    # Generate a model for each tract
    for tract_num in range(len(tract_order)):
        gen_model(norm_new_labels[tract_num], 'cluster ID', tract_order[tract_num], new_clusters[tract_num], 'rainbow', opacity=0.1, output_dir=output_dir)

    # Plot the k-means label distribution for each tract
    freqs = []
    tract_labels = []
    for tract_num in range(len(tract_order)):
        all_labels = new_labels[tract_num]
        freq = get_freq(all_labels, list(set(labels)))
        sorted_freq = [freq[x] for x in sorted(list(freq.keys()))]
        freqs.append(sorted_freq)
        tract_labels.append(tract_order[tract_num])
    freq_labels = sorted(list(freq.keys()))

    mpl.use('TkAgg')
    plt.close('all')
    plt.clf()

    num_elements = len(tract_labels)
    num_components = len(freqs[0])
    bar_width = 0.2
    index = np.arange(num_elements)
    fig, ax = plt.subplots()
    for i, component in enumerate(freq_labels):
        component_values = [freq[component] for freq in freqs]
        bars = plt.bar(index + i * bar_width, component_values, width=bar_width, label=f'KMeans Cluster {component}')

    # Add some text for labels, title, and axes ticks
    plt.xlabel('Tracts')
    plt.ylabel('Frequency')
    plt.title('Frequency of Components for Each Element')
    plt.xticks(index + bar_width, tract_labels)
    plt.legend()

    plt.savefig(output_dir + '/kmeans_tract_distribution.png', bbox_inches='tight')

def vis_kmeans_clusters(labels, cluster_names, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get the names of the clusters in each parcel
    per_cluster_names = [[] for i in range(len(set(labels)))]
    for i, label in enumerate(labels):
        name = cluster_names[i]
        per_cluster_names[label].append(name)

    # Generate the .mrml files
    for kmeans_cluster_id, val in enumerate(sorted(list(set(norm(labels))))):
        gen_model([val for i in range(len(per_cluster_names[kmeans_cluster_id]))], 'per_cluster', str(kmeans_cluster_id), per_cluster_names[kmeans_cluster_id], 'rainbow', opacity=0.1, output_dir=output_dir)

def tract_to_id(names):
    unique_names = []
    for item in names:
        if item not in unique_names:
            unique_names.append(item)

    maps = [i for i in range(len(unique_names))]

    ids = []
    for name in names:
        name_index = unique_names.index(name)
        ids.append(maps[name_index])

    return ids

def get_cluster_mean(salience, num_features):
    return np.mean(salience.reshape(-1, num_features), axis=1)

def norm(x):
    if np.min(x) == np.max(x):
        print('Min and Max are the same: %.2f' % (np.min(x)))
        return x
    return (x-np.min(x)) / (np.max(x)-np.min(x))

# Return a subject->label mapping for float labels
def get_labels_mapping(f):
    result = {}
    with open(f, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            result[row[0]] = float(row[1])
    return result

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value

# Given an array: num_violins x num_clusters, plot a violin plot
def plot_violin(scores, output_dir, output_fn, title, ylabel, ymin, ymax, labels, hbar=True):
    if type(scores) != type([]):
        print('ERROR: type must be list.')
        sys.exit()

    mpl.use('TkAgg'), plt.close('all'), plt.clf()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Do the actual plotting
    plt.violinplot(scores, showmeans=False, showextrema=False, showmedians=False)
    quartile1, medians, quartile3 = np.percentile(scores, [25, 50, 75], axis=1)
    whiskers = np.array([adjacent_values(sorted_array, q1, q3) for sorted_array, q1, q3 in zip(scores, quartile1, quartile3)])
    whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]
    inds = np.arange(1, len(medians) + 1)
    plt.scatter(inds, medians, marker='o', color='white', s=6, zorder=3)
    plt.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=3)
    if title is not None:
        plt.title(title)
    plt.ylabel(ylabel)
    plt.ylim(ymin, ymax)
    plt.xticks([i+1 for i in range(len(scores))], labels, rotation=90)
    if hbar:
        plt.axhline(y=0.5, linestyle='--', color='r')
    plt.rcParams['font.size'] = 12
    plt.savefig(output_dir + '/'+output_fn+'.png', bbox_inches='tight', dpi=300)
    plt.clf()


def gen_model(vals, label, output_suffix, cluster_names, color_map, output_dir, opacity=1):
    plt.close('all')
    # Create the output directory if required
    output_fn = output_dir + '/result_'+label+'_'+output_suffix+'.mrml'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Compute the colors based on the desired colormap
    if color_map == 'custom_bwr':
        colors = [(0, 0, 1), (0, 0, 0), (1, 0, 0)]
        cmap = LinearSegmentedColormap.from_list("custom_bwr", colors, N=100)
    else:
        cmap = mpl.colormaps[color_map]
    colors = cmap(vals) * 256

    # Collect a list of cluster filenames
    input_polydatas = list()
    for i in range(len(cluster_names)):
        if BILATERAL:
            input_dir = 'C:\Downloads\ORG-800FiberClusters'
        else:
            base_dir = 'C:\Downloads\ORG-separated'
            if 'left' in cluster_names[i]:
                input_dir = base_dir + '/tracts_left_hemisphere'
            elif 'right' in cluster_names[i]:
                input_dir = base_dir + '/tracts_right_hemisphere'
            elif 'commissural' in cluster_names[i]:
                input_dir = base_dir + '/tracts_commissural'
            else:
                print("ERROR: Invalid cluster")
                sys.exit()

        cluster_name = cluster_names[i].split('.')[-1]
        name = "{0}/"+cluster_name+"*"
        input_mask = name.format(input_dir)
        filepath = glob(input_mask)
        input_polydatas.append(filepath[0])

    # Save the .mrml file
    original_stdout = sys.stdout # these line just prevents wma.mrml.write from printing a bunch of useless info
    sys.stdout = io.StringIO() # these lines just prevents wma.mrml.write from printing a bunch of useless info
    wma.mrml.write(input_polydatas, colors, output_fn, ratio=1.0)
    sys.stdout = original_stdout # these lines just prevents wma.mrml.write from printing a bunch of useless info

    # Edit the file to change the opacity
    with open(output_fn) as f:
        data = f.read()
    data = data.replace('opacity="1"', 'opacity="0.1"')
    with open(output_fn, "w") as f:
        f.write(data)

def deep_str(d):
    for key, value in d.items():
        if isinstance(value, dict):
            deep_str(value)
        else:
            d[key] = "%.4f" % (value)

# Mean number of tracts per parcel
def tracts_per_parcel(tract_names, labels):
    parcel_ids = sorted(list(set(labels)))

    parcel_tracts = {parcel_id: set([]) for parcel_id in parcel_ids}

    for i, tract_name in enumerate(tract_names):
        curr_parcel = labels[i]
        parcel_tracts[curr_parcel].add(tract_name)

    all_counts = [len(parcel_tracts[parcel_id]) for parcel_id in parcel_ids]
    mean_score = sum(all_counts) / len(all_counts)
    max_score = max(all_counts)
    min_score = min(all_counts)

    return mean_score, min_score, max_score 

# Mean number of tracts per parcel
def parcels_per_tract(tract_names, labels):
    unique_tracts = sorted(list(set(tract_names)))

    parcel_tracts = {tract_id: set([]) for tract_id in unique_tracts}

    for i, parcel_id in enumerate(labels):
        curr_tract = tract_names[i]
        parcel_tracts[curr_tract].add(parcel_id)

    all_counts = [len(parcel_tracts[tract_id]) for tract_id in unique_tracts]
    mean_score = sum(all_counts) / len(all_counts)
    min_score = min(all_counts)
    max_score = max(all_counts)

    return mean_score, min_score, max_score

def generate_clustering_plots(scores, output_dir, fn, title, tract_names, whitening=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Apply whitening if enabled
    if whitening:
        pca = PCA(whiten=True)
        scores = pca.fit_transform(scores)

    intrinsic_scores = {
                        'silhouette': {'kmeans': {}, 'agglomerative': {}}, 
                        'dbi': {'kmeans': {}, 'agglomerative': {}},
                        'chi': {'kmeans': {}, 'agglomerative': {}},

                        'tracts_per_parcel': {'kmeans': {}, 'agglomerative': {}},
                        'tracts_per_parcel_max': {'kmeans': {}, 'agglomerative': {}},
                        'tracts_per_parcel_min': {'kmeans': {}, 'agglomerative': {}},

                        'parcels_per_tract': {'kmeans': {}, 'agglomerative': {}},
                        'parcels_per_tract_min': {'kmeans': {}, 'agglomerative': {}},
                        'parcels_per_tract_max': {'kmeans': {}, 'agglomerative': {}},
                       }

    # Perform the clustering and compute the scores for various K
    for k in range(2,15):
        # KMeans
        labels = KMeans(n_clusters=k, n_init=10).fit(scores).labels_
        intrinsic_scores['silhouette']['kmeans'][k] = SS(scores, labels, metric='euclidean')
        intrinsic_scores['dbi']['kmeans'][k] = DB(scores, labels)
        intrinsic_scores['chi']['kmeans'][k] = CH(scores, labels)
        intrinsic_scores['tracts_per_parcel']['kmeans'][k],intrinsic_scores['tracts_per_parcel_min']['kmeans'][k],intrinsic_scores['tracts_per_parcel_max']['kmeans'][k] = tracts_per_parcel(tract_names, labels)
        intrinsic_scores['parcels_per_tract']['kmeans'][k],intrinsic_scores['parcels_per_tract_min']['kmeans'][k],intrinsic_scores['parcels_per_tract_max']['kmeans'][k] = parcels_per_tract(tract_names, labels)

        # Agglomerative
        labels = AgglomerativeClustering(n_clusters=k).fit(scores).labels_
        intrinsic_scores['silhouette']['agglomerative'][k] = SS(scores, labels, metric='euclidean')
        intrinsic_scores['dbi']['agglomerative'][k] = DB(scores, labels)
        intrinsic_scores['chi']['agglomerative'][k] = CH(scores, labels)
        intrinsic_scores['tracts_per_parcel']['agglomerative'][k], intrinsic_scores['tracts_per_parcel_min']['agglomerative'][k],intrinsic_scores['tracts_per_parcel_max']['agglomerative'][k] = tracts_per_parcel(tract_names, labels)
        intrinsic_scores['parcels_per_tract']['agglomerative'][k],intrinsic_scores['parcels_per_tract_min']['agglomerative'][k],intrinsic_scores['parcels_per_tract_max']['agglomerative'][k] = parcels_per_tract(tract_names, labels)

    # Plot the score curves
    mpl.use('TkAgg'), plt.close('all'), plt.clf()
    for metric in intrinsic_scores.keys():
        if metric == 'silhouette':
            ylabel = 'Silhouette Coefficient'
            ymin, ymax = 0, 1
        elif metric == 'dbi':
            ylabel = 'Davies-Bouldin Index'
            ymin, ymax = 0, 1
        elif metric =='chi':
            ylabel = 'Calinski-Harabasz Index'
            ymin, ymax = None, None

        elif metric == 'tracts_per_parcel':
            ylabel = 'Mean number of tracts per parcel'
            ymin, ymax = 0, 6
        elif metric == 'parcels_per_tract':
            ylabel = 'Mean number of unique parcels per tract'
            ymin, ymax = 0, 14

        elif metric == 'tracts_per_parcel_max':
            ylabel = 'Max number of tracts per parcel'
            ymin, ymax = 0, 6
        elif metric == 'parcels_per_tract_max':
            ylabel = 'Max number of unique parcels per tract'
            ymin, ymax = 0, 14

        elif metric == 'tracts_per_parcel_min':
            ylabel = 'Min number of tracts per parcel'
            ymin, ymax = 0, 6
        elif metric == 'parcels_per_tract_min':
            ylabel = 'Min number of unique parcels per tract'
            ymin, ymax = 0, 14

        else:
            print('ERROR: Invalid metric.')
            sys.exit()

        plt.rcParams['font.size'] = 12
        all_kmeans_ks = sorted(list(intrinsic_scores[metric]['kmeans'].keys()))
        all_agglom_ks = sorted(list(intrinsic_scores[metric]['agglomerative'].keys()))

        kmeans_scores = [intrinsic_scores[metric]['kmeans'][k] for k in all_kmeans_ks]
        agglom_scores = [intrinsic_scores[metric]['agglomerative'][k] for k in all_agglom_ks]

        plt.scatter([k for k in all_kmeans_ks], kmeans_scores, color='orange')
        plt.plot([k for k in all_kmeans_ks], kmeans_scores, color='orange', label='k-means')
        plt.scatter([k for k in all_agglom_ks], agglom_scores, color='blue')
        plt.plot([k for k in all_agglom_ks], agglom_scores, color='blue', label='agglomerative')

        plt.xlabel('Number of Parcels'), plt.ylabel(ylabel), plt.title(title)
        plt.legend()

        if ymin is not None and ymax is not None:
            plt.ylim(ymin,ymax)
        plt.savefig(output_dir + '/'+fn+'_'+metric+'.png', bbox_inches='tight', dpi=300)

        # Now save a version with the mean visualised
        total_mean = sum(kmeans_scores + agglom_scores) / (len(kmeans_scores) + len(agglom_scores))
        kmeans_median = statistics.median(kmeans_scores)
        agglom_median = statistics.median(agglom_scores)
        plt.text(12,0.02, f"Mean: {total_mean:.2f}", horizontalalignment="left", color="red")
        plt.savefig(output_dir + '/'+fn+'_'+metric+'_mean.png', bbox_inches='tight', dpi=300)
        plt.clf()

    # Plot the tracts per cluster & cluster per tracts special graphs
    for metric in ['parcels_per_tract', 'tracts_per_parcel']:
        if metric == 'parcels_per_tract':
            ymin, ymax = 0, 14
            ylabel = 'Number of unique parcels per tract'
        else:
            ymin, ymax = 0, 6
            ylabel = 'Number of tracts per parcel'
        for clustering_algorithm in ['kmeans', 'agglomerative']:
            if clustering_algorithm == 'kmeans':
                clustering_algorithm_name = 'k-means'
                color='orange'
            else:
                clustering_algorithm_name = 'agglomerative'
                color='blue'

            all_ks = sorted(list(intrinsic_scores[metric][clustering_algorithm].keys()))

            mean_scores = [intrinsic_scores[metric][clustering_algorithm][k] for k in all_kmeans_ks]
            min_scores = [intrinsic_scores[metric+'_min'][clustering_algorithm][k] for k in all_kmeans_ks]
            max_scores = [intrinsic_scores[metric+'_max'][clustering_algorithm][k] for k in all_kmeans_ks]

            lower_error = np.array(mean_scores) - np.array(min_scores)
            upper_error = np.array(max_scores) - np.array(mean_scores)
            asymmetric_error = [lower_error, upper_error]

            plt.scatter(all_ks, mean_scores, color='orange')
            plt.errorbar(all_ks, mean_scores, yerr=asymmetric_error, label=clustering_algorithm_name, color=color)

            plt.ylim(ymin, ymax)
            plt.xlabel('Number of Parcels'), plt.ylabel(ylabel), plt.title(title)

            plt.savefig(output_dir + '/'+fn+'_'+metric+'_'+clustering_algorithm+'_errors.png', bbox_inches='tight', dpi=300)
            plt.clf()

    # Save in json format
    original_intrinsic_scores = copy.deepcopy(intrinsic_scores)
    deep_str(intrinsic_scores) # convert float32 values to strings
    with open(output_dir + '/' + fn + '.json', 'w') as json_file:
        json.dump(intrinsic_scores, json_file, indent=4)

    return original_intrinsic_scores

def perform_clustering_fixed(scores, whitening=False, k=None):
    # Apply whitening if enabled
    if whitening:
        pca = PCA(whiten=True)
        scores = pca.fit_transform(scores)

    # Determine the optimal cluster count
    kmeans_labels = KMeans(n_clusters=k, n_init=10).fit(scores).labels_
    agglom_labels = AgglomerativeClustering(n_clusters=k).fit(scores).labels_

    return kmeans_labels, agglom_labels

"""
1. Define the required variables
"""
dataset_dir = input("Name of dataset folder (e.g. new_dataset):")
dataset_dir = './' + dataset_dir
num_clusters, num_features = 97, 20

parser = ArgumentParser(description="Arguments for model training.")
parser.add_argument("-s", "--save_name", help="Save directory (e.g. './visualisations')", type=str)
parser.add_argument("-r", "--results_name", help="The name of the model to load.", type=str, default='cerebellum_optimised_transformer')
parser.add_argument("-b", "--bilateral", help="Presence of this flag enables bilateral clustering.", action='store_true')
parser.add_argument("-w", "--whitening", help="Presence of this flag enables whitening of data before clustering.", action='store_true')
parser.add_argument("-l", "--load", help="Presence of flag will load saliencies from 'all_saliencies.npy', 'tract_names.json', 'cluster_names.json' instead of calculating them", action='store_true')
parser.add_argument("-m", "--model", help="Indicates which model to use (transformer, 1dcnn, fc).", type=str, default="transformer")
args = parser.parse_args()

OUTPUT_BASE = args.save_name
BILATERAL = args.bilateral
WHITENING = args.whitening
RESULTS_NAME = args.results_name

input_channels = num_clusters * num_features
device = torch.device('cuda')

feature_names = ['FA1.Min', 'FA1.Max', 'FA1.Median', 'FA1.Mean', 'FA1.Variance', 'FA2.Min', 'FA2.Max', 'FA2.Median', 'FA2.Mean', 'FA2.Variance', 'NoS', 'NoP', 'Trace1.Min', 'Trace1.Max', 'Trace1.Median', 'Trace1.Mean', 'Trace2.Min', 'Trace2.Max', 'Trace2.Median', 'Trace2.Mean']

cluster_names = ["left_hemisphere.cluster_00118","left_hemisphere.cluster_00120","left_hemisphere.cluster_00493","left_hemisphere.cluster_00566","right_hemisphere.cluster_00118","right_hemisphere.cluster_00120","right_hemisphere.cluster_00493","right_hemisphere.cluster_00566","commissural.cluster_00110","commissural.cluster_00111","commissural.cluster_00115","commissural.cluster_00520","commissural.cluster_00523","commissural.cluster_00526","commissural.cluster_00544","commissural.cluster_00546","commissural.cluster_00550","left_hemisphere.cluster_00126","left_hemisphere.cluster_00130","left_hemisphere.cluster_00515",
        "right_hemisphere.cluster_00126","right_hemisphere.cluster_00130","right_hemisphere.cluster_00515","left_hemisphere.cluster_00109","left_hemisphere.cluster_00496","left_hemisphere.cluster_00498","left_hemisphere.cluster_00501","left_hemisphere.cluster_00504","left_hemisphere.cluster_00514","left_hemisphere.cluster_00527","left_hemisphere.cluster_00537","left_hemisphere.cluster_00540","left_hemisphere.cluster_00570","left_hemisphere.cluster_00572","left_hemisphere.cluster_00575","right_hemisphere.cluster_00109","right_hemisphere.cluster_00496","right_hemisphere.cluster_00498","right_hemisphere.cluster_00501","right_hemisphere.cluster_00504","right_hemisphere.cluster_00514","right_hemisphere.cluster_00527","right_hemisphere.cluster_00537","right_hemisphere.cluster_00540","right_hemisphere.cluster_00570","right_hemisphere.cluster_00572","right_hemisphere.cluster_00575","left_hemisphere.cluster_00495","left_hemisphere.cluster_00500","left_hemisphere.cluster_00502","left_hemisphere.cluster_00503","left_hemisphere.cluster_00505","left_hemisphere.cluster_00508","left_hemisphere.cluster_00510","left_hemisphere.cluster_00512","left_hemisphere.cluster_00513","left_hemisphere.cluster_00516","left_hemisphere.cluster_00519","left_hemisphere.cluster_00521","left_hemisphere.cluster_00528","left_hemisphere.cluster_00529","left_hemisphere.cluster_00536","left_hemisphere.cluster_00538","left_hemisphere.cluster_00539","left_hemisphere.cluster_00541","left_hemisphere.cluster_00548","left_hemisphere.cluster_00551","left_hemisphere.cluster_00552","left_hemisphere.cluster_00562","left_hemisphere.cluster_00563","left_hemisphere.cluster_00565","left_hemisphere.cluster_00571","right_hemisphere.cluster_00495","right_hemisphere.cluster_00500","right_hemisphere.cluster_00502","right_hemisphere.cluster_00503","right_hemisphere.cluster_00505","right_hemisphere.cluster_00508","right_hemisphere.cluster_00510","right_hemisphere.cluster_00512","right_hemisphere.cluster_00513","right_hemisphere.cluster_00516","right_hemisphere.cluster_00519","right_hemisphere.cluster_00521","right_hemisphere.cluster_00528","right_hemisphere.cluster_00529","right_hemisphere.cluster_00536","right_hemisphere.cluster_00538","right_hemisphere.cluster_00539","right_hemisphere.cluster_00541","right_hemisphere.cluster_00548","right_hemisphere.cluster_00551","right_hemisphere.cluster_00552","right_hemisphere.cluster_00562","right_hemisphere.cluster_00563","right_hemisphere.cluster_00565","right_hemisphere.cluster_00571"]
tract_names = ["SCP" for i in range(8)] + ["MCP" for i in range(9)] + ["ICP" for i in range(6)] + ["IP" for i in range(24)] + ["PF" for i in range(50)]

metrics = ['Endurance_AgeAdj', 'Strength_AgeAdj', 'ReadEng_AgeAdj', 'PicVocab_AgeAdj'] 
metric_names = ['Endurance', 'Strength', 'ReadEng', 'PicVocab']
categories = ['motor', 'motor', 'cognitive', 'cognitive']

# Plot just the tracts themselves for reference
tract_labels = tract_to_id(tract_names)

"""
2. The Main loop
"""
if args.load:
    all_data = np.load('all_saliencies.npy')
    with open('cluster_names.json', 'r') as f:
        cluster_names = json.load(f)
    with open('tract_names.json', 'r') as f:
        tract_names = json.load(f)
else:
    all_data = []
    for fold in range(5): # Loop over all folds of cross-validation
        t0 = time.time()

        dataset_metric = 'all'

        # Load the dataset
        train_val_dataset = CustomDataset(dataset_dir=dataset_dir, folds=[x for x in range(5) if x != fold], metric=dataset_metric)
        train_mean, train_stdev = train_val_dataset.get_stats()
        test_dataset = CustomDataset(dataset_dir=dataset_dir, folds=[fold], metric=dataset_metric, mean=train_mean, stdev=train_stdev)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, drop_last=True)

        # Load the model 
        checkpoint = torch.load('./results/'+RESULTS_NAME+'/fold_'+str(fold)+'_best_model_checkpoint.pth')
        if args.model == 'transformer':
            model = TransformerModel(input_channels=input_channels, dropout=0)
        elif args.model == '1dcnn':
            model = CNN(input_channels=input_channels, dropout=0, num_classes=11)
        elif args.model == 'fc':
            model = FC(input_channels=input_channels, dropout=0, num_classes=11)
        else:
            print('Error: Invalid model.')
            sys.exit()

        model.to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Set the saliency type
        saliency_map = SaliencyMap(model)
        #saliency_map = GuidedBackprop(model)

        # Loop over all the NIH metrics
        all_metric_data = []
        for metric in metrics:
            # Loop over all of the test data
            all_subject_data = []
            for data, _ in testloader:
                # Send to GPU
                data = data.to(device)

                # Compute the salience
                salience = np.abs(saliency_map.get_mask(image_tensor=data, category=metric))

                # Compute the average for each cluster and feature
                cluster_mean = norm(get_cluster_mean(salience, num_features))

                all_feature_scores = salience.reshape(-1,20)

                # Average the left/right hemisphere scores
                if BILATERAL:
                    cluster_mean, new_names, new_tract_names = average_hemispheres(cluster_mean, cluster_names, tract_names)
                    all_feature_scores, _, _ = average_hemispheres(all_feature_scores, cluster_names, tract_names)

                all_subject_data.append(norm(all_feature_scores))

            all_metric_data.append(all_subject_data)

        all_data.append(all_metric_data)
        print("%.1f seconds for fold %d" % (time.time() - t0, fold))

    if BILATERAL:
        cluster_names = new_names
        tract_names = new_tract_names

    all_data = np.array(all_data) # num_folds x num_metrics x num_subjects_per_fold x num_clusters x num_features

    # Save saliences
    np.save('all_saliencies.npy', all_data)
    with open('cluster_names.json', 'w') as f:
        json.dump(cluster_names, f)
    with open('tract_names.json', 'w') as f:
        json.dump(tract_names, f)

avg_over_subjects = np.mean(all_data, axis=(0,2)).transpose(1,0,2) # 53 x 4 x 20

# Generate an image for each task
plot_saliency_grid(avg_over_subjects, metric_names, categories, output_dir=OUTPUT_BASE)

# Plot the feature means over each fold
# Input shape is 5 x 20
plot_feature_saliencies(np.mean(all_data,axis=(1,2,3)), names=feature_names, output_dir=OUTPUT_BASE+'/saliency_plots', output_fn='feature_means', title='Overall', ymax=0.35) 
for metric_id, metric_name in enumerate(metric_names):
    plot_feature_saliencies(np.mean(all_data,axis=(2,3))[:,metric_id,:], names=feature_names, output_dir=OUTPUT_BASE+'/saliency_plots', output_fn=metric_name + '_feature_means', title=metric_name, ymax=0.35) 

# Calculate the mean within-task cosine-similarities (with std. across folds)
flat_vectors = all_data.reshape(5,4,213,-1) # flatten into 1060-element vectors
print("\nWithin-task mean pairwise cosine similarity")
print("Fold\tend.\tstr.\tread\tpicv")
all_sims = []
for fold in range(5):
    fold_data = []
    for metric in range(4):
        metric_vectors = []
        for subject in range(213):
            metric_vectors.append(flat_vectors[fold,metric,subject])
        sim = mean_cosine_similarity(metric_vectors)
        fold_data.append(sim[0])
    all_sims.append(fold_data)
    print("%d:\t%.2f\t%.2f\t%.2f\t%.2f" % (fold, fold_data[0], fold_data[1], fold_data[2], fold_data[3]))
all_sims = np.array(all_sims)
sim_means, sim_stdevs = np.mean(all_sims,0), np.std(all_sims,0)
print("\n--\nWithin-task mean pairwise cosine similarity (avg/std. across folds)")
for i in range(len(sim_means)):
    print("%s\t%.2f (+-%.2f)" % (metrics[i], sim_means[i], sim_stdevs[i]))

# Calculate the standard deviation across folds for each task's mean vector
per_fold_vectors = np.mean(all_data,axis=2).reshape(5,4,-1)
print('Per-Task Standard Deviation Across Folds')
for i in range(len(metrics)):
    metric = metrics[i]
    sim = mean_cosine_similarity(list(per_fold_vectors[:,i,:]))
    print("%s:\t\t%.2f (std)" % (metric, np.std(per_fold_vectors[:,i,:])))

# Plot the mean saliency for each tract
plot_tract_saliencies(np.mean(all_data,axis=(1,2)), tract_names, output_dir=OUTPUT_BASE+'/saliency_plots', output_fn='tract_scores')

# Form a reference vector for saliency calculation for parcels
reference_vector = norm(np.mean(avg_over_subjects,-1)) # 53 x 4

# Re-shape to form the desired vectors
eighty_elements = avg_over_subjects.reshape(-1,80)
eight_element_pca = PCA(n_components=8).fit_transform(eighty_elements)
four_element_pca = PCA(n_components=4).fit_transform(eighty_elements)
four_elements = np.mean(avg_over_subjects,-1)
category_scores = get_category_scores(four_elements.copy(), categories)
motor_scores, cognitive_scores = np.array(category_scores['motor']), np.array(category_scores['cognitive'])
two_elements = np.array([category_scores['motor'], category_scores['cognitive']]).T
displacements = np.array([(motor_scores[i]-cognitive_scores[i])/((2)**(1/2)) for i in range(len(motor_scores))])
displacements = displacements / np.max(np.abs(displacements)) # normalise by setting abs. max to 1
three_elements = np.array([motor_scores, cognitive_scores, displacements]).T

# Calculate the motor-cognitive difference for each tract
for tract_name in ['ICP', 'MCP', 'SCP', 'IP', 'PF']:
    tract_clusters = []
    for i in range(len(tract_names)):
        if tract_names[i] == tract_name:
            tract_clusters.append(cluster_names[i])

    all_cluster_scores = {}
    for name in tract_clusters:
        all_cluster_scores[name] = [motor_scores[cluster_names.index(name)], cognitive_scores[cluster_names.index(name)]]

    motor_cog_score = []
    for name in tract_clusters:
        m,c = all_cluster_scores[name]
        motor_cog_score.append([m,c])
    motor_cog_score = np.array([(x[0]-x[1])/(x[0]+x[1]) for x in motor_cog_score])
    motor_cog_score = motor_cog_score / np.max(np.abs(motor_cog_score))
    motor_cog_score = motor_cog_score/2+0.5

    gen_model(motor_cog_score, 'tract', tract_name+'_mixed', tract_clusters, 'custom_bwr', opacity=0.1, output_dir=OUTPUT_BASE+'/vis')

# Plot the per-category saliency for each tract
plot_per_category_saliency(all_data, tract_names, categories, cluster_names, output_dir=OUTPUT_BASE+'/saliency_plots', ymax=0.35)

all_vector_representation_scores = {}

encodings = [eighty_elements, eight_element_pca, four_element_pca, four_elements, three_elements, two_elements]
save_names = ['80_elements', '8_elements_pca', '4_elements_pca', '4_elements', '3_elements', '2_elements']
plot_titles = ['Original Vector\n(80 elements)', 'PCA\n(8 elements)', 'PCA\n(4 elements)', 'Mean of Each\nNIH Measure\n(4 elements)', 'Category\nDisplacements\n(3 elements)', 'Mean of Categories\n(2 elements)']
target_representations = [4]

for encoding_index, representation in enumerate(encodings):
    save_name = save_names[encoding_index]
    plot_title = plot_titles[encoding_index]
    output_folder = OUTPUT_BASE + '/' + save_name

    # Generate the parcel count graphs
    num_clusters_graph_data = generate_clustering_plots(representation, tract_names=tract_names, output_dir=output_folder+'/graphs', fn=save_name+'_scores', title=plot_title, whitening=WHITENING)
    all_vector_representation_scores[save_name] = copy.deepcopy(num_clusters_graph_data)

    # Peform clustering for the target representations
    if encoding_index in target_representations:
        for k in [2,4,5,8]:
            cluster_output_folder = output_folder + '/' + str(k)
            labels, _ = perform_clustering_fixed(representation, whitening=WHITENING, k=k)

            # Write the cluster counts for each parcel
            write_string = ""
            for label in sorted(list(set(list(labels)))):
                write_string += "%d clusters in parcel %d\n" % (list(labels).count(label), label)
            write_to_file(write_string, cluster_output_folder, 'cluster_counts.txt')

            # Save the .mrml files
            gen_model(norm(labels), 'clustered', 'all', cluster_names, 'rainbow', opacity=0.1, output_dir=cluster_output_folder)
            vis_kmeans_clusters(labels, cluster_names, output_dir=cluster_output_folder+"/clustered/each_kmeans_cluster")
            vis_tracts(labels, cluster_names, tract_names, output_dir=cluster_output_folder+"/clustered/clusters_in_each_tract")

            # Write the cluster IDs for each parcel
            parcel_cluster_mapping = {str(label): [] for label in set(labels)}
            for target_label in set(labels):
                for i, curr_label in enumerate(labels):
                    if curr_label == target_label:
                        parcel_cluster_mapping[str(target_label)].append([cluster_names[i],tract_names[i]])
            with open(cluster_output_folder + '/clustered/each_kmeans_cluster/cluster_assignments.json', 'w') as json_file:
                json.dump(parcel_cluster_mapping, json_file, indent=4)

            # Save the bar graphs
            gen_parcel_stats(labels, reference_vector, metrics, tract_names, categories, output_dir=cluster_output_folder+'/clustered/each_kmeans_cluster')

            # Compute and store statistics for later analysis
            if k == 4:
                print_parcels=True
            else:
                print_parcels=False
            parcel_analysis(labels, avg_over_subjects, metric_names, cluster_names, feature_names, tract_names, categories, output_dir=cluster_output_folder+'/clustered/stats', print_parcels=print_parcels)

# Generate the violin plots
krange = sorted(list(all_vector_representation_scores[save_names[0]]['silhouette']['kmeans'].keys()))
all_scores = []
for save_name in save_names:
    all_scores.append([all_vector_representation_scores[save_name]['silhouette']['kmeans'][k] for k in krange])
plot_violin(all_scores, OUTPUT_BASE+'/violin_plots', 'violin_kmeans', 'K-Means', 'Silhouette Coefficient', 0, 0.7, plot_titles)

all_scores = []
for save_name in save_names:
    all_scores.append([all_vector_representation_scores[save_name]['silhouette']['agglomerative'][k] for k in krange])
plot_violin(all_scores, OUTPUT_BASE+'/violin_plots', 'violin_agglomerative', 'Agglomerative', 'Silhouette Coefficient', 0, 0.7, plot_titles)

concatenate_images(OUTPUT_BASE + '/violin_plots/violin_kmeans.png', OUTPUT_BASE + '/violin_plots/violin_agglomerative.png', output_dir=OUTPUT_BASE+'/violin_plots', output_fn='merged')
