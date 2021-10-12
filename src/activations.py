"""
Analyze activation-based clustering and compare to weight-based
"""

import sys
import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.metrics.cluster import normalized_mutual_info_score
from scipy.stats import spearmanr, kendalltau, entropy
from sklearn.cross_decomposition import CCA
from pathos.multiprocessing import ProcessPool
import copy
import pickle
import time
from pathlib import Path
from classification_models.keras import Classifiers
from sacred import Experiment
from sacred.observers import FileStorageObserver
from src.utils import load_weights
from src.cnn.extractor import extract_cnn_weights_filters_as_units
from src.cnn import CNN_MODEL_PARAMS, CNN_VGG_MODEL_PARAMS
from src.generate_datasets import prep_imagenet_validation_data
from src.utils import (load_model2, suppress, all_logging_disabled, splitter, combine_ps,
                       chi2_categorical_test, compute_pvalue, imagenet_downsampled_dataset)
from src.spectral_cluster_model import (weights_array_to_cluster_quality, weights_to_graph,
    delete_isolated_ccs_refactored, compute_ncut, get_inv_avg_commute_time)
from src.pointers import DATA_PATHS

# set up some sacred stuff
activations_experiment = Experiment('activations_model')
activations_experiment.observers.append((FileStorageObserver.create('activations_runs')))

activations_cluster = Experiment('activations_clust')
activations_cluster.observers.append((FileStorageObserver.create('activations_runs2')))

RANDOM_STATE = 42


@activations_cluster.config
def my_config2():
    eigen_solver = 'arpack'
    assign_labels = 'kmeans'
    epsilon = 1e-8
    n_workers = 10
    n_outputs = 10
    corr_type = 'spearman'  # must be in ['kendall', 'pearson', 'spearman']
    n_samples = 0
    with_shuffle = False
    exclude_inputs = True
    local = False
    local_layerwise = True
    lucid = False


def load_train_data(model_path, dataset_name='', max_size=5000):

    if 'vgg' in model_path.lower():
        width, height = 32, 32
        depth = 3
        if not dataset_name:
            dataset_name = 'cifar10_full'
    else:
        if not dataset_name:
            dataset_name = 'mnist'
        width, height = 28, 28
        depth = 1
    data_path = DATA_PATHS[dataset_name]
    size = width * height

    try:
        with open(data_path, 'rb') as f:
            dataset = pickle.load(f)
    except:
        with open('.'+data_path, 'rb') as f:
            dataset = pickle.load(f)
    X_train = dataset['X_train']
    y_train = tf.keras.utils.to_categorical(dataset['y_train'])
    assert y_train.shape[-1] == 10

    if X_train.shape[0] > max_size:
        rdm_idxs = np.random.choice(X_train.shape[0], size=(max_size,), replace=False)
        X_train = X_train[rdm_idxs]
        y_train = y_train[rdm_idxs]

    if (X_train.min() == 0 and X_train.min() == 0 and X_train.max() <= 255 and X_train.max() >= 250):
        X_train = X_train / 255
    else:
        raise ValueError('X_train and X_test should be in the range [0, 255] or [0, 1].')
    assert X_train.min() == 0
    assert X_train.max() <= 1
    assert X_train.max() >= 0.95

    if 'cnn' in model_path:
        if 'stacked' in dataset_name:
            X_train = np.transpose(X_train, (0, 2, 3, 1))
        else:
            X_train = X_train.reshape([-1, height, width, depth])

        assert X_train.shape[-3:] == (height, width, depth)

    elif 'mlp' in model_path:
        X_train = X_train.reshape([-1, size])

    return X_train, y_train


def get_corr_adj(activations_mat, corr_type):

    # kendall has less gross error sensitivity and slightly smaller empirical variance
    # https://www.tse-fr.eu/sites/default/files/medias/stories/SEMIN_09_10/STATISTIQUE/croux.pdf
    # but spearman is much faster to compute

    # get the pearson, kendall, and spearman r^2 values from the activations matrix where rows=units, cols=examples
    n_units = activations_mat.shape[0]
    if corr_type == 'pearson':
        corr_mat = np.corrcoef(activations_mat, rowvar=True)
    elif corr_type == 'spearman':
        corr_mat, _ = spearmanr(activations_mat, axis=1)  # pearson r of ranks
    elif corr_type == 'kendall':
        corr_mat = np.diag(np.ones(n_units))  # n_concordant_pair - n_discordant_pair / n_choose_2
        for i in range(n_units):
            for j in range(i):
                kendall_tau, _ = kendalltau(activations_mat[i], activations_mat[j])
                corr_mat[i, j] = kendall_tau
                corr_mat[j, i] = kendall_tau
    else:
        raise ValueError("corr_type must be in ['kendall', 'pearson', 'spearman']")
    assert corr_mat.shape == (n_units, n_units)

    corr_adj = corr_mat**2
    np.fill_diagonal(corr_adj, 0)

    corr_adj = np.nan_to_num(corr_adj)
    corr_adj[corr_adj < 0] = 0
    corr_adj[corr_adj > 1] = 1

    return corr_adj


def shuffle_and_cluster_activations(n_samples, corr_adj, n_clusters,
                                    eigen_solver, assign_labels, epsilon):

    n_units = corr_adj.shape[0]
    shuff_ncuts = []

    time_str = str(time.time())
    dcml_place = time_str.index('.')
    time_seed = int(time_str[dcml_place + 1:])
    np.random.seed(time_seed)

    for _ in range(n_samples):
        # shuffle all edges
        corr_adj_shuff = np.zeros((n_units, n_units))
        upper_tri = np.triu_indices(n_units, 1)
        edges = corr_adj[upper_tri]
        np.random.shuffle(edges)
        corr_adj_shuff[upper_tri] = edges
        corr_adj_shuff = np.maximum(corr_adj_shuff, corr_adj_shuff.T)

        # cluster
        shuffled_ncut, _ = weights_array_to_cluster_quality(None, corr_adj_shuff, n_clusters,
                                                            eigen_solver, assign_labels, epsilon,
                                                            is_testing=False)
        shuff_ncuts.append(shuffled_ncut)

    return np.array(shuff_ncuts)


def do_clustering_activations(network_type, activations_path, activations_mask_path, local, local_layerwise,
                              corr_type, n_clusters, n_inputs, n_outputs, exclude_inputs, eigen_solver,
                              assign_labels, epsilon, n_samples, with_shuffle, n_workers):

    with open(activations_path, 'rb') as f:
        activations = pickle.load(f)
    with open(activations_mask_path, 'rb') as f:
        activations_mask = pickle.load(f)

    if 'cnn' in network_type:  # for the cnns, only look at conv layers
        # if 'stacked' in str(activations_path).lower():
        #     n_in = n_inputs * 2
        # else:
        #     n_in = n_inputs
        # cnn_params = CNN_VGG_MODEL_PARAMS if 'vgg' in network_type else CNN_MODEL_PARAMS
        # n_conv_filters = sum([cl['filters'] for cl in cnn_params['conv']])
        # n_start = np.sum(activations_mask[:n_in])
        # n_stop = n_start + np.sum(activations_mask[n_in: n_in+n_conv_filters])
        # activations = activations[n_start:n_stop, :]
        # activations_mask = activations_mask[n_in: n_in+n_conv_filters]
        pass
    elif exclude_inputs:
        n_in = n_inputs
        n_start = np.sum(activations_mask[:n_in])
        activations = activations[n_start: -n_outputs, :]
        activations_mask = activations_mask[n_in: -n_outputs]

    if local:

        assert exclude_inputs

        if 'cnn' in str(activations_path).lower():
            cnn_params = CNN_VGG_MODEL_PARAMS if 'vgg' in str(activations_path).lower() else CNN_MODEL_PARAMS
            layer_sizes = [cl['filters'] for cl in cnn_params['conv']]
        else:  # it's an mlp
            layer_sizes = [256, 256, 256, 256]
        mask_layerwise = list(splitter(activations_mask, layer_sizes))
        masked_sizes = [np.sum(ml) for ml in mask_layerwise]
        acts_layerwise = list(splitter(activations, masked_sizes))

        if local_layerwise:
            corr_adj = get_corr_adj(activations, corr_type)
            _, pre_labels = weights_array_to_cluster_quality(None, corr_adj, n_clusters, eigen_solver,
                                                             assign_labels, epsilon, is_testing=False)
            labels = -1 * np.ones(activations_mask.shape)
            labels[activations_mask] = pre_labels
            labels_in_layers = [np.array(lyr_labels) for lyr_labels in list(splitter(labels, layer_sizes))]
            n_clusters_per_layer = []
            for ll in labels_in_layers:
                ll = ll[ll != -1]
                n_clusters_per_layer.append(len(np.unique(ll)))
        else:
            n_clusters_per_layer = [n_clusters] * len(layer_sizes)

        unshuffled_ncut, pre_labels = [], []
        for layer_i, al in enumerate(acts_layerwise):
            corr_adj = get_corr_adj(np.array(al), corr_type)
            un, cl = weights_array_to_cluster_quality(None, corr_adj, n_clusters_per_layer[layer_i], eigen_solver,
                                                      assign_labels, epsilon, is_testing=False)
            unshuffled_ncut.append(un)
            pre_labels.append(cl)

        unshuffled_ncut = sum(unshuffled_ncut) / len(unshuffled_ncut)
        pre_labels = np.concatenate(pre_labels)

    else:
        corr_adj = get_corr_adj(activations, corr_type)
        unshuffled_ncut, pre_labels = weights_array_to_cluster_quality(None, corr_adj, n_clusters, eigen_solver,
                                                                       assign_labels, epsilon, is_testing=False)

    clustering_labels = -1 * np.ones(activations_mask.shape)
    clustering_labels[activations_mask] = pre_labels

    ave_in_out = (1 - unshuffled_ncut / n_clusters) / (2 * unshuffled_ncut / n_clusters)
    ent = entropy(clustering_labels)
    true_labels = clustering_labels[clustering_labels >= 0]
    label_proportions = np.bincount(true_labels.astype(int)) / len(true_labels)
    result = {'activations': activations, 'mask': activations_mask,
              'ncut': unshuffled_ncut, 'ave_in_out': ave_in_out, 'labels': clustering_labels,
              'label_proportions': label_proportions, 'entropy': ent}

    if with_shuffle and not local:  # don't do this if local
        n_samples_per_worker = n_samples // n_workers
        function_argument = (n_samples_per_worker, corr_adj,
                             n_clusters, eigen_solver,
                             assign_labels, epsilon)
        if n_workers == 1:
            print('No Pool! Single Worker!')
            shuff_ncuts = shuffle_and_cluster_activations(*function_argument)

        else:
            print(f'Using Pool! Multiple Workers! {n_workers}')

            workers_arguments = [[copy.deepcopy(arg) for _ in range(n_workers)]
                                 for arg in function_argument]

            with ProcessPool(nodes=n_workers) as p:
                shuff_ncuts_results = p.map(shuffle_and_cluster_activations,
                                            *workers_arguments)

            shuff_ncuts = np.concatenate(shuff_ncuts_results)

        shuffled_n_samples = len(shuff_ncuts)
        shuffled_mean = np.mean(shuff_ncuts, dtype=np.float64)
        shuffled_stdev = np.std(shuff_ncuts, dtype=np.float64)
        print('BEFORE', np.std(shuff_ncuts))
        percentile = compute_pvalue(unshuffled_ncut, shuff_ncuts)
        print('AFTER', np.std(shuff_ncuts))
        z_score = (unshuffled_ncut - shuffled_mean) / shuffled_stdev

        result.update({'n_samples': shuffled_n_samples,
                       'mean': shuffled_mean,
                       'stdev': shuffled_stdev,
                       'z_score': z_score,
                       'percentile': percentile})
    return result


@activations_cluster.automain
def activations_clustering(activations_path, activations_mask_path, local, local_layerwise, n_clusters,
                           corr_type, exclude_inputs, n_outputs, n_samples, with_shuffle, eigen_solver,
                           assign_labels, epsilon, n_workers, lucid):

    lower_path = str(activations_path).lower()
    if 'cnn_vgg' in lower_path:
        network_type = 'cnn'
        n_inputs = 32**2 * 3
    elif 'cnn' in lower_path:
        network_type = 'cnn_vgg'
        n_inputs = 28**2
    else:
        network_type = 'mlp'
        n_inputs = 28**2
    if lucid and not 'cnn_vgg' in lower_path:
        n_inputs *= 3

    act_cluster_results = do_clustering_activations(network_type, activations_path, activations_mask_path,
                                                    local, local_layerwise, corr_type, n_clusters, n_inputs,
                                                    n_outputs, exclude_inputs, eigen_solver, assign_labels,
                                                    epsilon, n_samples, with_shuffle, n_workers)
    labels = act_cluster_results['labels']
    mask = act_cluster_results['mask']
    n_total = len(mask)
    prop_dead = np.sum(mask) / n_total
    # labels = np.zeros(n_total)
    # labels[mask] = masked_labels
    # labels[np.logical_not(mask)] = -1

    metrics = {'ncut': act_cluster_results['ncut'], 'prop_dead': prop_dead}

    return {'labels': labels, 'metrics': metrics}


def get_max_act_images(model_path, savepath, labels, batch_size=256, n_top=10,
                       min_size=5, max_prop=0.75, n_random=19):

    with suppress(), all_logging_disabled():
        model = load_model2(model_path)
    model_path = str(model_path).lower()
    dset_X, dset_y = load_train_data(model_path)

    if 'mlp' in model_path:
        layer_sizes = [256, 256, 256, 256]
    else:
        cnn_params = CNN_VGG_MODEL_PARAMS if 'vgg' in model_path else CNN_MODEL_PARAMS
        layer_sizes = [cl['filters'] for cl in cnn_params['conv']]
    if 'vgg' in model_path:
        width = 32
        height = 32
    else:
        width = 28
        height = 28
    labels_in_layers = [np.array(lyr_labels) for lyr_labels in list(splitter(labels, layer_sizes))]

    in_dims = width * height
    if len(dset_X.shape) == 4:
        in_dims *= dset_X.shape[-1]
    n_data = dset_X.shape[0]
    n_data -= n_data % batch_size

    inp = model.input  # input placeholder
    if 'cnn' in model_path:
        # outputs = [layer.input for layer in model.layers if 'conv2d' in layer._name]
        pre_relus = [layer.output.op.inputs[0] for layer in model.layers if 'conv2d' in layer._name]
    else:
        # outputs = [layer.input for layer in model.layers if 'dense' in layer._name]
        pre_relus = [layer.output.op.inputs[0] for layer in model.layers if 'dense' in layer._name]

    functor = tf.keras.backend.function([inp, tf.keras.backend.learning_phase()], pre_relus)

    activations_single_batch = functor([dset_X[:batch_size], 0])
    n_layers = len(activations_single_batch)
    activations_dims = [] if 'cnn' in model_path else [(in_dims,)]
    for lyr_single in activations_single_batch:
        shp = np.squeeze(lyr_single).shape
        activations_dims.append((shp[-1],))  # each filter is a unit if a cnn

    activations = [np.zeros(((n_data,) + lyr_dims)) for lyr_dims in activations_dims]

    if 'cnn' in model_path:
        for test_i in range(0, n_data, batch_size):  # iterate through test set
            acts_batch = functor([dset_X[test_i: test_i + batch_size], 0])  # False for eval
            for lyr in range(n_layers):
                activations[lyr][test_i: test_i + batch_size] = np.linalg.norm(acts_batch[lyr], ord=1, axis=(1, 2))
    else:
        for test_i in range(0, n_data, batch_size):  # iterate through test set
            batch_in = dset_X[test_i: test_i + batch_size]
            batch_in = np.reshape(batch_in, (batch_size, -1))
            activations[0][test_i: test_i + batch_size] = batch_in
            acts_batch = functor([dset_X[test_i: test_i + batch_size], 0])  # False for eval
            for lyr in range(n_layers):
                activations[lyr + 1][test_i: test_i + batch_size] = acts_batch[lyr]

    all_act_mat = np.abs(np.hstack(activations).T)  # after taking .T, each row is a unit and each col an example
    all_act_split = [np.array(lyr_acts) for lyr_acts in splitter(all_act_mat, layer_sizes)]

    results = {}
    percentiles = []
    effect_sizes = []
    for layer_i in range(len(labels_in_layers)):
        layer_labels = labels_in_layers[layer_i]
        layer_size = len(layer_labels)
        for label_i in np.sort(np.unique(layer_labels)):
            sm_size = np.sum(layer_labels == label_i)
            if sm_size < min_size or sm_size > max_prop * len(layer_labels):
                continue
            sm_sums = np.sum(all_act_split[layer_i][layer_labels == label_i], axis=0)
            sm_max_i = np.argsort(sm_sums)[-(n_top+1):]
            max_ims = [np.reshape(dset_X[maxi], (width, height, -1)) for maxi in sm_max_i]

            max_labels = np.argmax(dset_y[sm_max_i], axis=1)
            max_labels[-1] = n_top
            rdm_max_labels = []
            rdm_ims = None
            for rdm_i in range(n_random):  # random max results
                rdm_idxs = np.random.choice(np.array(range(layer_size)), size=sm_size, replace=False)
                rdm_sums = np.sum(all_act_split[layer_i][rdm_idxs], axis=0)
                rdm_max_i = np.argsort(rdm_sums)[-(n_top+1):]
                rdm_max_sample = np.argmax(dset_y[rdm_max_i], axis=1)
                rdm_max_sample[-1] = n_top
                rdm_max_labels.append(rdm_max_sample)
                if rdm_i == 0:
                    rdm_ims = [np.reshape(dset_X[maxi], (width, height, -1)) for maxi in rdm_max_i]
            true_props = np.bincount(max_labels)[:-1] / n_top
            rdm_props = [np.bincount(rdm_max)[:-1] / n_top for rdm_max in rdm_max_labels]
            true_entropy = entropy(true_props)
            random_entropies = np.array([entropy(rdm_prop) for rdm_prop in rdm_props])
            percentiles.append(compute_pvalue(true_entropy, random_entropies))
            effect_sizes.append(np.mean(random_entropies)/true_entropy)

            results[f'layer_{layer_i}_label_{int(label_i)}'] = {'size': sm_size,
                                                                'ims': max_ims,
                                                                'rdm_ims': rdm_ims}
    percentiles = np.array(percentiles)
    effect_sizes = np.array(effect_sizes)
    results['fisher_fisher_p'] = combine_ps(percentiles, n_random)
    results['chi2_fisher_p'] = chi2_categorical_test(percentiles, n_random)
    results['mean_effect_size'] = np.mean(effect_sizes)

    with open(savepath, 'wb') as f:
        pickle.dump(results, f)

    return results


def get_labels_imagenet_activations(network, n_clusters, local=False, norm=1,
                                    eigen_solver='arpack', assign_labels='kmeans', n_samples=2000,
                                    batch_size=32, corr_type='spearman', epsilon=1e-8,
                                    data_dir='/project/nn_clustering/datasets/imagenet2012',
                                    val_tar='ILSVRC2012_img_val.tar'):

    net, preprocess = Classifiers.get(network)
    model = net((224, 224, 3), weights='imagenet')
    inp = model.input
    # outputs = [layer.output for layer in model.layers if 'conv' in layer.name]
    if 'resnet' in network:
        conv_idxs = [model.layers.index(cl) for cl in model.layers
                     if '.conv2d' in str(type(cl)).lower()]
        pre_relus = [model.layers[ci].output.op.inputs[0] for ci in conv_idxs[1:]]
        pre_relus.append(model.layers[conv_idxs[-1]].output)
    elif 'vgg' in network:
        pre_relus = [layer.output.op.inputs[0] for layer in model.layers if 'conv' in layer.name]
    else:
        raise ValueError

    functor = tf.keras.backend.function([inp, tf.keras.backend.learning_phase()], pre_relus)

    data_path = Path(data_dir)
    tfrecords = list(data_path.glob('*validation.tfrecord*'))
    if not tfrecords:
        prep_imagenet_validation_data(data_dir, val_tar)  # this'll take a sec
    imagenet = tfds.image.Imagenet2012()  # dataset builder object
    imagenet._data_dir = data_dir
    val_dataset_object = imagenet.as_dataset(split='validation')  # datast object
    dataset, _ = imagenet_downsampled_dataset(val_dataset_object, preprocess, n_images=n_samples)

    activations_single_batch = functor([dataset[:batch_size], 0])  # 0 for eval
    n_layers = len(activations_single_batch)
    activations_dims = []
    for lyr_single in activations_single_batch:
        shp = np.squeeze(lyr_single).shape
        activations_dims.append((shp[-1],))  # each filter is a unit if a cnn

    activations = [np.zeros(((n_samples,) + lyr_dims)) for lyr_dims in activations_dims]

    for test_i in range(0, n_samples, batch_size):  # iter through test set
        acts_batch = functor([dataset[test_i: test_i+batch_size], 0])
        for lyr in range(n_layers):
            activations[lyr][test_i: test_i + batch_size] = np.linalg.norm(acts_batch[lyr], ord=norm, axis=(1, 2))
    del model
    del dataset

    if local:
        masks_layerwise, unshuffled_ncut, clustering_labels = [], [], []
        for am in activations:
            col_stds = np.std(am, axis=0)
            act_mask = col_stds != 0
            masks_layerwise.append(act_mask)
            acts = am.T[act_mask]
            # activations_layerwise.append(acts)

            corr_adj = get_corr_adj(acts, corr_type)
            un, cl = weights_array_to_cluster_quality(None, corr_adj, n_clusters, eigen_solver,
                                                      assign_labels, epsilon, is_testing=False)
            unshuffled_ncut.append(un)
            clustering_labels.append(cl)
        del activations
        unshuffled_ncut = sum(unshuffled_ncut) / len(unshuffled_ncut)
        clustering_labels = np.concatenate(clustering_labels)
        activations_mask = np.concatenate(masks_layerwise)

    else:
        all_act_mat = np.hstack(activations).T  # after taking .T, each row is a unit and each col an example
        row_stds = np.std(all_act_mat, axis=1)
        activations_mask = row_stds != 0
        activations = all_act_mat[activations_mask]
        del all_act_mat
        corr_adj = get_corr_adj(activations, corr_type)
        del activations
        unshuffled_ncut, clustering_labels = weights_array_to_cluster_quality(None, corr_adj, n_clusters,
                                                                              eigen_solver, assign_labels,
                                                                              epsilon, is_testing=False)

    n_total = len(activations_mask)
    prop_dead = np.sum(activations_mask) / n_total
    labels = np.zeros(n_total)
    labels[activations_mask] = clustering_labels
    labels[np.logical_not(activations_mask)] = -1

    print(f'{network}: ncut: {unshuffled_ncut}, prop_dead: {prop_dead}')
    sys.stdout.flush()

    with open(data_dir + f'/{network}_activations_local={local}_k={n_clusters}.pkl', 'wb') as f:
        pickle.dump(labels, f)

    return labels


def get_max_act_images_imagenet(model_tag, savepath, use_activations, n_top=10, norm=1, n_samples=4000,
                                batch_size=32, min_size=5, max_prop=0.75,
                                infodir='/project/nn_clustering/results/',
                                data_dir='/project/nn_clustering/datasets/imagenet2012',
                                val_tar='ILSVRC2012_img_val.tar'):

    # with suppress(), all_logging_disabled():

    if use_activations:
        with open(infodir + model_tag + '_act_clustering_info.pkl', 'rb') as f:
            clustering_info = pickle.load(f)
    else:
        with open(infodir + model_tag + '_clustering_info.pkl', 'rb') as f:
            clustering_info = pickle.load(f)
    labels_in_layers = [np.array(lyr_labels) for lyr_labels in clustering_info['labels']]
    layer_sizes = [len(labels) for labels in labels_in_layers]

    data_path = Path(data_dir)
    tfrecords = list(data_path.glob('*validation.tfrecord*'))
    if not tfrecords:
        prep_imagenet_validation_data(data_dir, val_tar)  # this'll take a sec

    net, preprocess = Classifiers.get(model_tag)
    model = net((224, 224, 3), weights='imagenet')
    inp = model.input
    # outputs = [layer.output for layer in model.layers if 'conv' in layer.name]
    pre_relus = [layer.output.op.inputs[0] for layer in model.layers if 'conv' in layer.name]
    functor = tf.keras.backend.function([inp, tf.keras.backend.learning_phase()], pre_relus)

    data_path = Path(data_dir)
    tfrecords = list(data_path.glob('*validation.tfrecord*'))
    if not tfrecords:
        prep_imagenet_validation_data(data_dir, val_tar)  # this'll take a sec
    imagenet = tfds.image.Imagenet2012()  # dataset builder object
    imagenet._data_dir = data_dir
    val_dataset_object = imagenet.as_dataset(split='validation', shuffle_files=True)  # datast object
    dataset, dataset_y = imagenet_downsampled_dataset(val_dataset_object, preprocess,
                                                      n_images=n_samples)

    activations_single_batch = functor([dataset[:batch_size], 0])  # 0 for eval
    n_layers = len(activations_single_batch)
    activations_dims = []
    for lyr_single in activations_single_batch:
        shp = np.squeeze(lyr_single).shape
        activations_dims.append((shp[-1],))  # each filter is a unit if a cnn

    activations = [np.zeros(((n_samples,) + lyr_dims)) for lyr_dims in activations_dims]

    for test_i in range(0, n_samples, batch_size):  # iter through test set
        acts_batch = functor([dataset[test_i: test_i + batch_size], 0])
        for lyr in range(n_layers):
            activations[lyr][test_i: test_i + batch_size] = np.linalg.norm(acts_batch[lyr], ord=norm, axis=(1, 2))
    del model

    all_act_mat = np.abs(np.hstack(activations).T)  # after taking .T, each row is a unit and each col an example
    assert len(all_act_mat) == sum([len(layer_labels) for layer_labels in labels_in_layers]), \
        f'all_act_mat len {len(all_act_mat)} not compatible with layer_label lens' \
        f'{[len(layer_labels) for layer_labels in labels_in_layers]} with layer_sizes {layer_sizes}'
    all_act_split = [np.array(lyr_acts) for lyr_acts in list(splitter(all_act_mat, layer_sizes))]

    print(f'labels_in_layers: {[len(layer) for layer in labels_in_layers]}')
    print(f'all_act_split: {[len(layer) for layer in all_act_split]}')

    results = {}
    for layer_i in range(len(labels_in_layers)):
        assert len(labels_in_layers[layer_i]) == len(all_act_split[layer_i])
        layer_labels = labels_in_layers[layer_i]
        layer_size = len(layer_labels)
        for label_i in np.sort(np.unique(layer_labels)):
            sm_size = np.sum(layer_labels == label_i)
            if sm_size < min_size or sm_size > max_prop * len(layer_labels):
                continue
            sm_sums = np.sum(all_act_split[layer_i][layer_labels == label_i], axis=0)
            sm_max_i = np.argsort(sm_sums)[-n_top:]
            rdm_idxs = np.random.choice(np.array(range(layer_size)), size=sm_size, replace=False)
            rdm_sums = np.sum(all_act_split[layer_i][rdm_idxs], axis=0)
            rdm_max_i = np.argsort(rdm_sums)[-n_top:]

            # max_labels = dataset_y[sm_max_i]
            # rdm_max_labels = []
            # for _ in range(n_random):  # random max results
            #     rdm_idxs = np.random.choice(np.array(range(layer_size)), size=sm_size, replace=False)
            #     rdm_sums = np.sum(all_act_split[layer_i][rdm_idxs], axis=0)
            #     rdm_max_i = np.argsort(rdm_sums)[-n_top:]
            #     rdm_max_labels.append(dataset_y[rdm_max_i])

            results[f'layer_{layer_i}_label_{int(label_i)}'] = {'size': sm_size,
                                                                'ims': dataset[sm_max_i],
                                                                'rdm_ims': dataset[rdm_max_i]}
    del dataset

    with open(savepath, 'wb') as f:
        pickle.dump(results, f)

    return results

