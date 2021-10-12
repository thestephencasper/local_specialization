import pickle
import tempfile
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from src.visualization import run_spectral_cluster
from src.utils import load_weights, load_model2, splitter
from src.spectral_cluster_model import weights_to_graph
from src.cnn import CNN_MODEL_PARAMS, CNN_VGG_MODEL_PARAMS
from src.cnn.extractor import extract_cnn_weights_filters_as_units


def run_local_clustering(weights_path, n_clusters, local_layerwise=True):
    """Generate clustering labels for each bipartite subgraph.
    
    The bipartite subgraph is three layers.
    E.g., if the network consists of X Y Z W U V layers, then its
    bipartite clustering is for the subgraphs:
    XYZ, YZW, ZWU, WUV.
    
    Parameters:
    -----------
    weights_path : str, path
    n_clusters : int
    with_shuffle : bool
    n_samples : int
    
    Returns:
    --------
    bipartite_labels : list
        The clustering labels of each bipartite subgraph.
    bipartite_metrics : list
        The clustering metrics (e.g., ncut) of each bipartite subgraph.
    """

    weights_path = Path(weights_path)
    weights = load_weights(weights_path)

    if 'cnn' in str(weights_path).lower():  # for the cnns, only look at conv layers
        cnn_params = CNN_VGG_MODEL_PARAMS if 'vgg' in str(weights_path).lower() else CNN_MODEL_PARAMS
        layer_sizes = [cl['filters'] for cl in cnn_params['conv']]
        with_batch_norm = 'vgg' in str(weights_path).lower()
        if any(len(wgts.shape) > 2 for wgts in weights):
            weights = extract_cnn_weights_filters_as_units(weights, norm=1, with_batch_norm=with_batch_norm)
        n_conv_layers = len(cnn_params['conv'])
        weights = weights[1:n_conv_layers]
    else:
        layer_sizes = [256, 256, 256, 256]

    bipartite_labels = []
    bipartite_metrics = []

    if local_layerwise:
        labels, _ = run_spectral_cluster(weights_path, n_clusters=n_clusters, with_shuffle=False)
        labels_in_layers = [np.array(lyr_labels) for lyr_labels in list(splitter(labels, layer_sizes))]
        n_clusters_per_layer = []
        for ll in labels_in_layers:
            ll = ll[ll != -1]
            n_clusters_per_layer.append(len(np.unique(ll)))
    else:
        n_clusters_per_layer = [n_clusters] * len(weights)

    with tempfile.TemporaryDirectory() as tmpdirname:
        for layer_id, bipartite_net in enumerate(zip(weights, weights[1:])):

            bipartite_net_filename = f'{weights_path.stem}-tmp-local-{layer_id}{weights_path.suffix}'
            bipartite_net_path = Path(tmpdirname) / bipartite_net_filename
            with open(bipartite_net_path, 'wb') as f:
                pickle.dump(bipartite_net, f)

            labels, metrics = run_spectral_cluster(bipartite_net_path,
                                                   n_clusters=n_clusters_per_layer[layer_id],
                                                   with_metrics=False, with_shuffle=False)

            assert bipartite_net[0].shape[1] == bipartite_net[1].shape[0]
            n_before = bipartite_net[0].shape[0]
            n_current = bipartite_net[0].shape[1]
            n_after = bipartite_net[1].shape[1]

            before_labels = labels[: n_before].copy()
            current_labels = labels[n_before:n_before + n_current].copy()
            after_labels = labels[-n_after:].copy()

            metrics['full_labels'] = bipartite_labels

            bipartite_labels.append((before_labels, current_labels, after_labels))
            bipartite_metrics.append(metrics)

        splitted_local_labels = [bipartite_labels[0][0]] + [labels[1] for labels in bipartite_labels] + [bipartite_labels[-1][2]]
        local_labels = np.concatenate(splitted_local_labels)
            
    return local_labels, bipartite_metrics
            

def align_stochastically_bipartite_labels(bipartite_labels, n_trials=50):
    """Align the clustering labels of the bipartite subgraphs to each other.
    
    The labels are computed independently to each subgraph. If we want to use
    the induced "global" clustering, the labels should be aligned.
    For example, it could be that the same neurons in Y for XYZ and YZW
    would get a different label. To align them, we solve the Assignment problem stochastically
    for pairs of subgraphs at index i and (i+1).
    
    NOTE: It is not clear that this step is necessary.
    
    Parameters:
    -----------
    bipartite_labels : list
        The clustering labels of each bipartite subgraph.
    n_trials : int
        The number of sampled subgraphs to align.
    
    Returns:
    --------
    bipartite_labels : list
        The *aligned* clustering labels of each bipartite subgraph.
    """    
    
    bipartite_labels = deepcopy(bipartite_labels)
    
    bins = np.unique(np.concatenate([np.concatenate(labels) for labels in bipartite_labels]))


    for index in np.random.randint(0, len(bipartite_labels)-1, n_trials):

        mid_fist_mat, _, _ = np.histogram2d(bipartite_labels[index][1], bipartite_labels[index+1][0], (bins, bins))
        last_mid_mat, _, _ = np.histogram2d(bipartite_labels[index][2], bipartite_labels[index+1][1], (bins, bins))
        cost = mid_fist_mat + last_mid_mat

        row_indices, col_indices = linear_sum_assignment(cost, maximize=True)

        new_bipartite_labels = []
        for labels in bipartite_labels[index+1]:
            new_labels = labels.copy()
            for r, c in zip(row_indices, col_indices):
                new_labels[labels == bins[r]] = bins[c]
            new_bipartite_labels.append(new_labels)


        bipartite_labels[index+1] = new_bipartite_labels
        
    return bipartite_labels


def run_global_and_local_clustering(weights_path, n_clusters, with_shuffle=False, n_samples=10, to_align=True):
    """Perform local clustering based on the weights of each bipartite subgraph.
    
    The bipartite subgraph is three layers.
    E.g., if the network consists of X Y Z W U V layers, then its
    bipartite clustering is for the subgraphs:
    XY, YZW, ZWU, UV.
    
    The actual local labels are taken from the subgraph where the layer is in the middle
    (e.g., Z is taken from YZW), except for the first and last layers (because they appear
    only in a single subgraph).
    
    Parameters:
    -----------
    weights_path : str, path
    n_clusters : int
    with_shuffle : bool
    n_samples : int
    to_align : bool
    
    Returns:
    --------
    global_labels : list
        The labels of the global ("regular") clustering as a single array.
    local_labels : list
        The labels of the local clustering as a single array.
    splitted_global_labels : list
        The labels of the global ("regular") clustering broken into layers.
    splitted_local_labels : list
        The labels of the local clustering broken into layers.
    """
    
    global_labels, _ = run_spectral_cluster(weights_path,
                                            with_shuffle=with_shuffle,
                                            n_samples=n_samples,
                                            with_metrics=False)

    bipartite_labels, _ = run_local_clustering(weights_path, n_clusters,
                                                    with_shuffle, n_samples)

    if to_align:
        bipartite_labels = align_stochastically_bipartite_labels(bipartite_labels)

    splitted_local_labels = [bipartite_labels[0][0]] + [labels[1] for labels in bipartite_labels] + [bipartite_labels[-1][2]]
    local_labels = np.concatenate(splitted_local_labels)

    splitted_global_labels = list(splitter(global_labels, map(len, splitted_local_labels)))

    # assert ((global_labels == -1) == (local_labels == -1)).all()
    
    return global_labels, local_labels, splitted_global_labels, splitted_local_labels


def visualize_global_local_labels(splitted_global_labels, splitted_local_labels):
    """Visualize the count in labels for each layer and method.
    
    IMPORTANT: The local and global labels are not aligned to each other, except for
    the -1 label (pruned neuron).
    """

    dfs = []

    for index, labels in enumerate(splitted_local_labels):
        dfs.append(pd.DataFrame(labels, columns=['label'])
                   .assign(layer=index, method='local'))
        
    for index, labels in enumerate(splitted_global_labels):
        dfs.append(pd.DataFrame(labels, columns=['label'])
                   .assign(layer=index, method='global'))

    df = pd.concat(dfs, ignore_index=True)

    return sns.catplot(x='label', hue='method', col='layer',
                       data=df, kind='count', col_wrap=3)
