"""Actual experimentation of lesson test."""

import os
from copy import deepcopy
import copy
import cv2 as cv
import tensorflow as tf
import pickle
import tensorflow_datasets as tfds
import itertools as it
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from collections import defaultdict
from src.visualization import (run_spectral_cluster, extract_layer_widths,
                               run_activations_cluster, run_local_clustering)
from src.spectral_cluster_model import run_clustering_imagenet, get_dense_sizes
# from src.local_clustering import run_local_clustering
from src.utils import (splitter, load_model2, preprocess_dataset, fisher_stat,
                       suppress, all_logging_disabled, extract_weights, compute_pvalue,
                       get_weights_paths, get_model_paths, chi2_categorical_test,
                       combine_ps, imagenet_generator, imagenet_downsampled_dataset)
from classification_models.keras import Classifiers
from src.generate_datasets import prep_imagenet_validation_data
import warnings

MIN_SIZE = 2
MAX_PROP = 0.99


def _classification_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    metrics = {'acc_overall': np.diag(cm).sum() / cm.sum()}
    metrics.update({f'acc_{number}': acc
                   for number, acc in enumerate(np.diag(cm) / cm.sum(axis=1))})
    output_props = np.sum(cm, axis=0) / np.sum(cm)
    metrics.update({'output_props': output_props})

    return metrics


def _regression_metrics(y_true, y_pred):

    mses = np.mean((y_true - y_pred) ** 2, axis=0)

    return {'mses': mses}


def _evaluate(model, X, y, task, masks=None):

    if task == 'classification':
        y_pred = (model.predict_classes(X, masks)
                  if masks is not None
                  else model.predict_classes(X))

        return _classification_metrics(y, y_pred)

    else:  # task == 'regression'
        y_pred = model.predict(X)
        return _regression_metrics(y, y_pred)


def _damage_neurons(neurons_in_layers, experiment_model, weights, biases,
                    network_type, inplace=None):
    
    # make sure that the function is called with explicit setting of
    # inplace modification of experiment_model
    assert inplace is True, 'inplace argument should be set to True'

    experiment_weights = deepcopy(weights)
    experiment_biases = deepcopy(biases)

    weight_shapes = [w.shape for w in weights]
    n_conv = sum(len(ws) == 4 for ws in weight_shapes)

    for layer_id, _, damaged_neurons, _ in neurons_in_layers:

        if len(weight_shapes[layer_id - 1]) == 4:  # if a conv layer            
            experiment_weights[layer_id - 1][:, :, :, damaged_neurons] = 0
        else:  # if an fc layer
            experiment_weights[layer_id - 1][:, damaged_neurons] = 0

        experiment_biases[layer_id - 1][damaged_neurons] = 0

        if network_type == 'mlp':  # get all layers
            experiment_model_layers = (layer for layer in experiment_model.layers
                                       if 'dense' in layer._name)
        elif network_type == 'cnn':  # get only conv layers
            experiment_model_layers = (layer for layer in experiment_model.layers
                                       if 'conv2d' in layer._name)

        for layer, new_weights in zip(experiment_model_layers,
                                      zip(experiment_weights, experiment_biases)):
            layer.set_weights(new_weights)


def _layers_labels_gen(network_type, layer_widths, labels, ignore_layers,
                       to_shuffle=False, fixed=None):
    
    layer_data = zip(splitter(deepcopy(labels), layer_widths), layer_widths[:-1])

    if 'cnn' not in network_type:
        next(layer_data)

    for layer_id, (layer_labels, layer_width) in enumerate(layer_data, start=1):
        
        # for pool max
        if (ignore_layers and ignore_layers[layer_id-1]):
            # `layer_id-1` because we set `start=1` for `enumerate`
            continue
            
        layer_labels = np.array(layer_labels)

        # do not shuffle pruned nodes
        if to_shuffle:

            # Don't shuffle pruned neurons
            non_shuffled_mask = (layer_labels != -1)

            # We preform the same operation of unpacking `fixed_layer_label`
            # multiple times, because I wanted to put all the "fixed" processing
            # in one section.
            if fixed is not None:
                fixed_layer_id, fixed_label = fixed
                if fixed_layer_id == layer_id:
                    
                    assert not (~non_shuffled_mask
                                & (layer_labels == fixed_label)).any()
                    
                    non_shuffled_mask &= (layer_labels != fixed_label)
                 
            layer_labels[non_shuffled_mask] = np.random.permutation(layer_labels[non_shuffled_mask])

        yield layer_id, layer_labels


def _single_damaged_neurons_gen(layers_labels_iterable, to_shuffle=False):

    for layer_id, layer_labels in layers_labels_iterable:

        actual_layer_size = (layer_labels != -1).sum()

        for label in (l for l in np.unique(layer_labels) if l != -1):

            if to_shuffle:
                non_shuffled_mask = (layer_labels != -1)
                layer_labels[non_shuffled_mask] = np.random.permutation(layer_labels[non_shuffled_mask])

            damaged_neurons = np.nonzero(layer_labels == label)[0]

            if len(damaged_neurons) < MIN_SIZE or len(damaged_neurons) > (actual_layer_size * MAX_PROP):
                continue
            
            yield (layer_id, label, damaged_neurons, actual_layer_size)


def _double_conditional_damaged_neurons_gen(network_type, layer_widths, labels, to_shuffle):
    
    # assert network_type == 'mlp'

    fixed_iter = _layers_labels_gen(network_type, layer_widths, labels,
                                     ignore_layers=False, to_shuffle=False)

    for fixed_layer_id, fixed_layer_labels in fixed_iter:
        
        fixed_actual_layer_size = (fixed_layer_labels != -1).sum()

        for fixed_label in (l for l in np.unique(fixed_layer_labels) if l != -1):

            fixed_damaged_neurons = np.nonzero(fixed_layer_labels == fixed_label)[0]

            fixed_gen_tuple = (fixed_layer_id, fixed_label, fixed_damaged_neurons, fixed_actual_layer_size)
            
            shuffled_layers_labels_iterable = _layers_labels_gen(network_type, layer_widths, labels,
                                               ignore_layers=False, to_shuffle=to_shuffle,
                                               fixed=(fixed_layer_id, fixed_label))
            
            shuffled_damaged_neurons_iterable = _single_damaged_neurons_gen(shuffled_layers_labels_iterable)

            # There is no meaning to same layer-label in double conditional,
            # which it is always shuffled,
            # so we filter these cases out to reduce computation
            filtered_shuffled_damaged_neurons_iterable = ((layer_id, label, damaged_neurons, actual_layer_size)
                                                          for layer_id, label, damaged_neurons, actual_layer_size
                                                          in shuffled_damaged_neurons_iterable
                                                          if not (layer_id == fixed_layer_id
                                                                  and label == fixed_label))
            
            # fixed is second (i.e., first|second)
            yield from it.product(filtered_shuffled_damaged_neurons_iterable,
                                  [fixed_gen_tuple])


def _damaged_neurons_gen(network_type, layer_widths, labels, ignore_layers,
                         to_shuffle=False, n_way=1, n_way_type='joint', verbose=False):
    assert n_way in (1, 2), 'Currently supporting only single and double lesion test.'
    assert n_way_type in ('joint', 'conditional')

    if n_way_type == 'joint':
        
        # if `n_way=2` and `to_shuffle=True`, then we shuffle the cluster labeling once
        # in each of the layers for one trial of brian damage test
        # so we don't need to worry that we might damage the same neurons
        # in the same layer.
        # If the two neuron groups are from the same layer, they won't overlapped,
        # because they are not "sampled" simultaneously.
        layers_labels_iterable = _layers_labels_gen(network_type, layer_widths, labels,
                                                    ignore_layers, to_shuffle)

        damaged_neurons_iterable = _single_damaged_neurons_gen(layers_labels_iterable, to_shuffle)

        yield from it.combinations_with_replacement(damaged_neurons_iterable, n_way)
    
    elif n_way_type == 'conditional':
        assert n_way == 2, 'Conditional p-value works only with double lesion test.'
        yield from _double_conditional_damaged_neurons_gen(network_type, layer_widths, labels, to_shuffle)


def _apply_lesion_trial(X, y, network_type, experiment_model, weights, biases,
                              layer_widths, labels, ignore_layers, task,
                              to_shuffle=False, n_way=1, n_way_type='joint',
                              verbose=False):
    
    damage_results = []

    for neurons_in_layers in _damaged_neurons_gen(network_type, layer_widths, labels, ignore_layers,
                                                  to_shuffle, n_way, n_way_type, verbose):
        _damage_neurons(neurons_in_layers, experiment_model, weights, biases,
                        network_type, inplace=True)
        
        result = _evaluate(experiment_model, X, y, task)#, masks)

        result['labels_in_layers'] = tuple((layer_id, label) for layer_id, label, _, _ in neurons_in_layers)
        if n_way == 1:
            assert len(neurons_in_layers) == 1
            layer_id, label_id, neuron_ids, _ = neurons_in_layers[0]
            agg_weight_result = agg_weights_by_subcluster(layer_id, label_id, neuron_ids, network_type, weights)
            for direction in ('inbound', 'outbound'):
                result[direction] = agg_weight_result.get(direction)
                            
        damage_results.append(result)

    return damage_results


def _extract_layer_label_metadata(network_type, layer_widths, labels, ignore_layers):
    
    layer_label_metadata = []

    for neurons_in_layers in _damaged_neurons_gen(network_type, layer_widths, labels, ignore_layers,
                                                  to_shuffle=False, n_way=1):
        
        assert len(neurons_in_layers) == 1
        layer_id, label, neurons, actual_layer_size = neurons_in_layers[0]

        layer_label_metadata.append({'layer': layer_id,
                                     'label': label,
                                     'n_layer_label': len(neurons),
                                     'label_in_layer_proportion': len(neurons) / actual_layer_size})

    return layer_label_metadata


def _flatten_single_damage(damage_results):
    assert all(len(result['labels_in_layers']) == 1 for result in damage_results)
    
    for result in damage_results:
        layer_id, label = result['labels_in_layers'][0]
        del result['labels_in_layers']
        result['layer'] = layer_id
        result['label'] = label

    return damage_results


def _perform_lesion_sub_experiment(dataset_path, run_dir, n_clusters=4,
                                   n_shuffles=200, n_side=28, depth=1,
                                   with_random=True, model_params=None,
                                   n_way=1, n_way_type='joint',
                                   activations=False, local=False, unpruned=False,
                                   true_as_random=False, verbose=False):
    if verbose:
        print('Loading data...')

    ds = preprocess_dataset(dataset_path)
    X, y = ds['X_test'], ds['y_test']

    run_dir_path = Path(run_dir)
    weight_paths = get_weights_paths(run_dir_path)
    model_paths = get_model_paths(run_dir_path)
    if unpruned:
        model_path = str(model_paths[True])
        weight_path = str(next(run_dir_path.glob('*-unpruned-weights.pckl')))
        activations_path = str(next(run_dir_path.glob('*-unpruned-activations.pckl')))
        activations_mask_path = str(next(run_dir_path.glob('*-unpruned-activations_mask.pckl')))
    else:
        model_path = str(model_paths[False])
        weight_path = str(next(run_dir_path.glob('*-pruned-weights.pckl')))
        activations_path = str(next(run_dir_path.glob('*-pruned-activations.pckl')))
        activations_mask_path = str(next(run_dir_path.glob('*-pruned-activations_mask.pckl')))

    if 'mlp' in model_path.lower() or 'poly' in model_path.lower():
        network_type = 'mlp'
    elif 'cnn' in model_path.lower():
        network_type = 'cnn'
        X = np.reshape(X, (-1, n_side, n_side, depth))
        # assert model_params is not None, ('For CNN network type, '
        #                                   'the model_param parameter should be given.')
    else:
        raise ValueError('Network type should be expressed explicitly '
                         'either mlp or cnn in run directory files.')

    if 'poly' in model_path.lower():
        task = 'regression'
    else:
        task = 'classification'

    if verbose:
        print('Running spectral clustering...')

    if activations:
        labels, _ = run_activations_cluster(activations_path, activations_mask_path, n_clusters, local=local)
        if 'mlp' in model_path.lower():
            begin = -1 * np.ones((n_side**2 * depth,))
            end = -1 * np.ones((10,))
            labels = np.concatenate((begin, labels, end))
    else:
        if local:
            labels, _ = run_local_clustering(weight_path, n_clusters=n_clusters)
        else:
            labels, _ = run_spectral_cluster(weight_path, n_clusters=n_clusters, with_shuffle=False)

    if verbose:
        print('Loading model and extracting weights...')

    # os.environ['CUDA_VISIBLE_DEVICES'] = ''

    with suppress(), all_logging_disabled():
        experiment_model = load_model2(model_path)
    weights, biases = extract_weights(experiment_model, with_bias=True)
    ignore_layers = False

    if verbose:
        print('Evaluate original model...')

    evaluation = _evaluate(experiment_model, X, y, task)

    if network_type == 'mlp':
        layer_widths = extract_layer_widths(weights)
    else:
        layer_widths = []
        weight_shapes = [layer_weights.shape for layer_weights in weights]
        n_conv = sum(len(ws) == 4 for ws in weight_shapes)
        layer_widths.extend([weight_shapes[i][-1] for i in range(n_conv)])
        layer_widths.extend([ws[-1] for ws in weight_shapes[n_conv:]])        
    

    # omit non conv layers for lesioning
    # it was needed to agg weights
    if network_type == 'cnn':
        # TODO: ASK CASPER
        #weights = weights[:n_conv]
        #biases = biases[:n_conv]
        layer_widths = layer_widths[:n_conv+1]

     
    if verbose:
        print('Extract metadata...')
    
    metadata = _extract_layer_label_metadata(network_type, layer_widths, labels, ignore_layers)

    if verbose:
        print('Apply lesion trial on the true clustering...')

    true_results = _apply_lesion_trial(X, y, network_type, experiment_model, weights, biases,
                                             layer_widths, labels, ignore_layers, task,
                                             to_shuffle=true_as_random, n_way=n_way, n_way_type=n_way_type,
                                             verbose=verbose)
    if with_random:

        if verbose:
            print('Apply lesion trial on the random clusterings...')
            progress_iter = tqdm
        else:
            progress_iter = iter

        all_random_results = []
        for _ in progress_iter(range(n_shuffles)):
            random_results = _apply_lesion_trial(X, y, network_type, experiment_model, weights, biases,
                                            layer_widths, labels, ignore_layers, task,
                                            to_shuffle=True, n_way=n_way, n_way_type=n_way_type,
                                            verbose=verbose)
            
            all_random_results.append(random_results)

    else:
        all_random_results = None

    if n_way == 1:
        true_results = _flatten_single_damage(true_results)

        all_random_results = ([_flatten_single_damage(result) for result in all_random_results]
                              if all_random_results else None)
        
    return true_results, all_random_results, metadata, evaluation


def perform_lesion_experiment(dataset_path, run_dir, n_clusters=4,
                              n_shuffles=100, n_side=28, depth=1,
                              n_workers=1, n_way=1, n_way_type='joint',
                              activations=False, local=False, with_random=True,
                              model_params=None, unpruned=False,
                              true_as_random=False, verbose=False):
    
    if n_workers == 1:
        if verbose:
            print('Single worker!')
            
        return _perform_lesion_sub_experiment(dataset_path, run_dir, n_clusters=n_clusters,
                                              n_shuffles=n_shuffles, n_side=n_side,
                                              depth=depth, with_random=with_random,
                                              model_params=model_params, n_way=n_way,
                                              activations=activations, local=local,
                                              n_way_type=n_way_type, unpruned=unpruned,
                                              true_as_random=true_as_random, verbose=verbose)
    else:
        raise NotImplementedError('Check CNN Lesion Test Notebook')


def do_lesion_hypo_tests(evaluation, true_results, all_random_results):

    n_submodules = len(true_results)
    n_shuffles = len(all_random_results)

    if 'mses' in true_results[0]:  # if regression

        n_inputs = 2
        coefs = (0, 1)
        exps = (0, 1, 2)
        n_terms = len(exps) ** n_inputs
        n_outputs = len(coefs) ** n_terms
        poly_coefs = np.zeros((n_outputs, n_terms))
        for poly_i, coef_list in enumerate(it.product(coefs, repeat=n_terms)):
            poly_coefs[poly_i] = np.array(coef_list)
        term_exps = [exs for exs in it.product(exps, repeat=n_inputs)]
        n_terms = len(term_exps)

        # random_mses has shape (n_random, n_submodules, n_outputs)
        random_out_raw = np.array([[rand_sm['mses'] for rand_sm in rand_results] for rand_results in all_random_results])
        # true_mses has shape (n_submodules, n_outputs)
        true_out_raw = np.array([true_sm['mses'] for true_sm in true_results])
        # eval_out_raw has shape (n_outputs,)
        eval_out_raw = evaluation['mses']

        # random_outs has shape (n_random, n_submodules, n_terms)
        random_outs = np.array(
            [[[np.mean(
                np.array(
                    [rand_sm[output_i] for output_i in range(n_outputs)
                     if poly_coefs[output_i][term_i] == 1]
                )
            )
                for term_i in range(n_terms)]
                for rand_sm in rand_mses]
                for rand_mses in random_out_raw]
        )
        # true_outs has shape (n_submodules, n_terms)
        true_outs = np.array(
            [[np.mean(
                np.array(
                    [true_sm[output_i] for output_i in range(n_outputs)
                     if poly_coefs[output_i][term_i] == 1]
                )
            )
                for term_i in range(n_terms)]
                for true_sm in true_out_raw]
        )
        # eval_outs has shape (n_terms)
        eval_outs = np.array(
            [np.mean(
                np.array(
                    [eval_out_raw[output_i] for output_i in range(n_outputs)
                     if poly_coefs[output_i][term_i] == 1]
                )
            )
                for term_i in range(n_terms)]
        )

    else:  # if classification

        # random_outs has shape (n_random, n_submodules, n_outputs)
        random_outs = np.array([[[class_acc for key, class_acc in rand_sm.items()
                                  if 'acc' in key and 'overall' not in key]
                                 for rand_sm in rand_results]
                                for rand_results in all_random_results])
        # true_outs has shape (n_submodules, n_outputs)
        true_outs = np.array([[class_acc for key, class_acc in true_sm.items()
                               if 'acc' in key and 'overall' not in key]
                              for true_sm in true_results])
        # eval_outs has shape (n_outputs,)
        eval_outs = np.array([class_acc for key, class_acc in evaluation.items()
                              if 'acc' in key and 'overall' not in key])

    random_means = np.nanmean(random_outs, axis=-1)
    true_means = np.nanmean(true_outs, axis=-1)

    random_changes = random_outs - eval_outs
    random_normalized_changes = random_changes / np.nanmean(random_changes, axis=-1)[:, :, np.newaxis]
    random_ranges_normalized_changes = np.ptp(random_normalized_changes, axis=-1)
    random_changes_means = np.nanmean(random_changes, axis=-1)
    true_changes = true_outs - eval_outs
    true_normalized_changes = true_changes / np.nanmean(true_changes, axis=-1)[:, np.newaxis]
    true_ranges_normalized_changes = np.ptp(true_normalized_changes, axis=-1)
    true_changes_means = np.nanmean(true_changes, axis=-1)

    mean_percentiles = np.array([compute_pvalue(true_means[sm_i], random_means[:, sm_i])
                                 for sm_i in range(n_submodules)])
    range_percentiles = np.array([compute_pvalue(true_ranges_normalized_changes[sm_i],
                                                 random_ranges_normalized_changes[:, sm_i], side='right')
                                  for sm_i in range(n_submodules)])

    # get effect sizes
    all_ef_means = []
    for sm_i in range(n_submodules):
        if true_changes_means[sm_i] < 0 and np.nanmean(random_changes_means[:, sm_i]) < 0:
            all_ef_means.append(true_changes_means[sm_i] / (np.nanmean(random_changes_means[:, sm_i]) +
                                                            true_changes_means[sm_i]))
        else:
            all_ef_means.append(0.0)
    effect_factors_means = np.nan_to_num(np.array(all_ef_means))
    effect_factors_ranges = np.nan_to_num(np.array([true_ranges_normalized_changes[sm_i] /
                                             (np.nanmean(random_ranges_normalized_changes[:, sm_i]) +
                                              true_ranges_normalized_changes[sm_i])
                                             for sm_i in range(n_submodules)]))

    chi2_p_means = chi2_categorical_test(mean_percentiles, n_shuffles)
    chi2_p_ranges = chi2_categorical_test(range_percentiles, n_shuffles)
    fisher_p_means = combine_ps(mean_percentiles, n_shuffles)
    fisher_p_ranges = combine_ps(range_percentiles, n_shuffles)
    fisher_stat_means = fisher_stat(mean_percentiles, n_shuffles)
    fisher_stat_ranges = fisher_stat(range_percentiles, n_shuffles)

    results = {'mean_percentiles': mean_percentiles, 'range_percentiles': range_percentiles,
               'effect_factors_means': effect_factors_means, 'effect_factors_range': effect_factors_ranges,
               'chi2_p_means': chi2_p_means, 'chi2_p_ranges': chi2_p_ranges,
               'fisher_p_means': fisher_p_means, 'fisher_p_ranges': fisher_p_ranges,
               'fisher_stat_means': fisher_stat_means, 'fisher_stat_ranges': fisher_stat_ranges}

    return results


# ImageNet


def _get_classification_accs_imagenet(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    class_accs = np.array([acc for acc in (np.diag(cm) / cm.sum(axis=1))])

    return class_accs


def _reverse_connections(conns):
    """Reverse the connections dictionary, i.e. get the incoming weights of a conv layer neurons."""
    rev_conns = defaultdict(list)

    for conv_i, conv_connection in enumerate(conns):
        for next_conv in conv_connection:
            rev_conns[next_conv['to_idx']].append({'from_idx': conv_i,
                                                   'weights': next_conv['weights']})

    return rev_conns


def lesion_test_imagenet(model, dataset, y, labels_in_layers,
                         num_clusters, steps, batch_size, val_dataset_object,
                         preprocess, min_size, max_prop,
                         connections, num_samples,
                         shuffle=False, num_classes=1000, only_last=False):
    """
    Returns a dict with string keys of form 'cluster_i_layer_j' where i is the cluster number and j is the
    layer index w.r.t. conv layers in the network. Each value is a dict returned from _classification_metrics()
    """
    acc_results_list = []
    class_prop_list = []
    cluster_sizes_dict = {}
    agg_weight_results_list = []
    
    rev_conns = _reverse_connections(connections)

    for sample_i in range(num_samples):

        print(f'sample_i={sample_i}')

        acc_results = {}
        class_props = {}
        cluster_sizes = {}
        agg_weight_results = {}
        
        # Iterate over all (cluster_i, layer_i)
        # Some combination might be irrelevant (e.g., no cluster_i in layer_i)
        for cluster_i in range(num_clusters):

            conv_i = -1  # counter var for iterating through conv layers
            for layer_i, layer in enumerate(model.layers):

                layer_str = str(type(layer)).lower()
                if '.conv2d' in layer_str:

                    if conv_i == -1:  # skip the first layer which is not included in graph
                        conv_i += 1  # increment
                        continue

                    lesion_idxs = np.array(labels_in_layers[conv_i]) == cluster_i
                    
                    if np.sum(lesion_idxs) < min_size or np.sum(lesion_idxs) > (max_prop *
                                                                                len(labels_in_layers[conv_i])):
                        conv_i += 1  # increment
                        continue  # if this sub module has too few or too many units

                    # Note: that we recourc the layer not based on the actual indexing (conv_i+1)
                    layer_key = f'layer_{conv_i}'
                    cluster_key = f'cluster_{cluster_i}'
                    conv_i += 1  # increment

                    if only_last and conv_i != len(labels_in_layers) - 1:  # only do the last layer
                        continue

                    if len(model.layers[layer_i].get_weights()) > 1:  # get weights and biases
                        layer_weights, layer_biases = model.layers[layer_i].get_weights()
                    else:  # load weights
                        layer_weights = model.layers[layer_i].get_weights()[0]
                        layer_biases = None
                    restore_weights = copy.deepcopy(layer_weights)

                    if shuffle:
                        lesion_idxs = np.random.permutation(lesion_idxs)
                        
                    # AGGREGATE WEIGHTS
                    # note that conv_i was incremented already
                    # for debuging:
                    # print(len(lesion_idxs),
                    #       [(c['to_idx'], c['weights'].shape) for c in connections[conv_i]],
                    #       [(c['from_idx'], c['weights'].shape) for c in rev_conns[conv_i]])
                    assert all([c['weights'].shape[0] == len(lesion_idxs)
                               for c in connections[conv_i]])
                    assert all([c['weights'].shape[1] == len(lesion_idxs)
                               for c in rev_conns[conv_i]])                    
                    agg_weight = {'outbound': np.mean([c['weights'][lesion_idxs, :].mean()
                                                       for c in connections[conv_i]]),
                                  'inbound': np.mean([c['weights'][:, lesion_idxs].mean()
                                                      for c in rev_conns[conv_i]])}
                    
                    # DO LESION
                    layer_weights[:, :, lesion_idxs, :] = 0
                    # if layer_biases is not None:
                    #     layer_biases[lesion_idxs] = 0

                    if layer_biases is not None:
                        model.layers[layer_i].set_weights((layer_weights, layer_biases))
                    else:
                        model.layers[layer_i].set_weights((layer_weights,))

                    cluster_pred = np.argmax(model.predict(dataset, steps=steps,
                                                           batch_size=batch_size), axis=-1)
                    if not isinstance(dataset, np.ndarray):
                        dataset = imagenet_generator(val_dataset_object, preprocess)
                    cluster_accs = _get_classification_accs_imagenet(y, cluster_pred)
                    cluster_props = np.concatenate((np.bincount(cluster_pred),
                                                    np.zeros(num_classes - np.max(cluster_pred) - 1))) \
                                    / len(cluster_pred)

                    if layer_biases is not None:
                        model.layers[layer_i].set_weights((restore_weights, layer_biases))
                    else:
                        model.layers[layer_i].set_weights((restore_weights,))

                    if layer_key not in acc_results:
                        acc_results[layer_key] = {}
                        cluster_sizes[layer_key] = {}
                        class_props[layer_key] = {}
                        agg_weight_results[layer_key] = {}

                    acc_results[layer_key][cluster_key] = cluster_accs
                    cluster_sizes[layer_key][cluster_key] = np.sum(lesion_idxs) / len(lesion_idxs)
                    class_props[layer_key][cluster_key] = cluster_props
                    agg_weight_results[layer_key][cluster_key] = agg_weight

        acc_results_list.append(acc_results)
        cluster_sizes_dict = cluster_sizes
        class_prop_list.append(class_props)
        agg_weight_results_list.append(agg_weight_results)

    return acc_results_list, class_prop_list, cluster_sizes_dict, agg_weight_results_list


def perform_lesion_experiment_imagenet(network, use_activations, local=False, num_clusters=32,
                                       num_shuffles=10, with_random=True, downsampled=False,
                                       eigen_solver='arpack', batch_size=32, min_size=MIN_SIZE,
                                       max_prop=MAX_PROP, downsampled_n_samples=8000,
                                       data_dir='/project/nn_clustering/datasets/imagenet2012',
                                       val_tar='ILSVRC2012_img_val.tar'):
    
    assert network != 'inceptionv3', 'This function does not yet support inceptionv3'

    net, preprocess = Classifiers.get(network)  # get network object and preprocess fn
    model = net((224, 224, 3), weights='imagenet')  # get network tf.keras.model

    data_path = Path(data_dir)
    tfrecords = list(data_path.glob('*validation.tfrecord*'))
    if not tfrecords:
        prep_imagenet_validation_data(data_dir, val_tar)  # this'll take a sec
    imagenet = tfds.image.Imagenet2012()  # dataset builder object
    imagenet._data_dir = data_dir
    val_dataset_object = imagenet.as_dataset(split='validation')  # datast object
    # assert isinstance(val_dataset_object, tf.data.Dataset)

    if downsampled:
        # get the small dataset as an np.ndarray
        dataset, y = imagenet_downsampled_dataset(val_dataset_object, preprocess,
                                                  n_images=downsampled_n_samples)
        steps = None
        val_set_size = downsampled_n_samples

    else:
        dataset = imagenet_generator(val_dataset_object, preprocess)
        val_set_size = 50000
        steps = val_set_size // 250  # use batch_size of 250
        y = []  # to become an ndarray of true labels
        for _ in range(steps):
            _, logits = next(dataset)
            y.append(np.argmax(logits, axis=-1))
        y = np.concatenate(y)
        batch_size = None

    # get info from clustering
    clustering_results = run_clustering_imagenet(network, num_clusters=num_clusters, local=local,
                                                 with_shuffle=False, eigen_solver=eigen_solver)
    if use_activations:
        with open(data_dir + f'/{network}_activations_local={local}_k={num_clusters}.pkl', 'rb') as f:
            labels = pickle.load(f)
    else:
        labels = clustering_results['labels']
    connections = clustering_results['conv_connections']  # just connections for conv layers
    layer_widths = [cc[0]['weights'].shape[0] for cc in connections[1:]]  # skip first conv layer
    dense_sizes = get_dense_sizes(connections)
    layer_widths.extend(list(dense_sizes.values()))
    labels_in_layers = list(splitter(labels, layer_widths))    

    y_pred = np.argmax(model.predict(dataset, steps=steps, batch_size=batch_size), axis=-1)
    if not isinstance(dataset, np.ndarray):
        dataset = imagenet_generator(val_dataset_object, preprocess)
    evaluation = _get_classification_accs_imagenet(y, y_pred)  # an ndarray of all 1000 class accs

    # next get true accs and label bincounts for the 1000 classes
    accs_true, class_props_true, cluster_sizes, agg_weight_true = lesion_test_imagenet(model, dataset, y,
                                                                      labels_in_layers, num_clusters,
                                                                      steps, batch_size, val_dataset_object,
                                                                      preprocess, min_size, max_prop,
                                                                      connections, num_samples=1)
    accs_true = accs_true[0]  # it's a 1 element list, so just take the first
    class_props_true = class_props_true[0]  # same as line above
    agg_weight_true = agg_weight_true[0]  # same as line above

    if not with_random:

        # make and return a dict with a keys giving sub modules and values giving
        # num shuffles, overall acc, and class accs

        results = {}
        for layer_key in accs_true.keys():
            results[layer_key] = {}
            for cluster_key in accs_true[layer_key].keys():
                sm_results = {}
                true_accs = accs_true[layer_key][cluster_key]
                sm_results['num_shuffles'] = num_shuffles
                sm_results['overall_acc'] = np.mean(true_accs)
                sm_results['class_accs'] = true_accs
                sm_results['agg_weights'] = agg_weight_true[layer_key][cluster_key]
                results[layer_key][cluster_key] = sm_results

        return evaluation, results

    else:  # perform random lesion tests num_shuffles times

        # get random results
        all_acc_random, all_class_props, _, all_agg_weight = lesion_test_imagenet(model, dataset, y, labels_in_layers,
                                                                                  num_clusters, steps, batch_size,
                                                                                  val_dataset_object, preprocess,
                                                                                  min_size, max_prop, connections,
                                                                                  num_shuffles, shuffle=True)

        # now make and return a dict with a keys giving sub modules and values giving
        # stats about true labels, shufflings, and p values for hypothesis tests

        results = {}
        for layer_key in accs_true.keys():
            results[layer_key] = {}
            for cluster_key in accs_true[layer_key].keys():

                sm_results = {}

                true_accs = accs_true[layer_key][cluster_key]
                random_accs = np.vstack([all_acc_random[i][layer_key][cluster_key] for i in range(num_shuffles)])
                overall_acc = np.mean(true_accs)
                overall_random_accs = np.mean(random_accs, axis=1)
                overall_acc_percentile = compute_pvalue(overall_acc, overall_random_accs)
                true_changes = true_accs - evaluation
                mean_random_changes = np.mean(random_accs - evaluation, axis=0)
                # overall_acc_effect_factor = np.mean(overall_acc) / (np.mean(overall_acc) + overall_random_accs)
                if np.mean(true_changes) < 0 and np.mean(mean_random_changes) < 0:
                    overall_acc_effect_factor = np.mean(true_changes) / (np.mean(true_changes) +
                                                                         np.mean(mean_random_changes))
                else:
                    overall_acc_effect_factor = 0

                random_changes = random_accs - evaluation
                normalized_random_changes = (random_changes.T / np.mean(random_changes, axis=-1)).T
                random_range_normalized_changes = np.ptp(normalized_random_changes, axis=-1)
                true_changes = true_accs - evaluation
                normalized_true_changes = true_changes / np.mean(true_changes)
                true_range_normalized_changes = np.ptp(normalized_true_changes)
                range_percentile = compute_pvalue(true_range_normalized_changes,
                                                  random_range_normalized_changes,
                                                  side='right')
                range_effect_factor = np.mean(true_range_normalized_changes) / (np.mean(true_range_normalized_changes) +
                                                                                random_range_normalized_changes)

                sm_results['cluster_size'] = cluster_sizes[layer_key][cluster_key]
                sm_results['acc'] = overall_acc
                sm_results['acc_percentile'] = overall_acc_percentile
                sm_results['random_accs'] = random_accs
                sm_results['overall_acc_effect_factor'] = overall_acc_effect_factor
                sm_results['range'] = true_range_normalized_changes
                sm_results['range_percentile'] = range_percentile
                sm_results['range_effect_factor'] = range_effect_factor
                
                sm_results['agg_weights'] = agg_weight_true[layer_key][cluster_key]
                sm_results['random_agg_weights'] = [all_agg_weight[i][layer_key][cluster_key]
                                                    for i in range(num_shuffles)]
                
                results[layer_key][cluster_key] = sm_results

        return evaluation, results


def do_lesion_hypo_tests_imagenet(results, n_shuffles):

    acc_percentiles = []
    range_percentiles = []
    acc_effects = []
    range_effects = []

    for layer_key in results.keys():
        for cluster_key in results[layer_key].keys():
            sm_results = results[layer_key][cluster_key]
            range_percentiles.append(sm_results['range_percentile'])
            acc_percentiles.append(sm_results['acc_percentile'])
            acc_effects.append(sm_results['overall_acc_effect_factor'])
            range_effects.append(sm_results['range_effect_factor'])

    fisher_p_means = combine_ps(np.array(acc_percentiles), n_shuffles)
    chi2_p_means = chi2_categorical_test(np.array(acc_percentiles), n_shuffles)

    fisher_p_ranges = combine_ps(np.array(range_percentiles), n_shuffles)
    chi2_p_ranges = chi2_categorical_test(np.array(range_percentiles), n_shuffles)

    fisher_stat_means = fisher_stat(np.array(acc_percentiles), n_shuffles)
    fisher_stat_ranges = fisher_stat(np.array(range_percentiles), n_shuffles)

    acc_means_effects = np.nan_to_num(np.array(acc_effects))
    range_means_effects = np.nan_to_num(np.array(range_effects))

    results = {'chi2_p_means': chi2_p_means, 'chi2_p_ranges': chi2_p_ranges,
               'fisher_p_means': fisher_p_means, 'fisher_p_ranges': fisher_p_ranges,
               'effect_factors_means': acc_means_effects, 'effect_factors_range': range_means_effects,
               'fisher_stat_means': fisher_stat_means, 'fisher_stat_ranges': fisher_stat_ranges}

    return results


# because of the circular import between agg_weight and this module 
from src.lesion.agg_weight import agg_weights_by_partitioning, agg_weights_by_subcluster, conv2_agg
