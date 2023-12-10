import numpy as np
import torch

def accuracy(scores, labels):
    r"""
        Compute the per-class accuracies and the overall accuracy # TODO: complete doc
        Parameters
        ----------
        scores: torch.FloatTensor, shape (B?, C, N)
            raw scores for each class
        labels: torch.LongTensor, shape (B?, N)
            ground truth labels
        Returns
        -------
        list of floats of length num_classes+1 (last item is overall accuracy)
    """
    num_classes = scores.size(-2) # we use -2 instead of 1 to enable arbitrary batch dimensions

    predictions = torch.max(scores, dim=-2).indices

    accuracies = []

    accuracy_mask = predictions == labels
    for label in range(num_classes):
        label_mask = labels == label
        per_class_accuracy = (accuracy_mask & label_mask).float().sum()
        per_class_accuracy /= label_mask.float().sum()
        accuracies.append(per_class_accuracy.cpu().item())
    # overall accuracy
    accuracies.append(accuracy_mask.float().mean().cpu().item())
    return accuracies

def intersection_over_union(scores, labels):
    r"""
        Compute the per-class IoU and the mean IoU # TODO: complete doc
        Parameters
        ----------
        scores: torch.FloatTensor, shape (B?, C, N)
            raw scores for each class
        labels: torch.LongTensor, shape (B?, N)
            ground truth labels
        Returns
        -------
        list of floats of length num_classes+1 (last item is mIoU)
    """
    num_classes = scores.size(-2) # we use -2 instead of 1 to enable arbitrary batch dimensions

    predictions = torch.max(scores, dim=-2).indices

    ious = []
    unique_C, _ = np.unique(labels.cpu().numpy(), return_counts=True)
    for label in range(num_classes):
        if label in unique_C:
            pred_mask = predictions == label
            labels_mask = labels == label
            iou = (pred_mask & labels_mask).float().sum() / (pred_mask | labels_mask).float().sum()
            ious.append(iou.cpu().item())
        else:
            ious.append(np.nan)
    ious.append(np.nanmean(ious))
    return ious

def fast_confusion(true, pred, label_values=None):
    """
    Fast confusion matrix (100x faster than Scikit learn). But only works if labels are la
    :param true:
    :param false:
    :param num_classes:
    :return:
    """

    # Ensure data is in the right format
    true = np.squeeze(true)
    pred = np.squeeze(pred)
    if len(true.shape) != 1:
        raise ValueError('Truth values are stored in a {:d}D array instead of 1D array'. format(len(true.shape)))
    if len(pred.shape) != 1:
        raise ValueError('Prediction values are stored in a {:d}D array instead of 1D array'. format(len(pred.shape)))
    if true.dtype not in [np.int32, np.int64]:
        raise ValueError('Truth values are {:s} instead of int32 or int64'.format(true.dtype))
    if pred.dtype not in [np.int32, np.int64]:
        raise ValueError('Prediction values are {:s} instead of int32 or int64'.format(pred.dtype))
    true = true.astype(np.int32)
    pred = pred.astype(np.int32)

    # Get the label values
    if label_values is None:
        # From data if they are not given
        label_values = np.unique(np.hstack((true, pred)))
    else:
        # Ensure they are good if given
        if label_values.dtype not in [np.int32, np.int64]:
            raise ValueError('label values are {:s} instead of int32 or int64'.format(label_values.dtype))
        if len(np.unique(label_values)) < len(label_values):
            raise ValueError('Given labels are not unique')

    # Sort labels
    label_values = np.sort(label_values)

    # Get the number of classes
    num_classes = len(label_values)

    #print(num_classes)
    #print(label_values)
    #print(np.max(true))
    #print(np.max(pred))
    #print(np.max(true * num_classes + pred))
    # Start confusion computations
    if label_values[0] == 0 and label_values[-1] == num_classes - 1:

        # Vectorized confusion
        vec_conf = np.bincount(true * num_classes + pred)

        # Add possible missing values due to classes not being in pred or true
        #print(vec_conf.shape)
        if vec_conf.shape[0] < num_classes ** 2:
            vec_conf = np.pad(vec_conf, (0, num_classes ** 2 - vec_conf.shape[0]), 'constant')
        #print(vec_conf.shape)

        # Reshape confusion in a matrix
        return vec_conf.reshape((num_classes, num_classes))


    else:

        # Ensure no negative classes
        if label_values[0] < 0:
            raise ValueError('Unsupported negative classes')

        # Get the data in [0,num_classes[
        label_map = np.zeros((label_values[-1] + 1,), dtype=np.int32)
        for k, v in enumerate(label_values):
            label_map[v] = k

        pred = label_map[pred]
        true = label_map[true]

        # Vectorized confusion
        vec_conf = np.bincount(true * num_classes + pred)

        # Add possible missing values due to classes not being in pred or true
        if vec_conf.shape[0] < num_classes ** 2:
            vec_conf = np.pad(vec_conf, (0, num_classes ** 2 - vec_conf.shape[0]), 'constant')

        # Reshape confusion in a matrix
        return vec_conf.reshape((num_classes, num_classes))
