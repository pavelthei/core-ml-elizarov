import numpy as np
import scipy.sparse as sp
import torch.nn as nn
import torch
import torch.nn.functional as F
from tqdm import tqdm


# METRIC
def ndcg_at_k(predictions: np.array,
              true_matrix: sp.csr_matrix,
              k: int,
              batch_size: int = 100) -> float:
    """
    Function to return NDCG@K

    :param predictions: Matrix (NxM) with predictions with N users and M predicted values
    :param true_matrix: Matrix with interactions for these users
    :param k: The size of range to calculate this metric
    :param batch_size: Size of batch for predicting DCG@k and IDCG@k separately. Made for the memory economy
    :return: NDCG@K for the predicted values
    """

    # Make array to save NDCG@k for each of the users
    ndcg_all = np.array([])

    # Extracting all indices for the matrix
    all_indices = np.arange(0, true_matrix.shape[0])

    # Iterating over batches
    for start_ind in tqdm(range(0, true_matrix.shape[0], batch_size)):
        batch_indices = all_indices[start_ind: start_ind + batch_size]

        # Selecting batch for true values
        true_batch = true_matrix[batch_indices, :].A

        # Take the gain for each of the predictions
        gain = 2 ** np.take_along_axis(true_batch, predictions[batch_indices, :k], axis=1) - 1

        # Discount part
        discount = np.log2(np.arange(2, k + 2))

        # Calculating DCG@K
        dcg = np.sum(gain / discount, axis=1)

        # IDCG@K part
        ideal_gain = 2 ** np.sort(true_batch, axis=1)[:, ::-1][:, :k] - 1
        idcg = np.sum(ideal_gain / discount, axis=1)

        # Calculating NDCG@k and adding it to other users
        ndcg = dcg / idcg
        ndcg_all = np.concatenate((ndcg_all, ndcg))

    return ndcg_all


# PYTORCH TOOLS
class MaskedMse(torch.nn.Module):

    @staticmethod
    def forward(inputs, targets):
        """
        :param inputs: Predicted values
        :param targets: True values

        :return: MSE Loss of non-zero elements
        """
        if targets.is_sparse:
            targets = targets.to_dense()
        mask = targets != 0
        criterion = torch.nn.MSELoss(reduction='sum')
        return criterion(inputs * mask.float(), targets)


def torch_activation(x: torch.Tensor, kind: str):
    if kind == 'relu':
        return F.relu(x)
    elif kind == 'relu6':
        return F.relu6(x)
    elif kind == 'elu':
        return F.elu(x)
    elif kind == 'lrelu':
        return F.leaky_relu(x)
    elif kind == 'selu':
        return F.selu(x)
    elif kind == 'tanh':
        return F.tanh(x)
    elif kind == 'sigmoid':
        return F.sigmoid(x)
    elif kind == 'swish':
        return x * F.sigmoid(x)
    elif kind == 'identity':
        return x
    else:
        raise ValueError(f"{kind} activation function is unrecognized")


def torch_loss(kind: str):
    if kind == 'mse':
        return nn.MSELoss
    elif kind == 'bce':
        return nn.BCELoss
    elif kind == 'nll':
        return nn.NLLLoss
    elif kind == 'masked_mse':
        return MaskedMse
    else:
        raise ValueError(f"{kind} loss function is unrecognized")


def torch_optimizer(kind: str):
    if kind == 'adam':
        return torch.optim.Adam
    elif kind == 'sgd':
        return torch.optim.SGD
    elif kind == 'adamax':
        return torch.optim.Adamax
    else:
        raise ValueError(f"{kind} optimizer is unrecognized")


if __name__ == "__main__":
    pass
