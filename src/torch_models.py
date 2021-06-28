import scipy.sparse as sp
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from src.utils import torch_loss, torch_optimizer, torch_activation, ndcg_at_k


class TorchALS(nn.Module):
    def __init__(self,
                 n_factors: int = 100,
                 regularization: float = 0.01,
                 n_iterations: int = 15,
                 device: str = 'cpu',
                 batch_size: int = 256):
        """
        Init model

        :param n_factors: Number of latent factors
        :param regularization: Regularization term
        :param n_iterations: Number of iterations to control convergence
        :param device: select device to make calculations
        :param batch_size: size of batch for memory saving
        """

        # Creating attributes
        self.n_factors = n_factors
        self.regularization = regularization
        self.n_iterations = n_iterations
        self.device = device
        self.batch_size = batch_size

        # Set-up users and items latent vectors
        self.u = None
        self.v = None

    def _update_calculation(self,
                            rating: torch.Tensor,
                            factors_b: torch.Tensor,
                            regularization: float = 0.01) -> torch.Tensor:
        """
        Method to make ALS step: calculates updating

        :param rating: Matrix with ratings
        :param factors_b: Factors that used in formula in updating of factor a
        :param regularization: Regularization term
        :return: Updated factor a
        """

        # B.T @ B + lamnda * I
        A = torch.mm(torch.transpose(factors_b, 0, 1), factors_b) + \
            regularization * torch.eye(factors_b.shape[1], factors_b.shape[1]).to(self.device)

        # R @ B
        B = torch.mm(rating, factors_b)

        # R @ B @ (B.T @ B + lambda * I) ^ (-1)
        return torch.mm(B, torch.inverse(A))

    def fit(self,
            train_matrix: sp.csr_matrix) -> None:
        """
        Method to train model

        :param train_matrix: Sparse matrix to train model
        """

        # Creating users and items latent vectors
        self.u = torch.rand((train_matrix.shape[0], self.n_factors))
        nn.init.uniform_(self.u, 0, 1 / np.sqrt(self.n_factors))
        self.u = self.u.to(self.device)

        self.v = torch.rand((train_matrix.shape[1], self.n_factors))
        nn.init.uniform_(self.v, 0, 1 / np.sqrt(self.n_factors))
        self.v = self.v.to(self.device)

        # Extracting all indices for the matrix
        all_users = np.arange(0, train_matrix.shape[0])
        all_items = np.arange(0, train_matrix.shape[1])

        # Iterating over number of iterations
        for epoch in tqdm(range(self.n_iterations)):

            new_u = torch.Tensor([]).to(self.device)
            new_v = torch.Tensor([]).to(self.device)

            # Iterating over users
            for start_ind in range(0, train_matrix.shape[0], self.batch_size):
                batch_indices = all_users[start_ind: start_ind + self.batch_size]

                # Selecting rating
                r = torch.Tensor(train_matrix[batch_indices, :].A).to(self.device)

                # This trick should allow not to take into account 0 values
                r[r == 0] = (self.u[batch_indices, :] @ self.v.T)[r == 0]

                new_u = torch.cat(
                    (new_u, self._update_calculation(r, self.v, regularization=self.regularization).to(self.device)))

            # Iterating over items
            for start_ind in range(0, train_matrix.shape[1], self.batch_size):
                batch_indices = all_items[start_ind: start_ind + self.batch_size]

                # Selecting rating
                r = torch.Tensor(train_matrix[:, batch_indices].A.T).to(self.device)

                # This trick should allow not to take into account 0 values
                r[r == 0] = (self.v[batch_indices, :] @ self.u.T)[r == 0]

                new_v = torch.cat((new_v, self._update_calculation(r, self.u, regularization=self.regularization)))

            self.u = new_u
            self.v = new_v

    def predict(self,
                train_matrix: sp.csr_matrix,
                test_matrix: sp.csr_matrix,
                number_of_predictions: int = 100,
                excluding_predictions=None,
                drop_cold_users: bool = True,
                batch_size: int = 1000,
                user_ids=None) -> np.array:

        """
        Method for generating predictions

        :param train_matrix: The source of the recommendations
        :param test_matrix: Test matrix with users interactions.
        :param number_of_predictions: Number of recommendations for each user
        :param excluding_predictions: if not None, then sparse matrix with values, than should no be included in
               predictions
        :param drop_cold_users: if True, then all non active users will be dropped
        :param batch_size: Batch size of users to predict iteratively, to saving memory
        :param user_ids: ids of users
        :return: Matrix with the recommendations for each of the test users
        """

        # Extracting all indices for the matrix
        if user_ids is None:
            all_indices = np.arange(0, test_matrix.shape[0])
        else:
            all_indices = user_ids

        # Variable to save all predictions
        predictions_all = []

        # Iterating over batches
        for start_ind in range(0, len(all_indices), batch_size):
            batch_indices = all_indices[start_ind: start_ind + batch_size]

            # Predicting scores
            scores = torch.mm(self.u[batch_indices], self.v.T).cpu().numpy()

            # Excluding the predictions if needed
            if excluding_predictions is not None:
                scores[excluding_predictions[batch_indices, :].nonzero()] = -np.inf

            # Make recommendations
            predictions = np.argsort(scores, axis=1)[:, ::-1][:, :number_of_predictions]

            # Appending results
            predictions_all.extend(predictions)

        predictions_all = np.array(predictions_all)

        # Deleting cold users
        if drop_cold_users:
            predictions_all = predictions_all[test_matrix.getnnz(axis=1) > 0]

        return predictions_all


class AutoEncoderRecommender(nn.Module):
    """
    Training Deep AutoEncoders for Collaborative Filtering: https://arxiv.org/pdf/1708.01715.pdf
    """

    def __init__(self,
                 n_items: int,
                 layers_dims: list,
                 batch_size: int = 128,
                 n_epoch: int = 10,
                 lr: float = 0.001,
                 activation: str = 'relu',
                 last_activation: str = 'identity',
                 dropout: float = 0.5,
                 loss: str = 'masked_mse',
                 is_constrained: bool = False,
                 optimizer: str = 'adam',
                 device: str = 'cpu',
                 augmentation_step: int = 1):
        """
        Init model

        :param n_items: Number of items to work with
        :param layers_dims: List, which represents the number of layers and the number of neurons in each layer
        :param n_epoch: Number of epochs
        :param batch_size: Number of batches during training
        :param lr: Learning rate
        :param activation: Activation function type to catch non-linearity
        :param last_activation: Activation for the last layer
        :param dropout: Probability of dropout
        :param is_constrained: If True, then all weights of encoder are identical to decoder weights
        :param augmentation_step: In paper there is a augmentation step. This parameter sets the number of steps
        """
        super(AutoEncoderRecommender, self).__init__()

        # Init encoder
        all_dims = [n_items] + layers_dims
        self._last_layer = len(all_dims) - 2

        # Init encoder parameters
        self.encode_W = nn.ParameterList(
            [nn.Parameter(torch.rand(all_dims[i + 1], all_dims[i])) for i in range(len(all_dims) - 1)])
        for w in self.encode_W:
            nn.init.xavier_uniform_(w)
        self.encode_b = nn.ParameterList(
            [nn.Parameter(torch.zeros(all_dims[i + 1])) for i in range(len(all_dims) - 1)])

        # Init decoder params if model is not constrained
        if not is_constrained:
            rev_dims = list(reversed(all_dims))
            self.decode_W = nn.ParameterList(
                [nn.Parameter(torch.rand(rev_dims[i + 1], rev_dims[i])) for i in range(len(rev_dims) - 1)])
            for w in self.decode_W:
                nn.init.xavier_uniform_(w)
            self.decode_b = nn.ParameterList(
                [nn.Parameter(torch.zeros(rev_dims[i + 1])) for i in range(len(rev_dims) - 1)])

        # Init other attributes
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.lr = lr
        self.activation = activation
        self.last_activation = last_activation
        self.is_constrained = is_constrained
        self.dropout = dropout
        self.device = device
        self.aug_step = augmentation_step
        self.loss_class = torch_loss(kind=loss)
        self.loss_func = None
        self.optimizer_class = torch_optimizer(kind=optimizer)
        self.optimizer = None

        # Adding best params and best score
        self.best_params = None
        self.best_score = None

    def encode(self, x) -> torch.Tensor:
        """
        Method for encoding

        :param x: Input to encode
        :return: encoded values
        """
        for i, w in enumerate(self.encode_W):
            x = torch_activation(F.linear(input=x, weight=w, bias=self.encode_b[i]), kind=self.activation)
        # As in the paper, dropout only for the
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training, inplace=False)
        return x

    def decode(self, x) -> torch.Tensor:
        """
        Method for decoding

        :param x: Input to decode
        :return: decoded values
        """

        if self.is_constrained:
            for i, w, in enumerate(list(reversed(self.encode_W))):
                x = torch_activation(F.linear(input=x, weight=w.T, bias=self.encode_b[-(i + 1)]),
                                     kind=self.activation if i != self._last_layer else self.last_activation)

        else:
            for i, w in enumerate(self.decode_W):
                x = torch_activation(F.linear(input=x, weight=w, bias=self.decode_b[i]),
                                     kind=self.activation if i != self._last_layer else self.last_activation)

        return x

    def forward(self, x) -> torch.Tensor:
        """
        Forward method

        :param x: Input data to run through the net
        :return: results after net
        """
        return self.decode(self.encode(x))

    def fit(self,
            train_matrix: sp.csr_matrix,
            validation_matrix: sp.csr_matrix,
            val_user_ids=None,
            k_validation: int = 100) -> None:
        """
        Method to train models

        :param train_matrix: Matrix for training
        :param validation_matrix: Matrix for validating performance
        :param val_user_ids: ids of validation users
        :param k_validation: Number of predictions during validation
        """

        # Init optimizer and loss
        self.optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        self.loss_func = self.loss_class()

        # Removing zero rows
        train_matrix_nz = train_matrix[train_matrix.getnnz(1) > 0]

        # Iterating over epochs
        for epoch in range(self.n_epoch):
            all_ind = torch.randperm(train_matrix_nz.shape[0])

            # Switch model to the train mode
            self.train()

            # Iterating over batches
            with tqdm(range(0, train_matrix_nz.shape[0], self.batch_size)) as progress_epoch:
                for start_ind in progress_epoch:
                    # set description
                    progress_epoch.set_description(f"Epoch {epoch + 1}")

                    batch_ind = all_ind[start_ind: start_ind + self.batch_size]

                    # Pass small batches
                    if len(batch_ind) < self.batch_size:
                        continue

                    # Selecting batch
                    batch = torch.Tensor(train_matrix_nz[batch_ind].A).to(self.device)

                    # Make a optimizer step
                    predictions = self.forward(batch)
                    loss_value = self.loss_func(predictions, batch)
                    loss_value.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # Make  augmentation steps
                    if self.aug_step > 0:
                        for _ in range(self.aug_step):
                            batch = Variable(predictions.data)
                            predictions = self.forward(batch)
                            loss_value = self.loss_func(predictions, batch)
                            loss_value.backward()
                            self.optimizer.step()
                            self.optimizer.zero_grad()

                    # Set info
                    progress_epoch.set_postfix_str(f"Train Loss: {np.round(loss_value.item(), 5)}")

            # For the validating part set model to evaluation mode
            self.eval()

            # Calculating ndcg@k for validation
            predictions = self.predict(train_matrix, validation_matrix, number_of_predictions=k_validation,
                                       excluding_predictions=train_matrix, user_ids=val_user_ids)

            ndcg = ndcg_at_k(predictions, validation_matrix[validation_matrix.getnnz(1) > 0], k=k_validation).mean()
            print(f"Validation NDCG: {ndcg}")

            # Save best parameters
            if self.best_score is None or ndcg > self.best_score:
                self.best_score = ndcg
                self.best_params = self.state_dict()

    def predict(self,
                train_matrix: sp.csr_matrix,
                test_matrix: sp.csr_matrix,
                number_of_predictions: int = 100,
                excluding_predictions=None,
                drop_cold_users: bool = True,
                batch_size: int = 1000,
                user_ids=None) -> np.array:

        """
        Method for generating predictions

        :param train_matrix: The source of the recommendations
        :param test_matrix: Test matrix with users interactions.
        :param number_of_predictions: Number of recommendations for each user
        :param excluding_predictions: if not None, then sparse matrix with values, than should no be included in
               predictions
        :param drop_cold_users: if True, then all non active users will be dropped
        :param batch_size: Batch size of users to predict iteratively, to saving memory
        :param user_ids: ids of users
        :return: Matrix with the recommendations for each of the test users
        """

        # Extracting all indices for the matrix
        if user_ids is None:
            all_indices = np.arange(0, test_matrix.shape[0])
        else:
            all_indices = user_ids

        # Variable to save all predictions
        predictions_all = []

        # Iterating over batches
        for start_ind in range(0, len(all_indices), batch_size):
            batch_indices = all_indices[start_ind: start_ind + batch_size]

            # Predicting scores
            test_temp = torch.Tensor(train_matrix[batch_indices, :].A).to(self.device)
            with torch.set_grad_enabled(False):
                scores = self.forward(test_temp).cpu().numpy()

            # Excluding the predictions if needed
            if excluding_predictions is not None:
                scores[excluding_predictions[batch_indices, :].nonzero()] = -np.inf

            # Make recommendations
            predictions = np.argsort(scores, axis=1)[:, ::-1][:, :number_of_predictions]

            # Appending results
            predictions_all.extend(predictions)

        predictions_all = np.array(predictions_all)

        # Deleting cold users
        if drop_cold_users:
            predictions_all = predictions_all[test_matrix.getnnz(axis=1) > 0]

        return predictions_all


class DSSM(nn.Module):
    def __init__(self,
                 user_dim: int,
                 item_dim: int,
                 layers_dims: list,
                 user_embedding: bool = True,
                 n_users: int = 0,
                 n_epoch: int = 10,
                 batch_size: int = 128,
                 lr: float = 0.001,
                 activation: str = 'relu',
                 last_activation: str = 'identity',
                 dropout: float = 0.5,
                 loss: str = 'masked_mse',
                 is_constrained: bool = False,
                 optimizer: str = 'adam',
                 device: str = 'cpu'):

        """

        :param user_dim: Number of user features
        :param item_dim: Number of items features
        :param user_embedding: If True, then instead of users features, users embeddings will be used
        :param n_users: Number of users.
        :param layers_dims: List, which represents the number of layers and the number of neurons in each layer
        :param n_epoch: Number of epochs
        :param batch_size: Number of batches during training
        :param lr: Learning rate
        :param activation: Activation function type to catch non-linearity
        :param last_activation: Activation for the last layer
        :param dropout: Probability of dropout
        """
        super(DSSM, self).__init__()
        # Check the correctness of number of users
        if user_embedding and n_users <= 0:
            raise ValueError(f"Number of users should be more then {n_users}")

        # Set the device
        self.device = device

        # Creating embedding if required
        if user_embedding:
            self.users = nn.Embedding(n_users, user_dim).to(device)
            nn.init.xavier_uniform_(self.users.weight.data)

        # Creating net for users
        all_dims_user = [user_dim] + layers_dims
        self.user_net_W = nn.ParameterList(
            [nn.Parameter(torch.rand(all_dims_user[i + 1], all_dims_user[i])) for i in range(len(all_dims_user) - 1)])
        for w in self.user_net_W:
            nn.init.xavier_uniform_(w)
        self.user_net_b = nn.ParameterList(
            [nn.Parameter(torch.rand(all_dims_user[i + 1])) for i in range(len(all_dims_user) - 1)])

        # Creating net for items
        all_dims_item = [item_dim] + layers_dims
        self.item_net_W = nn.ParameterList(
            [nn.Parameter(torch.rand(all_dims_item[i + 1], all_dims_item[i])) for i in range(len(all_dims_item) - 1)])
        for w in self.user_net_W:
            nn.init.xavier_uniform_(w)
        self.item_net_b = nn.ParameterList(
            [nn.Parameter(torch.rand(all_dims_item[i + 1])) for i in range(len(all_dims_item) - 1)])

        # Init other attributes
        self.batch_size = batch_size
        self.user_emb = user_embedding
        self.n_epoch = n_epoch
        self.lr = lr
        self.activation = activation
        self.last_activation = last_activation
        self.is_constrained = is_constrained
        self.dropout = dropout
        self.device = device
        self.loss_class = torch_loss(kind=loss)
        self.loss_func = None
        self.optimizer_class = torch_optimizer(kind=optimizer)
        self.optimizer = None

        # Adding best params and best score
        self.best_params = None
        self.best_score = None

    def forward(self,
                item_features: torch.Tensor,
                users_features=None,
                users_ids=None) -> torch.Tensor:
        """
        Forward method

        :param item_features: Features of the items
        :param users_features: Features of the users
        :param users_ids: Ids for choosing embeddings
        :return: predicted scores
        """

        if users_features is None and users_ids is None:
            raise ValueError("Only one should be None")

        # Setting embedding
        if self.user_emb:
            users_features = self.users(users_ids)

        # Iterating over user net
        for ind, w in enumerate(self.user_net_W):
            users_features = F.linear(users_features, weight=w, bias=self.user_net_b[ind])
            if ind != len(self.user_net_W) - 1:
                users_features = torch_activation(users_features, kind=self.activation)
                users_features = F.dropout(users_features, training=self.training, p=self.dropout)
            else:
                users_features = torch_activation(users_features, kind=self.last_activation)

        # Iterating over item net
        for ind, w in enumerate(self.item_net_W):
            item_features = F.linear(item_features, weight=w, bias=self.item_net_b[ind])
            if ind != len(self.item_net_W) - 1:
                item_features = torch_activation(item_features, kind=self.activation)
                item_features = F.dropout(item_features, training=self.training, p=self.dropout)
            else:
                item_features = torch_activation(item_features, kind=self.last_activation)

        # Return matmul as a similarity
        return torch.mm(users_features, item_features.T)

    def fit(self,
            train_matrix: sp.csr_matrix,
            val_matrix: sp.csr_matrix,
            item_features: sp.csr_matrix,
            user_features_train=None,
            user_features_val=None,
            user_ind_train=None,
            user_ind_val=None,
            k_validation: int = 100) -> None:
        """
        Method for training model

        :param train_matrix: Sparse matrix to train model
        :param val_matrix:  Sparse matrix to validate model
        :param item_features: Sparse matrix with features for items for training (supposed to be full)
        :param user_features_train: Sparse matrix with features for users for training (supposed to be full)
        :param user_features_val: Sparse matrix with features for users for validation (supposed to be full)
        :param user_ind_train: User indices for training
        :param user_ind_val: User indices for validation
        :param k_validation: Number of predictions during validation
        """
        # Init optimizer and loss
        self.optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        self.loss_func = self.loss_class()

        # Iterating over epochs
        for epoch in range(self.n_epoch):
            self.train()
            all_ind = torch.randperm(train_matrix.shape[0])

            # Switch model to the train mode
            self.train()

            # Iterating over batches
            with tqdm(range(0, train_matrix.shape[0], self.batch_size)) as progress_epoch:
                for start_ind in progress_epoch:
                    # set description
                    progress_epoch.set_description(f"Epoch {epoch + 1}")

                    batch_ind = all_ind[start_ind: start_ind + self.batch_size]

                    # Pass small batches
                    if len(batch_ind) < self.batch_size:
                        continue

                    item_batch = torch.Tensor(item_features.A).to(self.device)

                    if self.user_emb:
                        user_features_batch = None
                        user_ids_batch = torch.LongTensor(user_ind_train)[batch_ind].to(self.device)

                    else:
                        user_features_batch = torch.Tensor(user_features_train[batch_ind, :].A).to(self.device)
                        user_ids_batch = None

                    predictions = self.forward(item_features=item_batch,
                                               users_features=user_features_batch,
                                               users_ids=user_ids_batch)

                    real_values = torch.Tensor(train_matrix[batch_ind, :].A).to(self.device)

                    loss_value = self.loss_func(predictions, real_values)
                    loss_value.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # Set info
                    progress_epoch.set_postfix_str(f"Train Loss: {np.round(loss_value.item(), 5)}")

            # For the validating part set model to evaluation mode
            self.eval()

            # Calculating ndcg@k for validation
            predictions = self.predict(item_features=item_features,
                                       users_features=user_features_val,
                                       excluding_predictions=train_matrix,
                                       number_of_predictions=k_validation,
                                       user_ids=user_ind_val)

            ndcg = ndcg_at_k(predictions, val_matrix[val_matrix.getnnz(1) > 0, :], k=k_validation).mean()
            print(f"Validation NDCG: {ndcg}")

            # Save best parameters
            if self.best_score is None or ndcg > self.best_score:
                self.best_score = ndcg
                self.best_params = self.state_dict()

    def predict(self,
                item_features: sp.csr_matrix,
                users_features=None,
                number_of_predictions: int = 100,
                excluding_predictions=None,
                drop_cold_users: bool = True,
                batch_size_users: int = 1000,
                batch_size_items: int = 1000,
                user_ids=None) -> np.array:
        """
        Method for generating predictions

        :param item_features: Matrix with features for items
        :param users_features: Features for users
        :param number_of_predictions: Number of recommendations for each user
        :param excluding_predictions: if not None, then sparse matrix with values, than should no be included in
               predictions
        :param drop_cold_users: if True, then all non active users will be dropped
        :param batch_size_users: Batch size of users to predict iteratively, to saving memory
        :param batch_size_items: Batch size of items to predict iteratively, to saving memory
        :param user_ids: ids of users
        :return: Matrix with the recommendations for each of the test users
        """
        # Extracting all indices for the matrix
        if user_ids is None:
            user_indices = np.arange(0, users_features.shape[0])
        else:
            user_indices = user_ids

        # As items matrix is supposed to be full, then indices is just a range
        items_indices = np.arange(0, item_features.shape[0])

        # Variable to save all predictions
        predictions_all = []

        # Iterating over batches of users
        for start_ind_user in range(0, len(user_indices), batch_size_users):
            batch_indices_user = user_indices[start_ind_user: start_ind_user + batch_size_users]
            predictions_for_users_batch = []

            # Iterating over batches of items
            for start_ind_item in range(0, len(items_indices), batch_size_users):
                batch_indices_item = items_indices[start_ind_item: start_ind_item + batch_size_items]

                # Items selection
                items_batch_features = torch.Tensor(item_features[batch_indices_item, :].A).to(self.device)

                if not self.user_emb:
                    # should be csr_matrix
                    users_batch_features = torch.Tensor(users_features[batch_indices_user, :].A).to(self.device)
                else:
                    users_batch_features = None

                with torch.set_grad_enabled(False):
                    scores_prom = self.forward(items_batch_features,
                                               users_batch_features,
                                               users_ids=torch.LongTensor(batch_indices_user). \
                                               to(self.device)).cpu().numpy()

                predictions_for_users_batch.append(scores_prom)

            scores = np.hstack(predictions_for_users_batch)

            # Excluding the predictions if needed
            if excluding_predictions is not None:
                scores[excluding_predictions[batch_indices_user, :].nonzero()] = -np.inf

            # Make recommendations
            predictions = np.argsort(scores, axis=1)[:, ::-1][:, :number_of_predictions]

            # Appending results
            predictions_all.extend(predictions)

        predictions_all = np.array(predictions_all)

        # # Deleting cold users
        # if drop_cold_users:
        #     predictions_all = predictions_all[test_matrix.getnnz(axis=1) > 0]

        return predictions_all
