import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

from sklearn.decomposition import TruncatedSVD
from src.utils import ndcg_at_k


class MostPopular:
    def __init__(self,
                 popularity_type: str = "by_rating"):
        """
        Init model

        :param popularity_type: selects the type of the popularity: if "by_count",
        """

        assert popularity_type in ("by_rating", "by_count")
        self.popularity_type = popularity_type
        self.popularity = None

    def fit(self, train_matrix: sp.csr_matrix) -> None:
        """
        Method to fit the model

        :param train_matrix: Sparse matrix for selecting the most popular items
        """

        # Selecting the type of popularity
        if self.popularity_type == 'by_rating':
            self.popularity = np.array(train_matrix.mean(axis=0)).squeeze()
        else:
            self.popularity = np.array([train_matrix[:, i].count_nonzero() for i in range(train_matrix.shape[1])])

    def predict(self,
                train_matrix: sp.csr_matrix,
                test_matrix: sp.csr_matrix,
                number_of_predictions: int = 100,
                excluding_predictions=None,
                drop_cold_users: bool = True,
                batch_size=1000) -> np.array:
        """
        Method for generating predictions

        :param train_matrix: The source of the recommendations
        :param test_matrix: Test matrix with users interactions.
        :param number_of_predictions: Number of recommendations for each user
        :param excluding_predictions: if not None, then sparse matrix with values, than should no be included in
               predictions
        :param drop_cold_users: if True, then all non active users will be dropped
        :param batch_size: Batch size of users to predict iteratively, to saving memory
        :return: Matrix with the recommendations for each of the test users
        """

        # Extracting all indices for the matrix
        all_indices = np.arange(0, test_matrix.shape[0])

        # Variable to save all predictions
        predictions_all = []

        # Iterating over batches to make predictions
        for start_ind in tqdm(range(0, test_matrix.shape[0], batch_size)):
            batch_indices = all_indices[start_ind: start_ind + batch_size]

            # Transforming some type of the "scores"
            predicted_popularity = np.tile(self.popularity, (len(batch_indices), 1)).astype('float16')

            # Excluding the predictions if needed
            if excluding_predictions is not None:
                predicted_popularity[excluding_predictions[batch_indices, :].nonzero()] = -np.inf

            # Make recommendations
            predictions = np.argsort(predicted_popularity, axis=1)[:, ::-1][:, :number_of_predictions]

            # Appending results
            predictions_all.extend(predictions)

        predictions_all = np.array(predictions_all)

        # Deleting cold users
        if drop_cold_users:
            predictions_all = predictions_all[test_matrix.getnnz(axis=1) > 0]

        return predictions_all


class SVDRecommender(TruncatedSVD):
    def __init__(self,
                 n_components: int = 2,
                 algorithm: str = 'randomized',
                 n_iter: int = 5,
                 random_state=None,
                 tol: float = 0.0):
        """
        Init SVDRecommender model as a heir of TruncatedSVD from sklern

        :param n_components: Desired dimensionality of output data. Must be strictly less than the number of features.
        :param algorithm: SVD solver to use.
        :param n_iter: Number of iterations for randomized SVD solver.
        :param random_state: Used during randomized svd.
        :param tol: Tolerance for ARPACK.
        """

        super(SVDRecommender, self).__init__(n_components=n_components, algorithm=algorithm,
                                             n_iter=n_iter, random_state=random_state, tol=tol)

        # Init user matrix as a matrix of embeddings
        self.user_matrix = None

    def fit(self,
            train_matrix: sp.csr_matrix,
            **kwargs) -> None:
        """
        Method to fit SVD recommender

        :param train_matrix: Sparse matrix to train model
        """

        # Fitting model
        super(SVDRecommender, self).fit(train_matrix)

        # Generating user embeddings
        self.user_matrix = self.transform(train_matrix)

    def predict(self,
                train_matrix: sp.csr_matrix,
                test_matrix: sp.csr_matrix,
                number_of_predictions: int = 100,
                excluding_predictions=None,
                drop_cold_users: bool = True,
                batch_size=1000,
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
            scores = self.inverse_transform(self.user_matrix[batch_indices, :])

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
