import os
import numpy as np
from sklearn.model_selection import KFold
import sys
sys.path.append('.')

from rampwf.score_types.negative_log_likelihood import NegativeLogLikelihood
from workflows.Image_regressor import ImageRegressor
from utils.create_dataset import create_dataset_from_raw, create_shifted_frames
from prediction_types.regression_3D import make_image_regression

problem_title = "Short-term precipitation forecasting"


# # ----------------------------------------------------------------------------
# # Worklow element
# # ----------------------------------------------------------------------------

workflow = ImageRegressor()

# # ----------------------------------------------------------------------------
# # Predictions type
# # ----------------------------------------------------------------------------
re_height = 200
re_width = 200

_prediction_label_names = list(range(re_height * re_width * 18))

Predictions = make_image_regression(label_names=_prediction_label_names)


# # ----------------------------------------------------------------------------
# # Score types
# # ----------------------------------------------------------------------------

score_types = [
    NegativeLogLikelihood(name='nll')
]

# # ----------------------------------------------------------------------------
# # Cross-validation scheme
# # ----------------------------------------------------------------------------


def get_cv(X, y):
    # using 5 folds as default
    k = 2
    # up to 10 fold cross-validation based on 5 splits, using two parts for
    # testing in each fold
    n_splits = 5
    cv = KFold(n_splits=n_splits)
    splits = list(cv.split(X, y))
    # 5 folds, each point is in test set 4x
    # set k to a lower number if you want less folds
    pattern = [
        ([2, 3, 4], [0, 1]),
        ([0, 1, 4], [2, 3]),
        ([0, 2, 3], [1, 4]),
        ([0, 1, 3], [2, 4]),
        ([1, 2, 4], [0, 3]),
        ([0, 1, 2], [3, 4]),
        ([0, 2, 4], [1, 3]),
        ([1, 2, 3], [0, 4]),
        ([0, 3, 4], [1, 2]),
        ([1, 3, 4], [0, 2]),
    ]
    for ps in pattern[:k]:
        yield (
            np.hstack([splits[p][1] for p in ps[0]]),
            np.hstack([splits[p][1] for p in ps[1]]),
        )


# # ----------------------------------------------------------------------------
# # Training / testing data reader
# # ----------------------------------------------------------------------------

def _read_data(path):
    dataset = create_dataset_from_raw(path, resize_to=(re_width, re_height))
    dataset = np.expand_dims(dataset, axis=-1)
    dataset_x, dataset_y = create_shifted_frames(dataset)
    return dataset_x, dataset_y

def get_train_data(path="."):
    return _read_data(os.path.join(path, 'data/public_train_raw/'))


def get_test_data(path="."):
        return _read_data(os.path.join(path, 'data/public_test_raw/'))
