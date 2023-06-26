from src.data.scut import ScutDataset
from src import REPO_ROOT
from src.jiyeretal import SecondaryFeatureSet
import numpy as np
from sklearn.model_selection import train_test_split

from src.models import linear_regression


def train_lr(X, y):
    return linear_regression.train(X, y)


if __name__ == "__main__":
    ds = ScutDataset(
        (REPO_ROOT / "data/SCUT-FBP5500_v2/Images").resolve(),
        (REPO_ROOT / "data/SCUT-FBP5500_v2/facial_landmark_txt").resolve(),
        (REPO_ROOT / "data/SCUT-FBP5500_v2/train_test_files/All_labels.txt"),
    )
    feature_set = SecondaryFeatureSet.from_scut_dataset(ds)
    X, y = feature_set.get_Xy()
    breakpoint()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    lr_model = train_lr(X_train, y_train)
    print(lr_model.score(X_test, y_test))
