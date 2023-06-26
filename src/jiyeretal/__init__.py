from src.jiyeretal import facial_features
from src.jiyeretal import tcs
from dataclasses import dataclass
from typing import Generator, List, Tuple
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from src.data.scut import ScutDataset
from src import REPO_ROOT


@dataclass
class SecondaryFeature:
    name: str
    score: float
    facial_ratios: facial_features.FacialRatio
    symmetry_ratios: facial_features.SymmetriRatios
    texture_color_shape: tcs.TCS

    def __str__(self) -> str:
        return (
            "#" * 10
            + "\n"
            + f" name: {self.name}\n score: {self.score}\n facial_ratios: {str(self.facial_ratios)}\n symmetry_ratios: {str(self.symmetry_ratios)}"
        )


class SecondaryFeatureSet:
    features: List[SecondaryFeature]

    def __init__(self, features: List[SecondaryFeature]) -> None:
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __iter__(self) -> Generator[SecondaryFeature, None, None]:
        for i in range(len(self)):
            yield self[i]

    def get_Xy(self) -> Tuple[np.ndarray, List[float]]:
        l = []
        scores = []
        for f in self.features:
            l.append(
                [
                    *f.facial_ratios.to_numpy(),
                    *f.symmetry_ratios.to_numpy(),
                    *f.texture_color_shape.to_numpy(),
                ]
            )
            scores.append(f.score)
        arr = np.array(l)
        return np.nan_to_num(arr, nan=0), scores

    @classmethod
    def from_scut_dataset(cls, dataset: ScutDataset):
        features: List[SecondaryFeature] = []
        print("Extracting features")
        for img in tqdm(dataset):
            facial_ratios = facial_features.FacialRatio(img.landmarks)
            symmetry_ratios = facial_features.SymmetriRatios(img.landmarks)
            texture_color_shape = tcs.TCS(img.img)
            features.append(
                SecondaryFeature(
                    img.name,
                    img.fs,
                    facial_ratios,
                    symmetry_ratios,
                    texture_color_shape,
                )
            )
        return cls(features)

    def __getitem__(self, idx):
        return self.features[idx]


if __name__ == "__main__":
    print("Calculating and saving feature extraction")
    ds = ScutDataset(
        (REPO_ROOT / "data/SCUT-FBP5500_v2/Images").resolve(),
        (REPO_ROOT / "data/SCUT-FBP5500_v2/facial_landmark_txt").resolve(),
        (REPO_ROOT / "data/SCUT-FBP5500_v2/train_test_files/All_labels.txt"),
    )

    feature_set = SecondaryFeatureSet.from_scut_dataset(ds)
    with open(
        (
            REPO_ROOT / "data/extracted_features/scut_extracted_features.pickle"
        ).resolve(),
        "wb",
    ) as file_:
        pickle.dump(feature_set, file_)
