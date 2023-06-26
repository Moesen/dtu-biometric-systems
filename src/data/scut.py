import numpy as np
from typing import Tuple
import pathlib
from src import REPO_ROOT
import os
import struct
import argparse
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataclasses import dataclass


class ScutImgInfo:
    def __init__(
        self,
        img_path: str | pathlib.Path,
        point_path: str | pathlib.Path,
        fs: float,
    ) -> None:
        self.img_path = img_path
        self.point_path = point_path
        self.fs = fs


@dataclass
class ScutImgLoaded:
    img: Image.Image
    landmarks: np.ndarray
    fs: float
    name: str


class ScutDataset:
    def __init__(
        self,
        img_path: str | pathlib.Path,
        point_path: str | pathlib.Path,
        fs_path: str | pathlib.Path,
    ) -> None:
        img_path = img_path
        point_path = point_path
        point_files = os.listdir(point_path)
        file_names = list(
            filter(
                lambda x: any(x in y for y in point_files),
                map(lambda x: x.replace(".jpg", ""), os.listdir(img_path)),
            )
        )
        fs_scores = {}
        for line in open(fs_path).read().splitlines():
            a, b = line.split()
            fs_scores[a.replace(".jpg", "")] = float(b)

        self.ScutImgInfos = []
        for fn in file_names:
            self.ScutImgInfos.append(
                ScutImgInfo(
                    os.path.join(img_path, f"{fn}.jpg"),
                    os.path.join(
                        point_path,
                        f"{fn}.txt",
                    ),
                    fs_scores[fn],
                )
            )

    def __getitem__(self, idx) -> ScutImgLoaded:
        info = self.ScutImgInfos[idx]
        ip, pp, fs = info.img_path, info.point_path, info.fs
        img = Image.open(ip)
        with open(pp, "r") as f:
            points = np.array(
                [[*map(float, x.split(" "))] for x in f.read().splitlines()]
            )
        return ScutImgLoaded(img, points, fs, ip.split("/")[-1].replace(".jpg", ""))

    def __iter__(self):
        for x in range(len(self)):
            yield self[x]

    def __len__(self) -> int:
        return len(self.ScutImgInfos)

    def display(self, idx: int, show_points: bool = True):
        scut = self[idx]

        plt.imshow(scut.img)
        if show_points:
            plt.scatter(*scut.landmarks.transpose())
        plt.title(f"{scut.name=}: {scut.fs=}")
        plt.show()


def pts2txt(din, dout, src):
    src_p = os.path.join(din, src)
    data = open(src_p, "rb").read()
    if len(data) < 692:
        return 0
    points = struct.unpack("i172f", data)

    dst = src
    dst = dst.replace("pts", "txt")
    dst_p = os.path.join(dout, dst)

    with open(dst_p, "w") as fout:
        pnum = len(points[1:])
        for i in range(1, pnum, 2):
            fout.write("%f " % points[i])
            fout.write("%f\n" % points[i + 1])

    return 1


class ArgSpace:
    points_path: str
    output_path: str
    img_path: str
    fs_path: str
    mode: str


modes = ["conversion", "datatest"]
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--points_path", help="path to .pts files")
    parser.add_argument(
        "-o",
        "--output_path",
        help="path to folder, where .txt files should be stored",
    )
    parser.add_argument("-f", "--fs_path", help="Path to fs file")
    parser.add_argument("-m", "--mode", required=True, help=f"Options: {modes}")
    parser.add_argument("-i", "--img_path", help="Path to images")
    args = parser.parse_args(namespace=ArgSpace)
    if args.mode == "conversion":
        print(
            f"Found {len(os.listdir(args.points_path))} in {args.points_path}. Beginning conversion"
        )
        for fn in tqdm(os.listdir(args.points_path)):
            pts2txt(args.points_path, args.output_path, fn)
        print(f"Done with conversion. Saved to {args.output_path}")
    elif args.mode == "datatest":
        ds = ScutDataset(args.img_path, args.points_path, args.fs_path)
        ds.display(0)
    else:
        raise Exception(f"Expected one of the modes: {modes}")
