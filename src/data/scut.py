import numpy as np
import pathlib
from src import REPO_ROOT
import os
import struct
import argparse
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt


class Dataset:
    def __init__(
        self, img_path: str | pathlib.Path, point_path: str | pathlib.Path
    ) -> None:
        self.img_path = img_path
        self.point_path = point_path
        self.file_names = list(
            map(lambda x: x.replace(".jpg", ""), os.listdir(img_path))
        )
        print(len(self.file_names), self.file_names[0:5])

    def display(self, idx: int, show_points: bool = True):
        fp = os.path.join(self.img_path, f"{self.file_names[idx]}.jpg")
        img = Image.open(fp)

        plt.imshow(img)
        if show_points:
            plt.scatter(*self._load_points(idx).transpose())
        plt.show()

    def _load_points(self, idx: int) -> np.ndarray:
        fp = os.path.join(self.point_path, f"{self.file_names[idx]}.txt")
        with open(fp, "r") as f:
            data = f.read()
        data_rows = data.splitlines()
        arr = np.zeros((len(data_rows), 2), dtype=np.float32)
        for i, row in enumerate(data_rows):
            x, y = map(float, row.split())
            arr[i, 0] = x
            arr[i, 1] = y
        return arr


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
        ds = Dataset(args.img_path, args.points_path)
        ds.display(0)
    else:
        raise Exception(f"Expected one of the modes: {modes}")
