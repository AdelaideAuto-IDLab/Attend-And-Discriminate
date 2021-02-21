import sys
import os

import scipy.io as sio
import numpy as np

from utils.utils import Logger, paint
from utils.utils_plot import plot_pie

from settings import get_args


__all__ = ["preprocess_pipeline"]


def load_mat(path_data, path_raw, class_map):

    # load .mat files
    print(f"[*] Reading data files from {path_data}")
    contents = sio.loadmat(path_data)

    if len(class_map) == 18:
        # opportunity dataset
        x_train = contents["trainingData"].astype(np.float32).T
        y_train = contents["trainingLabels"].reshape(-1).astype(np.int64) - 1
        x_val = contents["valData"].astype(np.float32).T
        y_val = contents["valLabels"].reshape(-1).astype(np.int64) - 1
        x_test = contents["testingData"].astype(np.float32).T
        y_test = contents["testingLabels"].reshape(-1).astype(np.int64) - 1

        # normalizing
        mean_train = np.mean(x_train, axis=0)
        std_train = np.std(x_train, axis=0)
        x_train = (x_train - mean_train) / std_train
        x_val = (x_val - mean_train) / std_train
        x_test = (x_test - mean_train) / std_train

    elif len(class_map) == 7:
        # hospital dataset
        x_train = contents["X_train"].astype(np.float32)
        y_train = contents["y_train"].reshape(-1).astype(np.int64)
        x_val = contents["X_valid"].astype(np.float32)
        y_val = contents["y_valid"].reshape(-1).astype(np.int64)
        x_test = contents["X_test"].astype(np.float32)
        y_test = contents["y_test"].reshape(-1).astype(np.int64)

        # normalizing
        mean_train = np.mean(x_train, axis=0)
        std_train = np.std(x_train, axis=0)
        x_train = (x_train - mean_train) / std_train
        x_val = (x_val - mean_train) / std_train
        x_test = (x_test - mean_train) / std_train

    else:
        # all other datasets
        x_train = contents["X_train"].astype(np.float32)
        y_train = contents["y_train"].reshape(-1).astype(np.int64)
        x_val = contents["X_valid"].astype(np.float32)
        y_val = contents["y_valid"].reshape(-1).astype(np.int64)
        x_test = contents["X_test"].astype(np.float32)
        y_test = contents["y_test"].reshape(-1).astype(np.int64)

    # show raw datasets info (sample-level)
    print(
        "[-] Train data : {} {}, target {} {}".format(
            x_train.shape, x_train.dtype, y_train.shape, y_train.dtype
        )
    )
    print(
        "[-] Valid data : {} {}, target {} {}".format(
            x_val.shape, x_val.dtype, y_val.shape, y_val.dtype
        )
    )
    print(
        "[-] Test data : {} {}, target {} {}".format(
            x_test.shape, x_test.dtype, y_test.shape, y_test.dtype
        )
    )

    # plot raw target distributions (sample-level)
    plot_pie(y_train, "train", path_raw, class_map)
    plot_pie(y_val, "val", path_raw, class_map)
    plot_pie(y_test, "test", path_raw, class_map)

    # save raw datasets (sample-level)
    np.savez_compressed(os.path.join(path_raw, "train.npz"), x=x_train, y=y_train)
    np.savez_compressed(os.path.join(path_raw, "val.npz"), x=x_val, y=y_val)
    np.savez_compressed(os.path.join(path_raw, "test.npz"), x=x_test, y=y_test)
    print("[+] Raw sample datasets successfully saved!")
    print(paint("--" * 50, "blue"))


def partition(path_raw, path_processed, window, stride, class_map):

    # read raw datasets (sample-level)
    print(f"[*] Reading raw files from {path_raw}")
    dataset_train = np.load(os.path.join(path_raw, "train.npz"))
    x_train, y_train = dataset_train["x"], dataset_train["y"]
    dataset_val = np.load(os.path.join(path_raw, "val.npz"))
    x_val, y_val = dataset_val["x"], dataset_val["y"]
    dataset_test = np.load(os.path.join(path_raw, "test.npz"))
    x_test, y_test = dataset_test["x"], dataset_test["y"]

    # apply sliding window over raw samples and generate segments
    data_train, target_train = sliding_window(x_train, y_train, window, stride)
    data_val, target_val = sliding_window(x_val, y_val, window, stride)
    data_test, target_test = sliding_window(x_test, y_test, window, stride)
    data_test_sample_wise, target_test_sample_wise = sliding_window(
        x_test, y_test, window, 1
    )

    # show processed datasets info (segment-level)
    print(
        "[-] Train data : {} {}, target {} {}".format(
            data_train.shape, data_train.dtype, target_train.shape, target_train.dtype
        )
    )
    print(
        "[-] Valid data : {} {}, target {} {}".format(
            data_val.shape, data_val.dtype, target_val.shape, target_val.dtype
        )
    )
    print(
        "[-] Test data : {} {}, target {} {}".format(
            data_test.shape, data_test.dtype, target_test.shape, target_test.dtype
        )
    )
    print(
        "[-] Test data sample-wise : {} {}, target sample-wise {} {}".format(
            data_test_sample_wise.shape,
            data_test_sample_wise.dtype,
            target_test_sample_wise.shape,
            target_test_sample_wise.dtype,
        )
    )

    # plot processed target distributions (segment-level)
    plot_pie(target_train, "train", path_processed, class_map)
    plot_pie(target_val, "val", path_processed, class_map)
    plot_pie(target_test, "test", path_processed, class_map)
    plot_pie(target_test_sample_wise, "test_sample_wise", path_processed, class_map)

    # save processed datasets (segment-level)
    np.savez_compressed(
        os.path.join(path_processed, "train.npz"), data=data_train, target=target_train
    )
    np.savez_compressed(
        os.path.join(path_processed, "val.npz"), data=data_val, target=target_val
    )
    np.savez_compressed(
        os.path.join(path_processed, "test.npz"), data=data_test, target=target_test
    )
    np.savez_compressed(
        os.path.join(path_processed, "test_sample_wise.npz"),
        data=data_test_sample_wise,
        target=target_test_sample_wise,
    )
    print("[+] Processed segment datasets successfully saved!")
    print(paint("--" * 50, "blue"))


def sliding_window(x, y, window, stride, scheme="last"):

    data, target = [], []
    start = 0
    while start + window < x.shape[0]:
        end = start + window
        x_segment = x[start:end]
        if scheme == "last":
            # last scheme: : last observed label in the window determines the segment annotation
            y_segment = y[start:end][-1]
        elif scheme == "max":
            # max scheme: most frequent label in the window determines the segment annotation
            y_segment = np.argmax(np.bincount(y[start:end]))
        data.append(x_segment)
        target.append(y_segment)
        start += stride

    data = np.array(data, dtype=np.float32)
    target = np.array(target, dtype=np.int64)

    return data, target


def preprocess_pipeline(args):

    # [STEP 0] load the .mat files (sample-level)
    if not os.path.exists(args.path_raw):
        sys.stdout = Logger(os.path.join(args.path_raw, "log_raw.txt"))
        print(paint("[STEP 0] Loading the .mat files..."))
        load_mat(
            path_data=args.path_data, path_raw=args.path_raw, class_map=args.class_map
        )
    else:
        print(paint("[STEP 0] Files already loaded!"))

    # [STEP 1] partition the datasets (segment-level)
    w, s = args.window, args.stride
    if not os.path.exists(args.path_processed):
        sys.stdout = Logger(os.path.join(args.path_processed, f"log_{w}_{s}.txt"))
        print(
            paint(f"[STEP 1] Partitioning the dataset (window,stride) = ({w},{s})...")
        )
        partition(
            path_raw=args.path_raw,
            path_processed=args.path_processed,
            window=w,
            stride=s,
            class_map=args.class_map,
        )
    else:
        print(
            paint(f"[STEP 1] Dataset already partitioned (window,stride) = ({w},{s})!")
        )


def main():

    # get experiment arguments
    args, _, _ = get_args()
    preprocess_pipeline(args)


if __name__ == "__main__":
    main()
