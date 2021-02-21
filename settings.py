import sys
import argparse
import datetime

__all__ = ["get_args"]


def get_args():

    parser = argparse.ArgumentParser(
        description="HAR dataset, model and optimization arguments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # get HAR arguments
    parser.add_argument("--experiment", default=None, help="experiment name")
    parser.add_argument(
        "--train_mode", action="store_true", help="execute code in training mode"
    )
    parser.add_argument(
        "--dataset",
        default="opportunity",
        type=str,
        choices=["opportunity", "skoda", "pamap2", "hospital"],
        help="HAR dataset",
    )
    parser.add_argument("--window", default=24, type=int, help="sliding window size")
    parser.add_argument("--stride", default=12, type=int, help="sliding window stride")
    parser.add_argument(
        "--stride_test", default=1, type=int, help="set to 1 for sample-wise prediction"
    )
    parser.add_argument(
        "--model", default="AttendDiscriminate", type=str, help="HAR architecture"
    )
    parser.add_argument(
        "--epochs", default=300, type=int, help="number of training epochs"
    )
    parser.add_argument(
        "--load_epoch", default=0, type=int, help="epoch to resume training"
    )
    parser.add_argument("--print_freq", default=40, type=int)
    args = parser.parse_args()

    if args.experiment is None:
        args.experiment = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

    # get HAR dataset arguments
    if args.dataset == "opportunity":
        args.num_class = 18
        args.input_dim = 79
        args.class_map = [
            "Null",
            "Open Door 1",
            "Open Door 2",
            "Close Door 1",
            "Close Door 2",
            "Open Fridge",
            "Close Fridge",
            "Open Dishwasher",
            "Close Dishwasher",
            "Open Drawer 1",
            "Close Drawer 1",
            "Open Drawer 2",
            "Close Drawer 2",
            "Open Drawer 3",
            "Close Drawer 3",
            "Clean Table",
            "Drink from Cup",
            "Toggle Switch",
        ]

    elif args.dataset == "skoda":
        args.num_class = 11
        args.input_dim = 60
        args.class_map = [
            "Null",
            "Write on Notepad",
            "Open Hood",
            "Close Hood",
            "Check Door Gaps",
            "Open Left Front Door",
            "Close Left Front Door",
            "Close Both Left Doors",
            "Check Trunk Gaps",
            "Open and Close Trunk",
            "Check Steering Wheel",
        ]

    elif args.dataset == "pamap2":
        args.num_class = 12
        args.input_dim = 52
        args.class_map = [
            "Rope Jumping",
            "Lying",
            "Sitting",
            "Standing",
            "Walking",
            "Running",
            "Cycling",
            "Nordic Walking",
            "Ascending Stairs",
            "Descending Stairs",
            "Vacuum Cleaning",
            "Ironing",
        ]

    elif args.dataset == "hospital":
        args.num_class = 7
        args.input_dim = 6
        args.class_map = [
            "Lying",
            "Standing Up",
            "Sitting",
            "Walking",
            "Lying Down",
            "Sitting Down",
            "Getting Up",
        ]
    else:
        print(f"[!] Unknown HAR dataset: {args.dataset}")
        sys.exit(0)

    args.path_data = f"./dataset/{args.dataset}.mat"
    args.path_raw = f"./data/{args.dataset}/raw/"
    args.path_processed = f"./data/{args.dataset}/processed/{args.window}_{args.stride}"

    # get HAR optimization arguments
    args.weighted_sampler = False

    args.batch_size = 256
    args.optimizer = "Adam"
    args.clip_grad = 0
    args.lr = 0.001
    args.lr_decay = 0.9
    args.lr_step = 10

    args.mixup = True
    args.alpha = 0.8

    args.lr_cent = 0.001

    if args.dataset == "opportunity":
        args.init_weights = "orthogonal"
        args.beta = 0.0003
        args.dropout = 0.5
        args.dropout_rnn = 0.25
        args.dropout_cls = 0.5

    elif args.dataset == "pamap2":
        args.init_weights = None
        args.beta = 0.003
        args.dropout = 0.9
        args.dropout_rnn = 0
        args.dropout_cls = 0.5

    elif args.dataset == "skoda":
        args.init_weights = "orthogonal"
        args.beta = 0.3
        args.dropout = 0.5
        args.dropout_rnn = 0.25
        args.dropout_cls = 0

    elif args.dataset == "hospital":
        args.init_weights = "orthogonal"
        args.beta = 0.3
        args.dropout = 0.5
        args.dropout_rnn = 0.25
        args.dropout_cls = 0.5

    # get HAR model arguments
    if args.model == "AttendDiscriminate":
        args.filter_num, args.filter_size = 64, 5
        args.enc_num_layers = 2
        args.enc_is_bidirectional = False
        args.hidden_dim = 128
        args.activation = "ReLU"
        args.sa_div = 1

    # set dataset and model arguments
    config_dataset = {
        "dataset": args.dataset,
        "window": args.window,
        "stride": args.stride,
        "stride_test": args.stride_test,
        "path_processed": args.path_processed,
    }
    config_model = {
        "model": args.model,
        "dataset": args.dataset,
        "input_dim": args.input_dim,
        "hidden_dim": args.hidden_dim,
        "filter_num": args.filter_num,
        "filter_size": args.filter_size,
        "enc_num_layers": args.enc_num_layers,
        "enc_is_bidirectional": args.enc_is_bidirectional,
        "dropout": args.dropout,
        "dropout_rnn": args.dropout_rnn,
        "dropout_cls": args.dropout_cls,
        "activation": args.activation,
        "sa_div": args.sa_div,
        "num_class": args.num_class,
        "train_mode": args.train_mode,
        "experiment": args.experiment,
    }

    return args, config_dataset, config_model


if __name__ == "__main__":
    get_args()
