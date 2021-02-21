## Attend and Discriminate: *Beyond the State-of-the-Art for Human Activity Recognition Using Wearable Sensors*

### Introduction

Wearable sensors are fundamental to improving our understanding of human activities, especially for an increasing number 
of healthcare applications from rehabilitation to fine-grained gait analysis. Although our collective know-how to solve 
Human Activity Recognition (HAR) problems with wearables has progressed immensely with end-to-end deep learning paradigms, 
several fundamental opportunities remain overlooked. 

The repository provides the codebase and pre-trained network parameters corresponding to our [publication](https://arxiv.org/abs/2007.07172) 
(to appear in IMWUT 2021), where we rigorously explore new opportunities to learn enriched and highly discriminating activity representations.

### Citation
If you find our work useful in your research, please consider citing:

	@article{10.1145/3448083,
	  title={Attend And Discriminate: Beyond the State-of-the-Art for Human Activity Recognition using Wearable Sensors},
	  author={Abedin, Alireza and Ehsanpour, Mahsa and Shi, Qinfeng and Rezatofighi, Hamid and Ranasinghe, Damith C},
	  journal = {Proc. ACM Interact. Mob. Wearable Ubiquitous Technol.},
	  volume = {5},
      number = {1},
	  doi={10.1145/3448083},
	  year={2021},
	}



## Quick Start

### Downloads

Firstly, clone this repository

```
git clone https://github.com/AdelaideAuto-IDLab/Attend-And-Discriminate.git
```

Subsequently, download the raw HAR [datasets](https://universityofadelaide.box.com/s/ag10ugotoqmbw3sw6q74s0pd7b7gkznj), and place the zip file inside the cloned directory.
Similarly, download the HAR architecture [weights](https://universityofadelaide.box.com/s/jwmfsri4s5092qfc83rbsbwt0dcpci4x), and place the zip file inside the cloned directory. Next,
run the bash script within the cloned directory to uncompress the zip files and cleanup the downloads:

```
bash clean_download.sh
```


### Training 
In order to train a model with the corresponding HAR dataset, simply run:

```
python main.py --dataset [opportunity,pamap2,skoda,hospital] --train_mode
```

The executed script generates the corresponding HAR dataset in `data/{dataset}`, and stores the experiment artefacts in
`models/{experiment}` directory. 

As training progresses, three sets of outputs are stored in `models/{experiment}/`:

* `checkpoints`: The model weights.
* `logs`: The tensorboard logged loss and evaluation metrics.
* `visuals`: The generated confusion matrices. 

We can monitor the training progress using TensorBoard:

```
tensorboard --logdir ./models/{experiment}
```

By default the experiment name is determined by the current system time. To customize training parameters, 
arguments can be set in `settings.py` (e.g. `dataset`, `batch_size`, `lr`,...). Further, to view the argument helper documentation: 

```
python main.py --help

usage: main.py [-h] [--experiment EXPERIMENT] [--train_mode]
               [--dataset {opportunity,skoda,pamap2,hospital}]
               [--window WINDOW] [--stride STRIDE] [--stride_test STRIDE_TEST]
               [--model MODEL] [--epochs EPOCHS] [--load_epoch LOAD_EPOCH]
               [--print_freq PRINT_FREQ]

HAR dataset, model and optimization arguments

optional arguments:
  -h, --help            show this help message and exit
  --experiment EXPERIMENT
                        experiment name (default: None)
  --train_mode          execute code in training mode (default: False)
  --dataset {opportunity,skoda,pamap2,hospital}
                        HAR dataset (default: opportunity)
  --window WINDOW       sliding window size (default: 24)
  --stride STRIDE       sliding window stride (default: 12)
  --stride_test STRIDE_TEST
                        set to 1 for sample-wise prediction (default: 1)
  --model MODEL         HAR architecture (default: AttendDiscriminate)
  --epochs EPOCHS       number of training epochs (default: 300)
  --load_epoch LOAD_EPOCH
                        epoch to resume training (default: 0)
  --print_freq PRINT_FREQ

```

### Inference 

In order to create a pre-trained HAR architecture and run inference on the corresponding hold-out test splits, simply run:

```
python main.py --dataset [opportunity,pamap2,skoda,hospital]
```

The executed script loads the pre-trained weights from `weights/checkpoint_{dataset}.pth`, generates the test dataset and reports
the recognition performance metrics. 

## Core Modules

#### Datasets Module

The `datasets.py` module extends the PyTorch Dataset class for multi-channel time-series data 
captured by wearable sensors. The module functionality can be tested
by simply running
```
python datasets.py --dataset [opportunity,pamap2,skoda,hospital]
```
The executed script creates the corresponding HAR dataset in `data` directory.
This loads the corresponding dataset files, runs sliding window segmentation to partition the raw samples and 
displays additional information on the created dataset and individual data segments.  

#### Models Module

The `models.py` module manages the HAR architecture according to the configured settings. The module functionality can be tested
by simply running
```
python models.py --dataset [opportunity,pamap2,skoda,hospital]
```
The executed script creates the corresponding HAR architecture for the dataset. It also displays a summary of trainable 
network parameters and layer information, and runs a synthetic data batch through the network to verify the correctness of input/output tensor dimensionalities. 