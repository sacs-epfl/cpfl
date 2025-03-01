# Cohort-Parallel Federated Learning (CPFL)
Repository for the source code of our paper *[Harnessing Increased Client Participation with Cohort-Parallel Federated Learning](https://arxiv.org/pdf/2405.15644)* published at [The Workshop on Machine Learning and System 2025](https://euromlsys.eu/#).

## Abstract

Federated learning (FL) is a machine learning approach where nodes collaboratively train a global model.
As more nodes participate in a round of FL, the effectiveness of individual model updates by nodes also diminishes.
In this study, we increase the effectiveness of client updates by dividing the network into smaller partitions, or _cohorts_.
We introduce Cohort-Parallel Federated Learning (CPFL): a novel learning approach where each cohort independently trains a global model using FL, until convergence, and the produced models by each cohort are then unified using knowledge distillation.
The insight behind CPFL is that smaller, isolated networks converge quicker than in a one-network setting where all nodes participate.
Through exhaustive experiments involving realistic traces and non-IID data distributions on the CIFAR-10 and FEMNIST image classification tasks, we investigate the balance between the number of cohorts, model accuracy, training time, and compute resources.
Compared to traditional FL, CPFL with four cohorts, non-IID data distribution, and CIFAR-10 yields a 1.9x reduction in train time and a 1.3x reduction in resource usage, with a minimal drop in test accuracy.

## Installation

Start by cloning the repository recursively (since CPFL depends on the PyIPv8 networking library):

```
git clone git@github.com:sacs-epfl/cpfl.git --recursive
```

Install the required dependencies (preferably in a virtual environment to avoid conflicts with existing libraries):

```
pip install -r requirements.txt
```

In our paper, we evaluate CPFL using the CIFAR-10 and FEMNIST datasets.
For CIFAR-10 we use `torchvision`. The FEMNIST dataset has to be downloaded manually and we refer the reader to the [decentralizepy framework](https://github.com/sacs-epfl/decentralizepy) that uses the same dataset.

## Running CPFL

Training with CPFL can be done by invoking the following scripts from the root of the repository:

```
# Running with the CIFAR-10 dataset
bash scripts/cohorts/run_e2e_cifar10.sh <number_of_cohorts> <seed> <alpha> <peers>

# Running with the FEMNIST dataset
bash scripts/cohorts/run_2e2_femnist.sh <number_of_cohorts> <seed>
```

We refer to the respective bash scripts for more configuration options, such as the number of local steps, the number of participants, and other learning parameters.

The script first splits the data across participants and participants across cohorts. These assignments are used during the distillation process.
Then during FL training, each cohort will periodically checkpoint the current global model, as well as checkpointing the current best model (based on the loss obtained with a validation testset). The output of this experiment can be found in a separate folder in the `data` directory.

After training, the checkpointed models can be distilled into a single model using the following command:

```
python3 scripts/distill.py $PWD/data n_200_cifar10_dirichlet0.100000_sd24082_ct10_dfl cifar10 stl10 --cohort-file cohorts/cohorts_cifar10_n200_c10.txt --public-data-dir <path_to_public_data> --learning-rate 0.001 --momentum 0.9 --partitioner dirichlet --alpha 0.1 --weighting-scheme label --check-teachers-accuracy > output_distill.log 2>&1
```

The above command invokes the `distill.py` script that scans the models in the `n_200_cifar10_dirichlet0.100000_sd24082_ct10_dfl` directory (created by the previous experiment) and merges them.
The command also requires the path to the cohort information file created during the previous steps.
The `distill.py` script automatically determines the attained accuracies of the obtained model after distillation.

## Reference

If you find our work useful, you can cite us as follows:

```
@inproceedings{dhasade2025cpfl,
  title={Harnessing Increased Client Participation with Cohort-Parallel Federated Learning},
  author={Dhasade, Akash and Kermarrec, Anne-Marie and Nguyen, Tuan-Ahn and Pires, Rafael and de Vos, Martijn},
  booktitle={Proceedings of the 5th Workshop on Machine Learning and Systems},
  year={2025}
}
```
