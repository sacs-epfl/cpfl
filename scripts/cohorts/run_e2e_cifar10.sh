#!/bin/bash

# Ensure the script receives exactly two arguments
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <number_of_cohorts> <seed> <alpha> <peers>"
    exit 1
fi

# Extract the arguments
COHORTS=$1
SEED=$2
ALPHA=$3
PEERS=$4
PARTICIPATION=50
PATIENCE=100
LR=0.002
CLUSTER_METHOD="uniform"
DATASET="cifar10"

python3 -u scripts/cohorts/create_cohort_file.py $PEERS $COHORTS \
--seed $SEED \
--partitioner dirichlet \
--alpha $ALPHA \
--dataset $DATASET \
--method $CLUSTER_METHOD \
--output "data/cohorts/cohorts_n${PEERS}_c${COHORTS}_s${SEED}_a${ALPHA}_${CLUSTER_METHOD}.txt"

python3 -u simulations/cpfl/${DATASET}.py \
--cohort-participation $PARTICIPATION \
--peers ${PEERS} \
--duration 0 \
--num-aggregators 1 \
--activity-log-interval 3600 \
--bypass-model-transfers \
--seed $SEED \
--stop-criteria-patience $PATIENCE \
--capability-trace data/client_device_capacity \
--accuracy-logging-interval 10 \
--partitioner dirichlet \
--alpha $ALPHA \
--fix-aggregator \
--cohort-file "cohorts/cohorts_n${PEERS}_c${COHORTS}_s${SEED}_a${ALPHA}_${CLUSTER_METHOD}.txt" \
--compute-validation-loss-global-model \
--log-level "ERROR" \
--local-steps 5 \
--learning-rate $LR \
--checkpoint-interval 18000 \
--checkpoint-interval-is-in-sec > output_${DATASET}_c${COHORTS}_s${SEED}_a${ALPHA}_p${PARTICIPATION}_${CLUSTER_METHOD}.log 2>&1
