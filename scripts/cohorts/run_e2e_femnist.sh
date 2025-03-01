#!/bin/bash

# Ensure the script receives exactly three arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <number_of_cohorts> <seed>"
    exit 1
fi

# Extract the arguments
COHORTS=$1
SEED=$2
PARTICIPATION=50
PATIENCE=100
PEERS=3597
LR=0.1
CLUSTER_METHOD="uniform"
DATASET="femnist"

python3 -u scripts/cohorts/create_cohort_file.py $PEERS $COHORTS \
--seed $SEED \
--partitioner realworld \
--dataset $DATASET \
--method $CLUSTER_METHOD \
--output "data/cohorts/cohorts_n${PEERS}_c${COHORTS}_s${SEED}_${CLUSTER_METHOD}.txt" \
--data-dir /mnt/nfs/devos/leaf/${DATASET}

python3 -u simulations/dfl/${DATASET}.py \
--dataset-base-path "/mnt/nfs/devos" \
--peers $PEERS \
--local-steps 5 \
--cohort-participation $PARTICIPATION \
--duration 0 \
--num-aggregators 1 \
--activity-log-interval 3600 \
--bypass-model-transfers \
--seed $SEED \
--stop-criteria-patience $PATIENCE \
--capability-trace data/client_device_capacity \
--accuracy-logging-interval 10 \
--partitioner realworld \
--fix-aggregator \
--cohort-file "cohorts/cohorts_n${PEERS}_c${COHORTS}_s${SEED}_${CLUSTER_METHOD}.txt" \
--compute-validation-loss-global-model \
--log-level "ERROR" \
--learning-rate $LR \
--checkpoint-interval 18000 \
--checkpoint-interval-is-in-sec > output_${DATASET}_c${COHORTS}_s${SEED}_p${PARTICIPATION}_${CLUSTER_METHOD}.log 2>&1
