GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-1235}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
PRECISION=${PRECISION:-bf16}

DATA_PATH=${DATA_PATH:-/lintaoLab/pengsun/data/cifar10/train}
CONFIG=${CONFIG:-idae_f1c3}
IMAGE_SIZE=${IMAGE_SIZE:-32}

accelerate launch \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $NODE_RANK \
    --num_processes  $(($GPUS_PER_NODE*$NNODES)) \
    --num_machines $NNODES \
    --mixed_precision $PRECISION \
    data.py \
    --image_size $IMAGE_SIZE \
    --data_path $DATA_PATH \
    --config $CONFIG \