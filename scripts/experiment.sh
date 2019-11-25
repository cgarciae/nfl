DATA_PATH="input/nfl-big-data-bowl-2020/train.csv"
PARAMS_PATH="scripts/params.yml"

python -m scripts.experiment \
    --data-path $DATA_PATH \
    --params_path $PARAMS_PATH \
    "$@"