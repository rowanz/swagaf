# generate_candidates

Stage 1 of the pipeline - generate a bunch of candidates.

Unfortunately, this is pretty slow, so we'll want to duplicate it to several GPUs.

The current pipeline as of now:

1. Generate the candidates on 5 different GPUs
    ```
    export PYTHONPATH=/home/rowan/code/commonsense
    export CUDA_VISIBLE_DEVICES=0
    nohup python sample_candidates.py -fold 0 > fold_0_log.txt &
    ```
2. Pretrain the assignments using the LM features. This also will split it up into 5 folds
    ```
    nohup python rebalance_dataset_mlp.py > mlp_log.txt &
    ```
3. Do the assignments using more sophisticated features
    ```
    export CUDA_VISIBLE_DEVICES=0
    nohup python rebalance_dataset_ensemble.py -fold -1 > rebalance_everything.txt &   
    ```
    
4. Use `questions2mturk.py` to come create a CSV for mturk.