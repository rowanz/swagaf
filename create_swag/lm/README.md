# LM

Contains hopefully everything you need to run the LM

# Setup

0. Update the config file with where your pretraining text is.

0. Create the vocabulary by running ```python load_data.py``` or copy things around manually. These commands will help you copy stuff to a different server
    ```
    scp -r lm-0-of-5.pkl lm-1-of-5.pkl lm-2-of-5.pkl lm-3-of-5.pkl lm-4-of-5.pkl vocabulary  rowan@magellanic:~/code/swagaf/create_swag/lm/
    ```
    
    and in pretrain
    ```
    scp e1-tbooks-pretrained-ckpt-370000.tar rowan@magellanic:~/code/swagaf/create_swag/lm
    ```
    
1. RUN
    
    ```
    export PYTHONPATH=/home/rowan/code/swagaf
    export CUDA_VISIBLE_DEVICES=0
    nohup python train_lm.py -fold 0 > fold_0_log.txt &
    export CUDA_VISIBLE_DEVICES=1
    nohup python train_lm.py -fold 1 > fold_1_log.txt &
    export CUDA_VISIBLE_DEVICES=2
    nohup python train_lm.py -fold 2 > fold_2_log.txt &
    ```
    
    And accordingly on the other machine
    ```
    export CUDA_VISIBLE_DEVICES=0
    nohup python train_lm.py -fold 3 > fold_3_log.txt &
    export CUDA_VISIBLE_DEVICES=1
    nohup python train_lm.py -fold 4 > fold_4_log.txt &
    ```

2. Pick the best checkpoints, then go generate stuff!!
