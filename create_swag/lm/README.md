# LM

Contains hopefully everything you need to run the LM

# Setup

0. Update the config file with where your pretraining text is.

1. Create the vocabulary by running ```python load_data.py``` or copy things around manually, then do the pretraining using `pretrain_lm.py`. Or, you can access my pretrained checkpoint [here](https://drive.google.com/file/d/1Ik7cbGs-wbAKKCeuYA8Uhe5O3pHJwHcj/view?usp=sharing)
    
2. To finetune on activitynet captions and LSMDC do
    
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

One of my checkpoints (for fold 0) is [here](https://drive.google.com/file/d/1J9QPJTIOIDR4V_zGB8ejilWAXXkxrogC/view?usp=sharing)

3. Pick the best checkpoints, then go generate stuff!!
