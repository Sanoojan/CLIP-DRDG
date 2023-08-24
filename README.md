# Exploring the Transfer Learning Capabilities of CLIP on Domain Generalization for Diabetic Retinopathy

**CoOpLVT: Context Optimization with Learnable Visual Tokens**

![Method](images/AI702_diagram_final.png)

## Abstract

Domain generalization is a challenging problem in machine learning that aims to build models capable of performing well on unseen target domains by leveraging knowledge from multiple source domains. To achieve this, researchers are exploring various techniques such as domain adaptation, multi-task learning, and regularization, while also leveraging large-scale datasets and pre-training on diverse data sources. In the medical domain, domain generalization is particularly important because of the challenges posed by collecting labeled data in different domains, such as patient demographics, imaging modalities, or disease stages. The state-of-the-art model Contrastive Language-Image Pre-Training (CLIP) [31] can aid in resolving the domain generalization issue. CLIP was pre-trained on a large and diverse dataset of images and their corresponding textual descriptions, which enables it to learn rich and general representations that can be utilized for a wide range of downstream tasks and has the potential to generalize well to new domains and tasks that it has not been explicitly trained on, especially those of natural images. In this work, we explore the transfer learning capabilities of CLIP with prevalent fine-tuning approaches, especially in the medical domain focusing on the problem of Domain Generalization for Diabetic Retinopathy (DR). We further propose a multi-modal fine-tuning strategy with context optimization by conditioning on visual features which we coin as Context Optimization with Learnable Visual Tokens (CoOpLVT). With our proposed approach, we improve the F1-score by 1.8% from the baseline. We conduct an extensive set of experiments to understand and analyze the capabilities of CLIP and its effectiveness in solving the DR problem.

## Results

![F1-Score](images/f1-score.png)

## Install Dependencies

Run the following command to install the required conda environment and dependencies:
```bash
conda env create --file=environment.yml
```

## Download Dataset

The dataset can be accessed in G42 cluster, more specifically in `/nfs/users/ext_group8/Dataset/224_data/DR`.

## Download Pre-trained Weights

The pre-trained weights can be accessed [here](https://drive.google.com/drive/folders/1w9gG3clV3ZlmhIT88n0QFNM29_rOou8Y?usp=sharing).

## CoOpLVT Algorithm Implementation

We implement our proposed **CoOpLVT** algorithm as a class in `domainbed/algorithms.py`:

```python
class Clip_train_prompt_from_image_v2(Algorithm):
    """
    CoOpLVT: Context Optimization with Learnable Visual Tokens
    """
    
    def __init__(self, input_shape, num_classes, num_domains, hparams, weights_for_balance):
       ...
```

## How To Use

There are two important scripts to run the training and evaluate the results: `run_train.sh` & `run_evaluation.sh`. By default, we run our experiments for 3 trials and then report the mean and standard deviation of the results. `run_train.sh` will perform a training process and `run_evaluation.sh` will give accuracy and f1-score from the experiment. In addition, we provide a sample output folder (`COOPLVT_TRAINING_LOGS`). In the actual output folder, best models are monitored and saved.

`run_train.sh`
``` bash
#!/bin/bash

nvidia-smi

for lr in  0.000005 
do
    for dataset in DR
    do
        for init in clip_full
        do
            for command in delete_incomplete launch
            do
                CUDA_VISIBLE_DEVICES=1,2,3,4,8,15 python -m domainbed.scripts.sweep $command\
                    --data_dir=/nfs/users/ext_group8/Dataset/224_data/ \
                    --output_dir=COOPLVT_TRAINING_LOGS \
                    --command_launcher multi_gpu\
                    --algorithms Clip_train_prompt_from_image_v2 \
                    --single_test_envs \
                    --datasets ${dataset} \
                    --n_hparams 1  \
                    --n_trials 3 \
                    --hparams """{\"weight_init\":\"${init}\",\"backbone\":\"ClipBase\",\"lr\":${lr}}"""\
                    --skip_confirmation
            done > Outs/V23_CLIP_COOP_3_LAYERS_MLP.out
        done
    done
done

# CUDA_VISIBLE_DEVICES --> Denotes the GPUs indexes that we use to run the experiment.
# --data_dir           --> Dataset path.
# --output_dir         --> The path where the experiment outputs are saved in.
# --algorithms         --> The algorithm class that we want to use. See domainbed/algorithms.py to find algorithm variants. CoOpLVT is implemented as Clip_train_prompt_from_image_v2 class.
# --n_trials           --> Denotes how many trials that we want to run the experiment. By default, we set n_trials as 3 to alleviate randomness during training, allowing us to better interprete our experiments.
# Outs/V23_CLIP_COOP_3_LAYERS_MLP.out --> If we want to store terminal outputs.
```

`run_evaluation.sh`
``` bash
python -m domainbed.scripts.collect_results --input_dir COOPLVT_TRAINING_LOGS

# --input_dir --> The path where the experiment outputs are saved in.

# Sample output

# -------- Dataset: DR, model selection method: training-domain validation set
# Algorithm             aptos                 eyepacs               messidor              messidor_2            Avg                  
# Clip_train_prompt_fr  46.2 +/- 4.4          65.9 +/- 2.0          65.5 +/- 0.4          70.6 +/- 0.6          62.1                 

# -------- Averages, model selection method: training-domain validation set
# Algorithm                  DR                         Avg                       
# Clip_train_prompt_from_im  62.1 +/- 1.4               62.1                      

# -------- Dataset: DR, model selection method: training-domain validation set
# Algorithm             aptos                 eyepacs               messidor              messidor_2            Avg                  
# Clip_train_prompt_fr  31.9 +/- 2.6          32.2 +/- 0.3          46.2 +/- 0.7          51.6 +/- 0.3          40.5                 

# -------- Averages, model selection method: training-domain validation set
# Algorithm                  DR                         Avg                       
# Clip_train_prompt_from_im  40.5 +/- 0.5               40.5
```