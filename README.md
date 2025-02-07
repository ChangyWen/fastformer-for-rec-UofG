# Updated on 2025-Feb

## Requirements
```bash
Python==3.12.3
transformers==4.48.2
torch==2.3.0+cu121
scikit-learn==1.6.1
huggingface-hub==0.28.1
```

## Preparation for running on AutoDL
1. Prepare the environment
2. Set mirror of huggingface (https://hf-mirror.com/)
```bash
export HF_ENDPOINT=https://hf-mirror.com
```
3. Prepare the data
4. Run `train.py` and `submission.py` accordingly




# [Outdated] Fastformer for recommandation (Unofficial)

Edditied under the licence provided by the original creator for research purpouses.


This example walks through the training and prediction of unilm-fastformer on MIND dataset. <br>
Fastformer model refers to [fastformer](https://github.com/wuch15/Fastformer) <br>
PLM structure refers to [PLM4Rec](https://github.com/wuch15/PLM4NewsRec) and [speedy_mind](https://github.com/microsoft/SpeedyRec/tree/main/speedy_mind)<br>
After cloning this repo, you can conduct experiments with following commands:

## Results on MIND
| models   | AUC    | MRR    | nDCG@5 | nDCG@10 |
|----------|--------|--------|--------|---------|
| official | 0.7268 | 0.3745 | 0.4151 | 0.4684  |
| ours     | 0.7256 | 0.3720 | 0.4101 | 0.4660  |


## Requirements
```bash
Python==3.6
transformers==3.4.0
tensforlow==1.15
scikit-learn==0.23
```

## Preparing Data
Download data from MIND [link](https://msnews.github.io/) and decompress these files. You will get three files:
`MINDlarge_train`, `MINDlarge_dev`, and `MINDlarge_test`, then put them in the same folder, e.g., `./data/`.

Script `data_generation.py` can help you to generate the data files which meet the need of SpeedyRec:
```
python data_generation.py --raw_data_path {./data or other path you save the decompressed data}
```
The processed data will be saved to `./data/speedy_data/`.

## Training
```
python train.py \
--pretreained_model unilm \
--pretrained_model_path {path to ckpt of unilmv2} \
--root_data_dir ./data/speedy_data/ \
--num_hidden_layers 8 \
--world_size 4 \
--lr 1e-4 \
--pretrain_lr 8e-6 \
--warmup True \
--schedule_step 240000 \
--warmup_step 1000 \
--batch_size 42 \
--npratio 4 \
--beta_for_cache 0.002 \
--max_step_in_cache 2 \
--savename speedyrec_mind \
--news_dim 256
```
The model will be saved to `./saved_models/`, and validation will be conducted after each epoch.
The default pretrained model is UniLM v2, and you can get it from [unilm repo](https://github.com/microsoft/unilm). For other pretrained model, you need set `--pretrained_model==others` and give a new path for `--pretrained_model_path`
(like `roberta-base` and `microsoft/deberta-base`, which needs to be supported by [transformers](https://huggingface.co/transformers/model_doc/auto.html?highlight=automodel#transformers.AutoModel)).



## Prediction
Run prediction using saved checkpoint:
```
python submission.py \
--pretrained_model_path {path to ckpt of unilmv2} \
--pretreained_model unilm \
--root_data_dir ./data/speedy_data/ \
--num_hidden_layers 8 \
--load_ckpt_name {path to your saved model} \
--batch_size 256 \
--news_dim 256
```
It will creates a zip file:`predciton.zip`, which can be submitted to the leaderboard of MIND directly.

## Training and prediction based on our trained model
You can download the config and the model trained by us from this [link](https://rec.ustc.edu.cn/share/2d76b930-3955-11ed-af65-11758bdcd0e4) and save them in `./speedymind_ckpts`.
- Prediction
You can run the prediction by following command:
```
python submission.py \
--pretrained_model_path ./speedymind_ckpts \
--pretreained_model unilm \
--root_data_dir ./data/speedy_data/ \
--num_hidden_layers 8 \
--load_ckpt_name ./speedymind_ckpts/fastformer4rec.pt \
--batch_size 256 \
--news_attributes title  \
--news_dim 256
```

- Training
If you want to finetune the model, set `--pretrained_model_path ./speedymind_ckpts --news_attributes title` and run the training command. Here is an example:
```
python train.py \
--pretreained_model unilm \
--pretrained_model_path ./speedymind_ckpts \
--root_data_dir ./data/speedy_data/ \
--news_attributes title \
--num_hidden_layers 8 \
--world_size 4 \
--lr 1e-4 \
--pretrain_lr 8e-6 \
--warmup True \
--schedule_step 240000 \
--warmup_step 1000 \
--batch_size 42 \
--npratio 4 \
--beta_for_cache 0.002 \
--max_step_in_cache 2 \
--savename speedyrec_mind \
--news_dim 256
```