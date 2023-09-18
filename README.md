# INST0062-22057703
The supplementary files (code) of the dissertation: Correlation Analysis of Evaluation Metrics in Image Captioning: An Empirical Study using the GRIT Model on COCO Dataset.

### Requirements
* Python >= 3.9, CUDA >= 11.3
* PyTorch >= 1.12.0, torchvision >= 0.6.1
* Other packages: pycocotools, tensorboard, tqdm, h5py, nltk, einops, hydra, spacy, and timm

* Install other requirements:
```shell
pip install -r requirements.txt
python -m spacy download en
```
* Install Deformable Attention:
```shell
cd models/ops/
python setup.py build develop
python test.py
```

### Data preparation
Extract the `val2014.zip` archive from the `coco_caption` folder.

### Training
```shell
export DATA_ROOT=path/to/coco_dataset
# with pretrained object detector on 4 datasets
python train_caption.py exp.name=caption_4ds model.detector.checkpoint=4ds_detector_path \
optimizer.freezing_xe_epochs=10 \
optimizer.freezing_sc_epochs=10 \
optimizer.finetune_xe_epochs=0 \
optimizer.finetune_sc_epochs=0 \
optimizer.freeze_backbone=True \
optimizer.freeze_detector=True
```

### Evaluation
```shell
python eval_caption.py +split='valid' exp.checkpoint=grit_checkpoint_4ds.pth
```
The purpose of this code is to sample the val dataset and then test it with a model and record statistics. The statistics are saved in `outputs/eval`.

The code for the correlation analysis is in `corr.ipynb` and the numerical results of the correlation coefficients are saved in `corr.csv`.
