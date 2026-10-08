# Small and Distant Face Detection with IoU-Aware RetinaFace

Trained and evaluated an IoU-aware RetinaFace for small, distant faces on 4 V100 GPUs (team project), improving AP on the medium and hard WIDER FACE sets for both backbones.

Face detection for small, distant faces (roughly 20 to 50 pixels tall), as in surveillance and crowd footage. We add an IoU prediction head to RetinaFace so the detector learns how well each predicted box is localized.

Course team project (Computer Vision), University of Houston.
Team: Dipayan Biswas and Gunawardhana Bhasura.

## What we changed
- Added a parallel IoU prediction head (sigmoid output) next to the box regression branch.
- Added an IoU loss (binary cross-entropy between predicted and actual IoU) to RetinaFace's multi-task loss.
- Also tested a mean squared error IoU loss for ResNet50.
- Motivation: detections with high confidence but poor localization can suppress better boxes during NMS, which lowers AP at high IoU thresholds.

## Training setup
- Dataset: WIDER FACE (train split); evaluated on the validation set (easy, medium, hard)
- Backbones: ResNet50 and MobileNet0.25
- SGD (momentum 0.9, weight decay 5e-4), batch size 24, 100 epochs, 4 NVIDIA V100 GPUs
- Training time: about 15 hours (ResNet50) and 11 hours (MobileNet0.25); the IoU head adds about 20 minutes

## Results (WIDER FACE validation AP)

| Backbone | Model | Easy | Medium | Hard |
|---|---|---|---|---|
| ResNet50 | Base | 95.20 | 93.75 | 83.77 |
| ResNet50 | IoU-aware (ours) | 95.14 | 93.92 | 83.88 |
| MobileNet0.25 | Base | 88.70 | 85.66 | 68.35 |
| MobileNet0.25 | IoU-aware (ours) | 88.71 | 86.23 | 68.83 |

Gains appear on the medium and hard sets, which contain the smaller faces; easy-set AP is essentially unchanged.

## Reproducing our results

1. Set up the Python environment from `requirements.txt`.
2. Download the pretrained models from this [Google Drive folder](https://drive.google.com/drive/folders/1JltX6UmexEq7gA02n0jPlXCcOSl5sxNi?usp=sharing). The same folder also holds the PR curve data for each model (`.pkl` files).
3. Arrange the weights as follows:
```Shell
  ./weights/Pretrained-models/
    Pretrained_baseModel_MobileNet.pth
    Pretrained_baseModel_ResNet50.pth

    Pretrained_ourModel_MobileNet.pth
    Pretrained_ourModel_ResNet50.pth
    Pretrained_ourModel_ResNet50_mse.pth
```
4. Place the WIDER FACE dataset as described in the [original README's data section](#data) below.
5. Generate the result txt files and save the PR curve data (`.pickle`) in the pretrained model directory:
```Shell
python test_widerface.py --trained_model './weights/Pretrained-models/Pretrained_ourModel_ResNet50.pth' --save_folder './widerface_evaluate/txt_files/'
```
   - `--trained_model`: path to the pretrained model
   - `--save_folder`: directory for the txt result files
6. Plot the PR curves (saved to `outputs/`):
```Shell
python plot_results.py --network "resnet50" --IoU_lossFunction "bce"
```
   - `--network`: backbone, `"mobile0.25"` or `"resnet50"`
   - `--IoU_lossFunction`: IoU loss, `"bce"` or `"mse"` (`"mse"` is available for `"resnet50"` only)

## Credits
Built on [biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface), a PyTorch implementation of RetinaFace (Deng et al., CVPR 2020). The IoU-aware idea follows Wu et al., "IoU-aware single-stage object detector for accurate localization," Image and Vision Computing, 2020.

---

# From the original README (biubug6/Pytorch_Retinaface)

Condensed to the parts this project needs. See the [upstream repository](https://github.com/biubug6/Pytorch_Retinaface) for the full version.

A [PyTorch](https://pytorch.org/) implementation of [RetinaFace: Single-stage Dense Face Localisation in the Wild](https://arxiv.org/abs/1905.00641), with MobileNet0.25 (1.7M parameters) and ResNet50 backbones. Requires Python 3, PyTorch 1.1.0+ and torchvision 0.3.0+.

### Data
1. Download the [WIDERFACE](http://shuoyang1213.me/WIDERFACE/WiderFace_Results.html) dataset.

2. Download annotations (face bounding boxes & five facial landmarks) from [baidu cloud](https://pan.baidu.com/s/1Laby0EctfuJGgGMgRRgykA) or [dropbox](https://www.dropbox.com/s/7j70r3eeepe4r2g/retinaface_gt_v1.1.zip?dl=0)

3. Organise the dataset directory as follows:

```Shell
  ./data/widerface/
    train/
      images/
      label.txt
    val/
      images/
      wider_val.txt
```
ps: wider_val.txt only include val file names but not label information.

An already organized copy of the dataset is available from [google cloud](https://drive.google.com/open?id=11UGV3nbVv1x9IC--_tK3Uxf7hA6rlbsS) or [baidu cloud](https://pan.baidu.com/s/1jIp9t30oYivrAvrgUgIoLQ) (password: ruck).

### Training
The ImageNet-pretrained MobileNet0.25 backbone and the original trained models are on [google cloud](https://drive.google.com/open?id=1oZRSG0ZegbVkVwUd8wUIQx8W7yfZ_ki1) and [baidu cloud](https://pan.baidu.com/s/12h97Fy1RYuqMMIV-RpzdPg) (password: fstq). Place them as follows:
```Shell
  ./weights/
      mobilenet0.25_Final.pth
      mobilenetV1X0.25_pretrain.tar
      Resnet50_Final.pth
```
1. Before training, you can check network configuration (e.g. batch_size, min_sizes and steps etc..) in ``data/config.py and train.py``.

2. Train the model using WIDER FACE:
  ```Shell
  CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --network resnet50 or
  CUDA_VISIBLE_DEVICES=0 python train.py --network mobile0.25
  ```

### Evaluation (WIDER FACE val)
1. Generate txt file
```Shell
python test_widerface.py --trained_model weight_file --network mobile0.25 or resnet50
```
2. Evaluate txt results. Demo come from [Here](https://github.com/wondervictor/WiderFace-Evaluation)
```Shell
cd ./widerface_evaluate
python setup.py build_ext --inplace
python evaluation.py
```

### References
- [Retinaface (mxnet)](https://github.com/deepinsight/insightface/tree/master/RetinaFace)
- [FaceBoxes](https://github.com/zisianw/FaceBoxes.PyTorch)
```
@inproceedings{deng2019retinaface,
title={RetinaFace: Single-stage Dense Face Localisation in the Wild},
author={Deng, Jiankang and Guo, Jia and Yuxiang, Zhou and Jinke Yu and Irene Kotsia and Zafeiriou, Stefanos},
booktitle={arxiv},
year={2019}
}
```
