# FocusCut
The official PyTorch implementation of oral paper ["FocusCut: Diving into a Focus View in Interactive Segmentation"](https://openaccess.thecvf.com/content/CVPR2022/papers/Lin_FocusCut_Diving_Into_a_Focus_View_in_Interactive_Segmentation_CVPR_2022_paper.pdf) in CVPR2022.

## Prepare
See requirements.txt for the environment.
```shell
pip3 install -r requirements.txt
```
Put the pretrained models into the folder "pretrained_model" and the unzipped datasets into the folder "dataset".

### Train:
```shell
CUDA_VISIBLE_DEVICES=0 python main.py -rf -ap "backbone='resnet50'"
```

### Evalution:
```shell
CUDA_VISIBLE_DEVICES=0 python main.py -v -r focuscut-resnet50.pth -ap "backbone='resnet50'" -hrv -dv GrabCut,Berkeley,DAVIS 
```
### Demo with UI: 
```shell
CUDA_VISIBLE_DEVICES=0 python annotator.py -r focuscut-resnet50.pth -ap "backbone='resnet50'" -hrv  -img test.jpg
```

## Datasets
*(These datasets are organized into a unified format, our Interactive Segmentation Format (ISF)*
- GrabCut ( [GoogleDrive](https://drive.google.com/file/d/1CKzgFbk0guEBpewgpMUaWrM_-KSVSUyg/view?usp=sharing) | [BaiduYun](https://pan.baidu.com/s/1Sc3vcHrocYQr9PCvti1Heg?pwd=2hi9) )
- Berkeley ([GoogleDrive](https://drive.google.com/file/d/16GD6Ko3IohX8OsSHvemKG8zqY07TIm_i/view?usp=sharing) | [BaiduYun](https://pan.baidu.com/s/16kAidalC5UWy9payMvlTRA?pwd=4w5g)  )
- DAVIS  ([GoogleDrive](https://drive.google.com/file/d/1-ZOxk3AJXb4XYIW-7w1-AXtB9c8b3lvi/view?usp=sharing) | [BaiduYun](https://pan.baidu.com/s/1hXXxIfFhpaO8P0YqjQEnvQ?pwd=b5kh)  )
- SBD  ([GoogleDrive](https://drive.google.com/file/d/1trmUNY_qI151GiNS3Aqfkskb6kbpam3o/view?usp=sharing) | [BaiduYun](https://pan.baidu.com/s/1ik1pIWCwyKBDq6zsiA0iRA?pwd=t1gk) )

## Pretrained models
- focuscut-resnet50  ([GoogleDrive](https://drive.google.com/file/d/1cNt84bF7p8XYVuVudQlVaTgQyJszw_GC/view?usp=sharing) | [BaiduYun](https://pan.baidu.com/s/14L8AHh4S1bQsxujfNv1xVA?pwd=jjmu) )
- focuscut-resnet101  ([GoogleDrive](https://drive.google.com/file/d/1tmetXPTWnakghDHhm0uToQUsz2cZIfEL/view?usp=sharing) | [BaiduYun](https://pan.baidu.com/s/1MIE4YhdEmYLxbJya6HqmPQ?pwd=ebmv ) )


## Citation
If you find this work or code is helpful in your research, please cite:
```
@inproceedings{lin2022focuscut,
  title={FocusCut: Diving into a Focus View in Interactive Segmentation},
  author={Lin, Zheng and Duan, Zheng-Peng and Zhang, Zhao and Guo, Chun-Le and Cheng, Ming-Ming},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2637--2646},
  year={2022}
}
```
## Contact
If you have any questions, feel free to contact me via: `frazer.linzheng(at)gmail.com`.  
Welcome to visit [the project page](http://mmcheng.net/focuscut/) or [my home page](https://www.lin-zheng.com/).

## License
Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for Non-commercial use only.
Any commercial use should get formal permission first.