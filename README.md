# 🔥 OASNet: OASNet: Orthogonal Attention-Guided Spatial–Semantic Representation Learning Network for Infrared Small Target Detection [[📄 Paper Link]](https://ieeexplore.ieee.org/document/11219229)
### Xiaoyang Yuan, Chunling Yang, Yuze Li, Yan Zhang, IEEE Geoscience and Remote Sensing Letters 2025. 
![OASNet](https://github.com/Y-xiaoyang/MNHU-Net/blob/main/Structure.png)
# If the implementation of this repo is helpful to you, just star it！⭐⭐⭐

# Usage
### 1. Data
- [The SIRST dataset download dir [ACM]](https://github.com/YimianDai/sirst)
- [The NUDT-SIRST dataset download dir [DNANet]](https://pan.baidu.com/s/1WdA_yOHDnIiyj4C9SbW_Kg?pwd=nudt)
- [The IRSTD-1k dataset download dir [ISNet]](https://github.com/RuiZhang97/ISNet?tab=readme-ov-file)
#### Our project has the following structure:
```text
├──./dataset/
│    ├── IRSTD-1K
│    │    ├── images
│    │    │    ├── XDU0.png
│    │    │    ├── XDU1.png
│    │    │    ├── ...
│    │    ├── masks
│    │    │    ├── XDU0.png
│    │    │    ├── XDU1.png
│    │    │    ├── ...
│    │    ├── 80_20
│    │    │    ├── train.txt
│    │    │    ├── test.txt
│    ├── NUDT-SIRST
│    │    ├── images
│    │    │    ├── 000001.png
│    │    │    ├── 000002.png
│    │    │    ├── ...
│    │    ├── masks
│    │    │    ├── 000001.png
│    │    │    ├── 000002.png
│    │    │    ├── ...
│    │    ├── 80_20
│    │    │    ├── train.txt
│    │    │    ├── test.txt
│    ├── ...
│    ├── ...
│    ├── NUAA-SIRST
│    │    ├── images
│    │    │    ├── Misc_1.png
│    │    │    ├── Misc_2.png
│    │    │    ├── ...
│    │    ├── masks
│    │    │    ├── Misc_1.png
│    │    │    ├── Misc_2.png
│    │    │    ├── ...
│    │    ├── 80_20
│    │    │    ├── train.txt
│    │    │    ├── test.txt
```
### 2. Train.
```bash
python train_oas.py 
```
### 3. Test.
```bash
python test_oas.py 
```
* This code is highly borrowed from [AMFU](https://github.com/cwon789/AMFU-net). Thanks to Won Young Chung.
* This code is highly borrowed from [DNANet](https://github.com/YeRen123455/Infrared-Small-Target-Detection). Thanks to Boyang Li.

# Citation
If you find the code useful, please consider citing our paper using the following BibTeX entry.
```text
@ARTICLE{11219229,
  author={Yuan, Xiaoyang and Yang, Chunling and Li, Yuze and Zhang, Yan},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={OASNet: Orthogonal Attention-Guided Spatial–Semantic Representation Learning Network for Infrared Small Target Detection}, 
  year={2025},
  volume={22},
  number={},
  pages={1-5},
  doi={10.1109/LGRS.2025.3626394}}
```
# Contact
Welcome to raise issues or email to yuanxiaoyang1998@outlook.com for any question regarding our MNHU-Net.
 
