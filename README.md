# FaceImageKit

Python实现的人脸图像相关算法Pipeline

# License

本项目采用双重许可：

- 开源版：AGPL-3.0。开源项目、个人和学术用途可免费使用；衍生作品需按 AGPL-3.0 要求开源；闭源商业项目不能在未遵守 AGPL-3.0 开源义务的情况下使用。
- 商业版：付费授权。闭源商业项目可使用，可免除 AGPL-3.0 开源义务，并可按授权协议获得技术支持。

商业授权请联系：blake120386@163.com

# Install
## python
example:
```bash
uv venv --python 3.12
```
## deploy
### cpu
```shell
uv pip install -e .[cpu]
```
### cuda
```shell
uv pip install -e .[cuda]
```

## develop
### cpu
```shell
uv sync --extra cpu
```
### gpu/cuda
```shell
uv sync --extra cuda
```
### dev
```shell
uv sync --extra dev --extra cpu
```

# Models
## Face Dtection
### scrfd
Models accuracy on WiderFace benchmark:
| Model            | Easy  | Medium | Hard  |
| :--------------- | :---: | :----: | :---: |
| scrfd_10g_gnkps  | 95.51 | 94.12  | 82.14 |
| scrfd_2.5g_gnkps | 93.57 | 91.70  | 76.08 |
| scrfd_500m_gnkps | 88.70 | 86.11  | 63.57 |

来源https://github.com/SthPhoenix/InsightFace-REST/    

**runtime**
+ [x] onnxruntime(cpu, cuda)
+ [x] opencv
+ [ ] TensorRT


## Face Landmark
datasets: Lapa134(Lapa106 + 28)
### rtmface
模型输出大小: $134\times 2$

关键点说明图:

![rtmface 134 landmarks](asserts/rtmface_134.png)

| Model                 |  NME   |
| :-------------------- | :----: |
| rtmpose-m-ort-lapa134 | 0.0288 |
| rtmpose-s-ort-lapa134 | 0.0258 |
+ [x] onnxruntime(cpu, cuda)
+ [x] opencv
+ [ ] TensorRT

## Face Segmentation
### ppliteseg
项目来源：https://github.com/tfrbt/FaceSeg.git

- label map 类别数量32
  
| id   |      class      |
| :--- | :-------------: |
| 0    |  'background'   |
| 1    |     'skin'      |
| 2    |     'cheek'     |
| 3    |     'chin'      |
| 4    |      'ear'      |
| 5    |     'helix'     |
| 6    |    'lobule'     |
| 7    |  'bottom_lid'   |
| 8    |     'pupil'     |
| 9    |     'iris'      |
| 10   |    'sclera'     |
| 11   |   'tear_duct'   |
| 12   |    'top_lid'    |
| 13   |    'eyebrow'    |
| 14   |    'forhead'    |
| 15   |     'frown'     |
| 16   |     'hair'      |
| 17   |    'temple'     |
| 18   |      'jaw'      |
| 19   |     'beard'     |
| 20   | 'inferior_lip'  |
| 21   | 'oral comisure' |
| 22   | 'superior_lip'  |
| 23   |     'teeth'     |
| 24   |     'neck'      |
| 25   |     'nose'      |
| 26   |   'ala_nose'    |
| 27   |    'bridge'     |
| 28   |   'nose_tip'    |
| 29   |    'nostril'    |
| 30   |     'DU26'      |
| 31   |   'sideburns'   |

| Model                | val IOU | val Dice |
| :------------------- | :-----: | :------: |
| face_seg_ppliteseg_t |  0.727  |  0.834   |

+ [x] onnxruntime(cpu)
+ [ ] TensorRT

- label map 类别数量12

| id   |           class            |
| :--- | :------------------------: |
| 0    |   "background"  # 0.背景   |
| 1    |      "skin"  # 1.皮肤      |
| 2    |      "eye"  # 2.眼睛       |
| 3    |     "pupil"  # 3.瞳孔      |
| 4    |  "bottom_lid", # 4.下眼皮  |
| 5    |   "top_lid", # 5.上眼皮    |
| 6    |    "eyebrow"  # 6.眉毛     |
| 7    |      "hair"  # 7.头发      |
| 8    | "superior_lip"  # 8上嘴唇  |
| 9    |     "teeth"  # 9.牙齿      |
| 10   | "inferior_lip" # 10.下嘴唇 |
| 11   |      "nose" # 11.鼻子      |

| Model                 | val IOU | val Dice |
| :-------------------- | :-----: | :------: |
| face12_seg_pplitesegb |  0.774  |  0.869   |

+ [x] onnxruntime(cpu)
+ [ ] TensorRT
