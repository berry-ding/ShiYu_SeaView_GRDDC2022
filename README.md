# ShiYu_SeaView_CRDDC2022
# Installation
conda create -n crddc2022 python=3.8  
conda activate crddc2022  
git clone https://github.com/berry-ding/ShiYu_SeaView_GRDDC2022.git  
cd ShiYu_SeaView_GRDDC2022  
#cuda >= 11.1  
pip3 install torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio===0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html  
#official yolov5 r6.1  
pip install -r yolov5/requirements.txt  
#official yolov7  
pip install -r yolov7/requirements.txt  
#mmdetection v2.25.0  
pip install openmim    
mim install mmcv-full==1.6.0
mim install mmcls  
cd mmdetection  
pip install -r requirements/build.txt  
pip install -v -e .  
# Quick Start
following ipynb  
# Model
| model | pretrained weight (google drive) | 
| :-----| :---- | 
| YOLOv5x_640 | [YOLOv5x_640] | 
| YOLOv5x6_1280 | [YOLOv5x6_1280] | 
| YOLOv7x_640 | [YOLOv7x_640] | 
| Faster_Swin_l_w7_ms_1and2 | [Faster_Swin_l_w7_ms_1and2] | 
| Faster_Swin_l_w12_DeformRoI_ms_1and2 | [Faster_Swin_l_w12_DeformRoI_ms_1and2] | 
| Faster_Swin_l_w12_DeformRoI_ms_3 |[Faster_Swin_l_w12_DeformRoI_ms_3] | 
| YOLOv5x_1600 | [YOLOv5x_1600] | 
| YOLOv5m_3200 | [YOLOv5m_3200] | 

# Acknowledgement
- https://github.com/ultralytics/yolov5
- https://github.com/WongKinYiu/yolov7
- https://github.com/open-mmlab/mmdetection
- 


[YOLOv5x_640]: https://drive.google.com/file/d/1nwUIbd_eYiOSLU1hSnwIsqF5HTFXvq02/view?usp=sharing
[YOLOv5x6_1280]: https://drive.google.com/file/d/1x97py7w7ruKAE5p6ZrSgDi_tmSZ73e6G/view?usp=sharing
[YOLOv7x_640]: https://drive.google.com/file/d/1DGfQivLLGR-uP3INUZv6P77VrOa_rtNt/view?usp=sharing
[Faster_Swin_l_w7_ms_1and2]: https://drive.google.com/file/d/11S-B7JEq_uALWKYJTsEmVt5v8IvW8VBF/view?usp=sharing
[Faster_Swin_l_w12_DeformRoI_ms_1and2]: https://drive.google.com/file/d/1DG5yor2pIKG2UKquQ0FZHkV2tBkPI7_k/view?usp=sharing
[Faster_Swin_l_w12_DeformRoI_ms_3]: https://drive.google.com/file/d/1B9FPVqvhPGbXgUdS2eRKIUP2M5zhnhES/view?usp=sharing
[YOLOv5x_1600]: https://drive.google.com/file/d/1yg_Sy2Z8rLKrl7LqcwovCeIzsOI3ygt7/view?usp=sharing
[YOLOv5m_3200]: https://drive.google.com/file/d/1BK8K0PwfkBDun36Js3iFUwd_z6HsgosG/view?usp=sharing
