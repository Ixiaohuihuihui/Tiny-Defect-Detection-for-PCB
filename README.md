# PCB-defect-detection
## Note 
This project code is forked from https://github.com/DetectionTeamUCAS/FPN_Tensorflow. I have only made minor changes on this wonderful and clear project. Thanks for their perfect code. I can learn and apply it to a new problem.
##  PCB defect dataset
The Open Lab on Human Robot Interaction of Peking University has released the PCB defect dataset.

You can download at http://robotics.pkusz.edu.cn/resources/dataset/. 

And another student has published his paper "A PCB Dataset for Defects Detection and Classification" on arxiv. More datails about this dataset: https://arxiv.org/pdf/1901.08204.pdf. 

6 types of defects are made by photoshop, a graphics editor published by Adobe Systems. The defects defined in the dataset are: missing hole, mouse bite, open circuit, short, spur, and spurious copper. 
For example:

![1](a(missinghole).png)
## Download Model
Please download [resnet50_v1](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz)、[resnet101_v1](http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz) pre-trained models on Imagenet, put it to $PATH_ROOT/data/pretrained_weights. 
## My Development Environment
1、python2.7 (anaconda recommend)             
2、CUDA Version 8.0.44 , CUDNN=5.1.10           
3、[opencv(cv2)](https://pypi.org/project/opencv-python/)    
4、[tfplot](https://github.com/wookayin/tensorflow-plot)             
5、tensorflow == 1.121 
## Demo(available)

**Select a configuration file in the folder ($PATH_ROOT/libs/configs/) and copy its contents into cfgs.py, then download the corresponding [weights](ttps://pan.baidu.com/s/1rvHjihG1fL499SqU28Nang).**code：shac 

```   
cd $PATH_ROOT/tools
python inference.py --data_dir='/PATH/TO/IMAGES/' 
                    --save_dir='/PATH/TO/SAVE/RESULTS/' 
                    --GPU='0'
```

## Eval
```  
cd $PATH_ROOT/tools
python eval.py --eval_imgs='/PATH/TO/IMAGES/'  
               --annotation_dir='/PATH/TO/TEST/ANNOTATION/'
               --GPU='0'
```   
## Some results 
[the more results](https://github.com/Ixiaohuihuihui/PCB-defect-detection/tree/master/tools/inference_results)
![1](01_missing_hole_01.jpg)
![2](04_mouse_bite_10.jpg)
