
## Other Implementations of Distance Estimation
### [harshilpatel312/KITTI-distance-estimation](https://github.com/harshilpatel312/KITTI-distance-estimation/tree/master)
Takes in bouding box coordinates  and outputs an estimated distance

### Drone to Obstacle Distance Estimation Using YOLO V3 Network and Mathematical Principles
- Seems to only use YOLO to determine bounding boxes, then uses conventional math as shown on page 7 which gives the equation:
$$D=\frac{\text{Real height of Object(mm)}*\text{Camera Frame height(pixels)}*\text{f(mm)}}{\text{Image Height(pixels)}*\text{Sensor Height(mm)}}$$
- Not entirely sure though since this isn't quite enough info, maybe they are using the YOLO classifier and assigning a static size to every object classification.

### [Bounding Box Dataset Augmentation for Long-range Object Distance Estimation](https://openaccess.thecvf.com/content/ICCV2021W/ILDAV/papers/Franke_Bounding_Box_Dataset_Augmentation_for_Long-Range_Object_Distance_Estimation_ICCVW_2021_paper.pdf)
- Start with COCO-trained YOLO v3 model, cut off last 3 head layers
- Replace head layers with new random ones
- Retrain only head layer on dataset
- Feed output of retrained YOLO model into [DisNet](https://github.com/guanjianyu/DisNet)
- This is a similar method to the drone obstacle estimation except using an ANN for distance estimation now

Tools:
- [DisNet](https://project.inria.fr/ppniv18/files/2018/10/paper22.pdf): ANN built to estimate object distances from bounding boxes

### [Approximate Supervised Object Distance Estimation on Unmanned Surface Vehicles](https://arxiv.org/html/2501.05567v1)
- Modify YOLO models to also have an output for distance
- Explorers some distance normalization strategies for training

###  [anandasaia/NeuralDistance](https://github.com/anandasaia/NeuralDistance)
- Uses YOLOv3 with pre-trained weights for image detection
- Used the following set of features to train the 4 layer model (150 neurons per layer): ('xmin', 'ymin', 'xmax', 'ymax', 'xloc', 'yloc', 'observation angle')
- Seems to be somewhat incomplete as the README discription of the model features doesn't match the source code

### [Unsupervised Monocular Depth Estimation with Left-Right Consistency](https://ieeexplore.ieee.org/document/8100182)
- Differs from other approaches as it isn't object based.
- Takes in monocular image and estimates a full depth map
- [pdf from arxiv](https://arxiv.org/pdf/1609.03677)

## Datasets

### [KITTI](https://www.cvlibs.net/datasets/kitti/)
- Full annotated depth map training data: [Depth Evaluation](https://www.cvlibs.net/datasets/kitti/eval_depth_all.php)
