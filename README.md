# Thesis Title: Computer Vision-based Advanced Driver Assistance System (ADAS)
Author: [Hikmatullah Mohammadi](https://github.com/hikmatullah-mohammadi)<br>
*This is my bachelor's thesis; it encompasses three main features: lane detection, road segmentation, and a simple Forward Collision Warning (FCW) system.*

Below is the overall flow of the project, a computer vision-based ADAS <br>
![Vision-based ADAS](https://github.com/hikmatullah-mohammadi/vision-based-adas/blob/main/assets/vision-based-adas.png?raw=true)

## Table of Contents
- [1. Abstract](#1-abstract)
- [2. Problem Statement](#2-problem-statement)
- [3. Research Question](#3-research-question)
- [4. Proposed Solution: A Vision-based ADAS](#4-proposed-solution-a-vision-based-ADAS)
- [5. Components of the ADAS](#5-Components-of-the-ADAS)
- [6. Folders Structure](#6-Folders-Structure)
- [7. How to Use It](#7-how-to-use-it)
- [8. Limitations and Challenges](#8-Limitations-and-Challenges)
- [9. Future Works](#9-future-works)
- [10. Conclusion](#10-conclusion)
- [11. Acknowledgment](#11-Acknowledgment)

## 1. Abstract
Road traffic accidents remain a significant threat to human safety, necessitating innovative approaches to enhance road safety. This thesis presents the development of a comprehensive Computer Vision-based Advanced Driver Assistance System (ADAS) that leverages deep learning techniques and computer vision algorithms. The primary goal of the vision-based ADAS is to augment the driver's capabilities by detecting and interpreting relevant visual cues from the surrounding environment. The proposed ADAS integrates lane detection, road segmentation, and Forward Collision Warning (FCW) components to provide real-time assistance and enhance road safety.

## 2. Problem Statement
Driving on roads can be dangerous due to factors such as poor visibility, unexpected road conditions, and human errors. Road traffic accidents continue to pose a significant threat to human safety worldwide, despite advancements in automotive technology. Human errors remain a major contributing factor to these accidents, emphasizing the urgent need for innovative approaches to enhance road safety.

## 3. Research Question
How can computer vision and deep learning techniques be leveraged to assist car drivers in detecting lane lines, other vehicles, pedestrians, traffic signs, and the distance from cars in front of them using road video footage?

## 4. Proposed Solution: A Vision-based ADAS
This project aims to develop an ADAS that leverages cutting-edge computer vision technology to enhance road safety. The proposed ADAS identifies and alerts drivers of potential hazards, including traffic signs, rear-view blind spots, and lane departures, thereby reducing the risk of accidents caused by human errors. By providing timely and accurate assistance to drivers, this system has the potential to significantly minimize injuries and fatalities on the roads.

## 5. Components of the ADAS

### 1. Lane Detection:
In this project, lane detection was accomplished by assigning each pixel in an image to either the lane or the background. This pixel-level classification was achieved using deep learning models such as Fully Convolutional Networks (FCNs) and U-Net architecture, which effectively segmented and differentiated lane markings from the surrounding environment, enabling precise lane identification and analysis.
#### Developed lane detection models
- Dataset: The models were trained on [the TuSimple Dataset](https://github.com/TuSimple/tusimple-benchmark)
- Images were downscaled from 720x1280 to 256x320 to accommodate memory constraints
- Model Architectures: Convolutional Neural Networks (CNNs): training two distinct lane detection models:
  - Fully Convolutional Network (FCN) [ResNet50 as the encoder]
    - Accuracy: 97.84%
    - Precision: 76.61%
    - Recall: 72.48%
    - F1-Score: 74.49%
    - IoU: 59.35%
    - Model architecture below
      - <img alt="Lane Detection Model architecture 1"  width="500" src="https://github.com/hikmatullah-mohammadi/vision-based-adas/blob/main/assets/lane-detection/fcn-model-short.png?raw=true"/>
  - U-Net architecture
    - Accuracy: 97.90%
    - Precision: 76.61%
    - Recall: 74.56%
    - F1-Score: 75.57%
    - IoU: 60.74%
    - Model architecture below
      - <img alt="Lane Detection Model architecture 2"  width="500" src="https://github.com/hikmatullah-mohammadi/vision-based-adas/blob/main/assets/lane-detection/u-net-architecture.png?raw=true"/>
- Results <br>
![Lane Detection models results](https://github.com/hikmatullah-mohammadi/vision-based-adas/blob/main/assets/screenshots/ld-scshot.png?raw=true)

### 2. Road Image Segmentation:
Road segmentation involves assigning each pixel in an image to specific subparts corresponding to objects of interest on the road. This is achieved using U-Net with skip connections to effectively classify pixels and differentiate between road elements and the background, enabling comprehensive analysis and enhancing road safety. In our case, there are 19 distinct objects annotated. 
#### Developed road segmentation model
- Dataset: The model was trained on [the BDD100k dataset](https://github.com/bdd100k/bdd100k), which is a comprehensive collection of diverse road images.
- Model Architecture: U-Net-style architecture with skip-connections to capture both local and global features
  - <img alt="Road Segmentation Model architecture"  width="500" src="https://github.com/hikmatullah-mohammadi/vision-based-adas/blob/main/assets/road-segmentation/road-segmentation-model-architecture.png?raw=true"/>
      
- Training Process:
  - Training was performed on a CPU due to the limited available memory.
  - Images were downscaled from 720x1280 to 192x256 to accommodate memory constraints.
  - Training spanned 66 epochs and took several days to complete.
- Evaluation Metrics:
  - Overall Pixel Accuracy: 85.79%
  - Mean IoU (Intersection over Union): 36.67%
  - mAP (mean Average Precision): 51.52%
- Performance and Challenges:
  - Despite the adverse effects of downsizing on performance and generalization capabilities, the model yielded satisfactory results.
  - The model demonstrated commendable performance considering the challenges posed by memory constraints and CPU-based training.
- Application: The road segmentation model was utilized to construct a simple yet effective Forward Collision Warning (FCW) system
- Results <br>
![Road image segmentation model results](https://github.com/hikmatullah-mohammadi/vision-based-adas/blob/main/assets/screenshots/rs-scshot.png?raw=true)

### 3. Forward Collision Warning (FCW) System:
This feature implements a simple FCW system using computer vision techniques. The road image is first divided into different categories using the image segmentation model. Then, employing image processing techniques, the system detects the presence of additional objects near the equipped vehicle on the road, issuing warnings to the driver when necessary.
#### Methodology
- The Region of Interest (RoI):
  - A rectangular RoI measuring 216 pixels in width and 60 pixels in height is selected in front of the equipped vehicle.
  - The system continuously monitors this RoI for the presence of additional objects.
- Object Detection in RoI:
  - The segmentation mask generated by the image segmentation model detects extra objects in the designated RoI.
  - Each object in the mask has a unique pixel value, allowing easy identification.
- Visual warning of the presence of extra objects in the RoI
  - Pixels in the RoI with a value other than the gray road pixel value represent other objects.
  - A binary mask is created, where 1s represent other objects and 0s represent road pixels.
  - Pixels representing extra objects are converted to red, visually warning the driver of their presence in the RoI.
- Results <br>
![Road image segmentation model results](https://github.com/hikmatullah-mohammadi/vision-based-adas/blob/main/assets/screenshots/fcw-scshot.png?raw=true)

In summary, the FCW system detects and visually alerts the driver to the occurrence of extra objects within the specified RoI. By converting the corresponding pixels to red, potential accidents can be identified and prevented.

### Combining all the components:
We discussed the development and properties of the three most crucial components of the ADAS that we aim to build: the road lane detection model, the road segmentation model, and a simple FCW system. In this phase, we combine all three components mentioned above to build a vision-based ADAS.

As we talked about in the introduction section, the captured pictures of the road will first be fed into the road lane detection model we built to detect the lane markings. After generating the road lane markings' mask, the picture is then passed through the road segmentation model. The road segmentation model divides the different categories of objects present on the road and generates a mask, assigning distinct pixel values to different objects. Finally, the FCW system is applied to the segmentation mask produced by the road segmentation model. This system recognizes extra objects in the determined Region of Interest (RoI) and highlights them in red, warning the driver of any potential hazards.
- Here is a sample result of the ADAS in action.<br>
![Road image segmentation model results](https://github.com/hikmatullah-mohammadi/vision-based-adas/blob/main/assets/screenshots/adas-scshot.png?raw=true)

Overall, using this ADAS makes it way more convenient for the driver to observe the vehicle's surroundings and take action accordingly.

## 6. Folders Structure
```
- assets
  |-- FCW-images
  |-- lane-detection
  |-- road-segmentation
  |-- screenshots
- models
- notebooks
  |-- lane-detection
  |-- road-segmentation
- utils_scripts
```
**Explanations:**
  - `assets`: This folder stores various assets related to the project. It contains three subdirectories:
    - `FCW-images`: This subdirectory contains images related to Forward Collision Warning (FCW) functionality.
    - `lane-detection`: This subdirectory contains images or data related to lane detection.
    - `road-segmentation`: This subdirectory contains images or data related to road segmentation.
    - `screenshots`: This subdirectory contains screenshots of different components of the project
  - `models`: This folder stores machine learning models trained in the project
  - `notebooks`: This folder contains Jupyter notebooks or other code files used for experimentation, analysis, or development. It has two subdirectories:
    - `lane-detection`: This subdirectory contains notebooks related to lane detection tasks.
    - `road-segmentation`: This subdirectory contains notebooks related to road segmentation tasks.
  - `utils_scripts`: This folder contains utility scripts or helper functions used throughout the project. These scripts provide common functionalities or reusable code snippets to facilitate development, data preprocessing, or other tasks in the project.

## 7. How to Use It
To run the project locally, follow the following steps.
1. Clone the repository.
  ```
  git@github.com:hikmatullah-mohammadi/vision-based-adas.git
  ```
2. See how the components work individually.
  - Lane Detection
    ```
    python utils_scripts/detect_lane_markings.py
    ```
  - Road Image Segmentation
    ```
    python utils_scripts/segment_road.py
    ```
  - Forward Collision Warning (FWC) system
    ```
    python utils_scripts/forward_collision_warning_system.py
    ```
3. See how the components work as a whole.
  - The ADAS as a whole
    ```
    python index-developed_adas.py
    ```    
4. Run on custom images
- All these files: `utils_scripts/detect_lane_markings.py`, `utils_scripts/segment_road.py`, `utils_scripts/forward_collision_warning_system.py`, and `index-developed_adas.py` contains classes that are responsible for performing the corresponding tasks. Simply import them, create instances, and use them accordingly. Everything you need to know about the arguments of the class, for example, is provided within the files mentioned above.

## 8. Limitations and Challenges
The ADAS system faced the following limitations and challenges:
- Limited computational resources: Training on reduced datasets due to resource constraints impacted performance. Access to more powerful devices and GPUs would improve training on high-resolution images and the entire dataset.
- Reduced image sizes: Downscaling the image sizes by over one-fourth adversely affected the models' performance.
- Reduced dataset size: Training on smaller chunks of the datasets weakened the models' ability to generalize.
- Inability to use GANs for segmentation: Due to computational limitations, GANs couldn't be utilized for road image segmentation. Exploring GANs could improve system performance compared to traditional techniques.
- Training time constraints: Training on CPUs with the entire dataset was time-consuming, limiting hyperparameter and architecture exploration. Access to GPUs would enable thorough experimentation.
Addressing these limitations and pursuing future works would enhance the ADAS system's capabilities for lane detection, road segmentation, and road safety features.

## 9. Future Works
In the next versions of the system, we plan to address its limitations and flaws and introduce new features and improvements. Here are the key future works:

- **Enhanced Training:** We aim to improve the system's performance by training on high-resolution images and the complete dataset from TuSimple and BDD100K, subject to sufficient computational resources.
- **Advanced Features:** We will add sophisticated features to enhance road safety, including:
  - Inappropriate lane departure warning system
  - Blind spot vision
  - Driver drowsiness detection
- **Warning Strategies:** We will explore different warning strategies, such as visual and aural cues, to effectively alert drivers in critical situations. This may involve visual indicators or LEDs and aural notifications.
- **Improved FCW System:** The Forward Collision Warning (FCW) system will be enhanced by:
  - Improving the segmentation model's performance
  - Investigating new approaches to accurately measure distances between vehicles
  - Integrating distance measures to create a more robust and reliable FCW system

By addressing these future works, we aim to improve the overall performance, functionality, and road safety features of the ADAS system.

## 10. Conclusion
In conclusion, the integration of lane detection, road segmentation, and FCW components results in a comprehensive vision-based Advanced Driver Assistance System (ADAS). This system utilizes deep learning techniques and computer vision algorithms to improve road safety and assist drivers in real-time. The lane detection component accurately identifies and tracks lane markings, reducing the risk of unintended lane departures. The road segmentation component partitions the road scene, enabling a detailed understanding of the driving environment. The FCW system detects collision risks and alerts the driver through visual warnings. This vision-based ADAS enhances driver awareness and promotes safer driving. As we continue to refine and enhance these components, we anticipate a future where vision-based ADAS becomes a standard feature, ensuring safer roads for all.
The figure below shows the overall flow of the ADAS.<br>
![Vision-based ADAS](https://github.com/hikmatullah-mohammadi/vision-based-adas/blob/main/assets/vision-based-adas.png?raw=true)

## 11. Acknowledgment
I would like to express my heartfelt gratitude and acknowledge the following individuals and entities who have been instrumental in the completion of this thesis:
- First and foremost, I would like to thank God for providing me with the strength, determination, and blessings throughout this academic journey.
- To my loving family: Their unwavering support and encouragement have been my rock. Their belief in me and sacrifices made this achievement possible.
- I am deeply grateful to my supervisor, Asst. Prof. Haqmal, for his invaluable guidance, expertise, and mentorship. His unwavering dedication and insightful feedback have shaped me into a better researcher and scholar.
- To the faculty members of Kabul University, Faculty of Computer Science, Department of Information Systems: I am grateful to them for their commitment to education and for imparting valuable knowledge and skills that have contributed to the development of this thesis.
- To my friends, who have been by my side through thick and thin: Their constant support, encouragement, and camaraderie are highly appreciated. Their presence has made this academic journey enjoyable and memorable.
- I would like to acknowledge the wider computer science community for their groundbreaking research and contributions. Their work has been a source of inspiration and has guided me throughout this thesis. Lastly, I am grateful to all those who have played a role, big or small, in shaping my academic and personal growth. Their encouragement, assistance, and belief in my abilities have been invaluable.

*To all, I offer my heartfelt appreciation. Without these individuals’ support, this thesis would not have been possible.*

<hr>

I'd appreciate it if you could read the complete documentation [here](https://drive.google.com/file/d/1R7SMaYmE69UMcFL8GeP2wYLF316LQYuZ/view?usp=sharing) for more information, such as in-depth explanations of the approaches and methodologies.
