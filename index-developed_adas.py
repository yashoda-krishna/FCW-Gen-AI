# import required libraries
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from utils_scripts.detect_lane_markings import LaneDetection
from utils_scripts.forward_collision_warning_system import ForwardCollisionWarning


# define the class
class ADAS:
    '''
    This class helps to streamline the task of using the ADAS developed in this project
    '''
    def __init__(self, seg_model_path, lane_det_model_path):
        # create the fcw and ld objects
        self.fcw = ForwardCollisionWarning(seg_model_path, )
        self.ld = LaneDetection(lane_det_model_path)

    def run(self, img_path):
        '''
        This function takes the image path (img_path) as the input,
        and it produces the final output, considering lane detection, raod segmentation and FCW system.
        It outputs: flag, image, lane_mask, fcw_mask, fcw_image_mask, fcw_image_roi
        '''

        # segment the road image
        image, pred = self.fcw.segment_road(img_path)
        # run the FWC system
        flag, fcw_mask, fcw_image_mask, fcw_image_roi = self.fcw.detect_potential_collision(image, pred)

        # detect lane markings
        _, lane_mask = self.ld.detect_lanes(img_path)

        # combine lane detection and the segmentation mask 
        # the lane mask to match the segmentation mask
        lane_mask = tf.expand_dims(lane_mask, axis=-1)
        lane_mask = tf.image.resize(lane_mask, (192, 256), method='nearest')
        lane_mask = tf.squeeze(lane_mask, axis=-1).numpy()
        # overlay the lane mask over the other images
        fcw_mask[lane_mask.astype(bool)] = [1, 1, 1]
        fcw_image_mask[lane_mask.astype(bool)] = [1, 1, 1]
        fcw_image_roi[lane_mask.astype(bool)] = [1, 1, 1]

        # flag shows whether or not a potential collision is detected
        return flag, image, lane_mask, fcw_mask, fcw_image_mask, fcw_image_roi


    # visualize the outputs of the adas
    def visualize_output_adas(self, flag, lane_mask, fcw_mask, fcw_image_mask, fcw_image_roi):
        flag = 'PC Detected!' if flag == 1 else ('No PC Detected!' if flag == 0 else '---')

        # the lane markings mask
        plt.subplot(2, 2, 1)
        plt.title(f'The Lanes Mask')
        plt.imshow(lane_mask, cmap='gray')

        # the segmentation mask
        plt.subplot(2, 2, 2)
        plt.title(f'The Mask: [{flag}]')
        plt.imshow(fcw_mask)

        # the image with only the ROI overlayed
        plt.subplot(2, 2, 3)
        plt.title(f'The Image - ROI: [{flag}]')
        plt.imshow(fcw_image_roi)

        # the image with the mask overlayed 
        plt.subplot(2, 2, 4)
        plt.title(f'The Image-Mask: [{flag}]')
        plt.imshow(fcw_image_mask)

        plt.show()

# see how it works
if __name__ == '__main__':
    seg_model_path = './models/road-segmentation/road_segmentation_ep-66.h5'
    lane_det_model_path = './models/lane-detection/u-net-model.h5'
    adas = ADAS(seg_model_path, lane_det_model_path)
    
    img_path = './assets/lane-detection/sample images from the val set/ground-truth/img-5.jpg'
    flag, image, lane_mask, fcw_mask, fcw_image_mask, fcw_image_roi = adas.run(img_path)
    
    # visualize the outputs
    plt.figure(figsize=(25, 20))
    adas.visualize_output_adas(flag, lane_mask, fcw_mask, fcw_image_mask, fcw_image_roi)