# import required libraries
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2

try:
    from segment_road import RoadSegmentation
except:
    from utils_scripts.segment_road import RoadSegmentation

# define the class
class ForwardCollisionWarning(RoadSegmentation):
    '''
    This class helps to streamline the task of using the Forward Collision Warning sytem
    '''
    def __init__(self, model_path, input_shape=(192, 256)):
        super().__init__(model_path, input_shape)
        self.__image = None
        
    # overlay the predicted mask on the image
    def overlay_mask_on_image(self, image, colored_mask):
        
        # overlay mask on the image
        mask_image_overlayed = image * .5 + colored_mask/255. * .5

        # make the 'road pixels' more distinguishable
        road_pixel = np.array([178, 178, 178])
        mask_image_overlayed[colored_mask == road_pixel] = .7 * .6 + mask_image_overlayed[colored_mask == road_pixel] * .4
        return mask_image_overlayed
    
    def colorize_segment(self, pred):
        # remove the extra dimension
        mask = np.squeeze(pred)
        # Generate the colored image using the color dictionary
        colored_mask = np.zeros((*mask.shape[0:2], 3))
        for pixel_value, color in self.color_dict.items():
            colored_mask[mask == pixel_value] = color
        else:
            # convert every object that is in the roi to 'red'
            colored_mask[mask == 20] = (1, 0, 0)
        # Convert the image to 8-bit unsigned integer
        colored_mask = (colored_mask * 255).astype(np.uint8)
        return colored_mask
    
    def __potential_coll_util_image_with_mask_overlayed(self, x1, x2, y1, y2, colored_mask):
        image_ = self.__image
        # Draw the rectangle on the image
        color = (0, 255, 0)  # Green
        image_ = cv2.rectangle(image_, (x1, y1), (x2, y2), (0, 1, 0), thickness=1)
        
        # overlay mask on the image
        image_ = self.overlay_mask_on_image(image_, colored_mask)
        
        return image_

    def __potential_coll_util_image_only_roi_overlayed(self, x1, x2, y1, y2, colored_mask):
        image_ = self.__image
        
        # Draw the rectangle on the image
        color = (0, 255, 0)  # Green
        image_ = cv2.rectangle(image_, (x1, y1), (x2, y2), (0, 1, 0), thickness=1)

        image_[y2:y1, x1:x2] = colored_mask[y2:y1, x1:x2]/255. * .7 + image_[y2:y1, x1:x2] * .3
        return image_

    def detect_potential_collision(self, image, mask):
        self.__image = image
        image = self.__image
        
        # determine the RoI
        # Define the rectangle coordinates
        x1, x2 = 20, 256-20
        y1, y2 = 190, 130

        # crop the RoI
        roi = mask[y2:y1, x1:x2]
        # detect objects other than road in the RoI
        mask[y2:y1, x1:x2] = np.where(np.logical_or(roi == 0, roi == 19), roi, 20)

        # colorize the mask: converts objects in the roi to 'red'
        mask_colored = self.colorize_segment(mask)

        # Draw the rectangle on the mask
        color = (0, 255, 0)  # Green
        mask_colored = cv2.rectangle(mask_colored, (x1, y1), (x2, y2), color, thickness=1)    

        # get image with mask overlayed
        image_with_mask_overlayed = self.__potential_coll_util_image_with_mask_overlayed(x1, x2, y1, y2, mask_colored)

        # get the image with only the roi overlayed
        image_with_only_roi_overlayed = self.__potential_coll_util_image_only_roi_overlayed(x1, x2, y1, y2, mask_colored)
        
        # is there any objects in the roi
        flag = (True in np.logical_and(roi != 0, roi != 19))
        return  flag, mask_colored/255., tf.clip_by_value(image_with_mask_overlayed, 0, 1).numpy() , tf.clip_by_value(image_with_only_roi_overlayed, 0, 1).numpy()
    
    # visualize the outputs
    def visualize_output(self, mask, image_mask_overlayed, image_roi_overlayed, flag=None):
        '''
            mask, image_mask_overlayed, and image_roi_overlayed are the outputs of the method detect_potential_collision(self)
            The pixel values should be in range 0 - 1
        '''
        flag = 'PC Detected!' if flag == 1 else ('No PC Detected!' if flag == 0 else '---')
        
        
        # the mask
        plt.subplot(1, 3, 1)
        plt.title(f'The Mask: [{flag}]')
        plt.imshow(mask)
        
        # the image with the mask overlayed 
        plt.subplot(1, 3, 2)
        plt.title(f'The Image-Mask: [{flag}]')
        plt.imshow(image_mask_overlayed)
        
        # the image with only the ROI overlayed
        plt.subplot(1, 3, 3)
        plt.title(f'The Image - ROI: [{flag}]')
        plt.imshow(image_roi_overlayed)
        
        plt.show()
        
        
if __name__ == '__main__':
    # test how it works
    fcw = ForwardCollisionWarning('./../models/road-segmentation/road_segmentation_ep-66.h5')
    img_path = './../assets/road-segmentation/sample images from the val set/7.jpg'
    # segment the road image
    image, pred = fcw.segment_road(img_path)
    # 
    flag, r_mask, r_image_mask, r_image_roi = fcw.detect_potential_collision(image, pred)
    # visualize the results
    plt.figure(figsize=(20, 8))
    fcw.visualize_output(r_mask, r_image_mask, r_image_roi, flag)