# import required libraries
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# define the class
class LaneDetection:
    '''
    This class helps to streamline the task of using the lane detection models trained in this project
    '''
    def __init__(self, model_path, input_shape=(256, 320)):
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.input_shape = input_shape
        self.__image = None
        self.__pred = None
    
    # preprocess the image
    def __preprocess_image(self, image_path):
        self.__image = tf.keras.preprocessing.image.load_img(image_path)
        self.__image = tf.keras.preprocessing.image.img_to_array(self.__image)
        self.__image = tf.image.resize(self.__image, self.input_shape)
        
    # detect lanes
    def detect_lanes(self, image_path):
        self.__preprocess_image(image_path)
        self.__pred = self.model.predict(np.array([self.__image]))[0]
        self.__pred = (self.__pred > .5).astype('int').reshape(*self.input_shape)
        return self.__image, self.__pred
    
    # visualize the outputs
    def visualize_output(self):
        plt.figure(figsize=(15, 6))
        
        # the input image
        plt.subplot(1, 2, 1)
        plt.title('Input Image')
        plt.imshow(self.__image/255.)

        # predicted mask
        plt.subplot(1, 2, 2)
        plt.title('Predicted Lane Markings')
        plt.imshow(self.__pred, cmap='gray')
        
        plt.show()


if __name__ == '__main__':
    # test how it works
    ld = LaneDetection('./../models/lane-detection/u-net-model.h5')
    img_path = './../assets/lane-detection/sample images from the val set/ground-truth/img-5.jpg'
    # detect lane markings
    image, pred = ld.detect_lanes(img_path)
    
    # visualize the outputs
    ld.visualize_output()