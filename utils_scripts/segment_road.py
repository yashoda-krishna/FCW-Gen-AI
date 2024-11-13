# import required libraries
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# define the class
class RoadSegmentation:
    '''
    This class helps to streamline the task of using the road image segmentation model trained in this project
    '''
    def __init__(self, model_path, input_shape=(192, 256)):
        self.model = tf.keras.models.load_model(model_path, compile=False) # load the model
        self.input_shape = input_shape
        self.__image = None 
        self.__pred = None
        # color dictionary to assign different colors to detected objects
        self.color_dict = {
            0: (0.7, 0.7, 0.7),     # road - gray
            1:  (0.9, 0.9, 0.2),     # sidewalk - light yellow
            2: (1.0, 0.4980392156862745, 0.054901960784313725),
            3: (1.0, 0.7333333333333333, 0.47058823529411764),
            4: (0.8, 0.5, 0.1),  # Fence - rust orange
            5: (0.596078431372549, 0.8745098039215686, 0.5411764705882353),
            6: (0.325, 0.196, 0.361),
            7: (1.0, 0.596078431372549, 0.5882352941176471),
            8:  (0.2, 0.6, 0.2),     # vegetation - green
            9: (0.7725490196078432, 0.6901960784313725, 0.8352941176470589),
            10: (0.5, 0.7, 1.0),     # sky - light blue
            11: (0.6, 0.2, 0.8), # person - purple
            12: (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
            13: (0.0, 0.0, 1.0),  # Car - blue
            14: (0.0, 0.0, 1.0),  # Track - blue
            15: (0.0, 0.0, 1.0),  # Bus - blue
            16: (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
            17: (0.8588235294117647, 0.8588235294117647, 0.5529411764705883),
            18: (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),
            19: (0, 0, 0) # unknown - black
        }

    # load and preprocess the image
    def __preprocess_image(self, image_path):
        # Read the image file using tf.io.read_file
        self.__image = tf.io.read_file(image_path)
        # Decode the image into a tensor
        self.__image = tf.image.decode_image(self.__image)
        # Resize the image to the desired size using Lanczos3 method
        self.__image = tf.image.resize(self.__image, self.input_shape, method=tf.image.ResizeMethod.LANCZOS3)
        self.__image = self.__image / 255.0  # Normalize the image

    # colorize the predicted mask
    def colorize_segments(self, pred_mask):
        # remove the extra dimension
        pred_mask = np.squeeze(pred_mask)
        # Generate the colored image using the color dictionary
        colored_image = np.zeros((*self.input_shape, 3))

        for pixel_value, color in self.color_dict.items():
            colored_image[pred_mask == pixel_value] = color

        # Convert the image to 8-bit unsigned integer
        colored_image = (colored_image * 255).astype(np.uint8)

        return tf.clip_by_value(colored_image, 0, 255)
    
    
    # Segment the road image
    def segment_road(self, image_path):
        self.__preprocess_image(image_path)
        self.__pred = self.model.predict(np.array([self.__image]))[0]
        self.__pred = np.argmax(self.__pred, axis=-1)
        self.__image = tf.clip_by_value(self.__image, 0, 1).numpy()
        return self.__image, self.__pred
            
    # visualize the outputs
    def visualize_output(self, pred_mask):
        '''
            pred_mask is 20-channel mask: (h, w, channel) in one-hot encoding format
        '''
        plt.figure(figsize=(15, 6))
        
        # the input image
        plt.subplot(1, 2, 1)
        plt.title('Input Image')
        plt.imshow(self.__image)

        # predicted mask
        plt.subplot(1, 2, 2)
        plt.title('Predicted Mask')
        plt.imshow(self.colorize_segments(pred_mask))
        
        plt.show()
        
if __name__ == '__main__':
    # test how it works
    rs = RoadSegmentation('./../models/road-segmentation/road_segmentation_ep-66.h5')
    img_path = './../assets/road-segmentation/sample images from the val set/8.jpg'
    
    # segment the road image
    image, pred = rs.segment_road(img_path)
    
    # visualize the results
    rs.visualize_output(pred)