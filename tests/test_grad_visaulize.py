# For every rose has its thorns
import sys
sys.path.append('..')

import unittest
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from h5py import h5l
from functions import guided_backprop_cam, guided_backprop_image, register_gradient, modify_backprop, plot_row, grad_cam
import matplotlib.pyplot as plt

class VisualizeTestCase(unittest.TestCase):
    """Tests the gradient for vggnet"""

    def test_guided_backprop(self):
        """Is five successfully determined to be prime?"""
        #self.assertTrue(is_prime(5))
        model = VGG16(weights='imagenet')
        img_path = 'test_cat_dog.png'
        label_path = 'test_dog_guided_backprop.png'
        img = image.load_img(img_path, target_size=(224, 224,3))
        label = imread(label_path)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        features = model.predict(x)
        pred_class = np.argmax(features)
        
        register_gradient()
        guided_model = modify_backprop(model, 'GuidedBackProp', lambda : VGG16(weights='imagenet'))   

        final = guided_backprop_cam(model, guided_model, x, pred_class, 'predictions', 'block5_conv3')
        #plt.imshow(final)
        print(final.mean(), label.mean())

        np.testing.assert_array_equal(final, label)

if __name__ == "__main__":
    unittest.main()
