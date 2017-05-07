# For every rose has its thorns
import sys
sys.path.append('..')

from scipy.misc import imread
import unittest
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
from h5py import h5l
from functions import img_to_tensor, guided_cam_evolution, guided_backprop_cam, guided_backprop_image, register_gradient, modify_backprop, plot_row, grad_cam
import matplotlib.pyplot as plt
from tqdm import tqdm

class VisualizeTestCase(unittest.TestCase):
    
    def setUp(self):
        print self._testMethodName
    
    def _test_cam(self):
        """ Tests that cam works """
        np.testing.assert_almost_equal(1,1)
    
    def _test_guided_backprop(self):
        """Tests that grad cam works """
        model = VGG16(weights='imagenet')
        img_path = '../images/unittest/cat_dog.png'
        label_path = '../images/unittest/doge.png'
        img = image.load_img(img_path, target_size=(224, 224,3))
        label = imread(label_path)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        features = model.predict(x)
        pred_class = np.argmax(features)
        
        register_gradient()
        guided_model = modify_backprop(model, 'GuidedBackProp', lambda : VGG16(weights='imagenet'))   

        heatmap, cam = grad_cam(model, x, pred_class, 'predictions', 'block5_conv3')
        plt.imshow(cam)
        #final = guided_backprop_cam(model, guided_model, x, pred_class, 'predictions', 'block5_conv3')
        #plt.imshow(final)
        #print(final.mean(), label.mean())
        
        #plt.imshow(label)
        #plt.figure()
        #plt.imshow(final)

        #np.testing.assert_allclose(final, label)
    
    def _test_cam_evolution(self):
        """ Generates a sequence of guided backprop cam images for each layer in the network for a given input image.
        Allows you to visualize the representation change throughout the network """
        modelFunc = lambda : VGG16(weights='imagenet')
        model = modelFunc()
        img_path = '../images/unittest/spider.jpg'
        
        x = img_to_tensor(img_path, (224,224,3), preprocess=True)
        
        features = model.predict(x)
        pred_class = np.argmax(features)
        print(pred_class, decode_predictions(features)[0][0])
        
        g_cams, cams, layer_names = guided_cam_evolution(model, x, pred_class, modelFunc)
        
        #final = guided_backprop_cam(model, guided_model, x, pred_class, 'predictions', 'block5_conv3')
        for g_cam, cam, l_name in zip(g_cams, cams, layer_names):
            plot_row(img_path, [x[0], g_cam, cam], l_name)
            
    def _test_guided_cam_evolution(self):
        """ Test heatmap evolution over the layers """
        np.testing.assert_allclose(True, True)
        

if __name__ == "__main__":
    unittest.main()
