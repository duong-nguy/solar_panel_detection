import os
import re
import shutil

import tifffile
import tensorflow as tf
import numpy as np



class Predict:
    def __init__(self,
                 project_root,
                 sample_path
                 ):
        self.project_root= project_root
        self.model_path = os.path.join(self.project_root,'train',f'{project_root}_model.keras')
        self.test_path = os.path.join(self.project_root,'feature_selection','test_dataset')
        self.prediction_path = prediction_path
        self.sample_path = sample_path

    def _load_model(self):
        self.model = tf.keras.models.load_model(self.model_path)


    def _resize_output(self,predict_mask,test):
        '''
            From:
            [[1,2,3,0],
             [4,5,6,0],
             [0,0,0,0]]
            To:
            [[1,2,3],
             [4,5,6]]
        '''
        sample_name = re.sub('s2_image','mask',test)
        sample_name = re.sub('.npy','.tif',sample_name)
        sample_path = os.path.join(self.sample_path,sample_name)
        sample= tifffile.imread(sample_path)
        predict_mask = predict_mask[0,:sample.shape[0],:sample.shape[1],0]
        predict_mask = np.squeeze(predict_mask)
        return predict_mask,sample_name


    def _make_prediction(self):
        self._load_model()
        if os.path.exists(self.prediction_path):
            print(f'Remove {self.prediction_path} and its contents')
            shutil.rmtree(self.prediction_path)
        print(f'Create {self.prediction_path}')
        os.makedirs(self.prediction_path)
        tests = os.listdir(self.test_path)
        for test in tests:
            test_image_path = os.path.join(self.test_path,test)
            test_data = np.load(test_image_path)
            predict_mask = self.model.predict(test_data)
            predict_mask,predict_mask_save_name = self._resize_mask(predict_mask,test)
            predict_mask_save_path = os.path.join(self.prediction_path,predict_mask_save_name)
            tifffile.imsave(predict_mask_save_path,predict_mask)

    def run(self):
        self._make_prediction()
        print('Finish')
