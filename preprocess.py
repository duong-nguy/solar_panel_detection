import os
import shutil
import tifffile

import numpy as np
import tensorflow as tf
from tensorflow import keras

BANDS= ['Aerosol','Blue','Green','Red','Red Edge 1','Red Edge 2','Red Edge 3',
         'NIR','Narror NIR','Water Vapour','SWIR','SWIR']
def dvi(data):
    NIR, Red = BANDS.index('NIR'),BANDS.index('Red')
    dvi = (data[:,:,NIR] - data[:,:,Red])
    return np.expand_dims(dvi,axis=-1)

def ndvi(data):
    NIR, Red = BANDS.index('NIR'),BANDS.index('Red')
    ndvi = (data[:,:,NIR] - data[:,:,Red]) / (data[:,:,NIR] + data[:,:,Red])
    return np.expand_dims(ndvi,axis=-1)

def fdi(data):
    NIR, Red = BANDS.index('NIR'),BANDS.index('Red')
    Blue = BANDS.index('Blue')
    fdi = data[:,:,NIR] - (data[:,:,Red] + data[:,:,Blue])
    return np.expand_dims(fdi,axis=-1)

def si(data):
    Blue, Green= BANDS.index('Blue'),BANDS.index('Green')
    Red = BANDS.index('Red')
    si =((1. - data[:, :, Blue]) * (1. - data[:, :, Green]) *
            (1. - data[: , :, Red])) ** (1.0/3.0)
    '''Use fillna'''
    return np.expand_dims(si,axis=-1)

def dwi(data):
    Green, NIR = BANDS.index('Green'),BANDS.index('NIR')
    dwi = data[: , :, Green] - data[: , :, NIR]
    return np.expand_dims(dwi,axis=-1)

def ndwi(data):
    Green, NIR = BANDS.index('Green'),BANDS.index('NIR')
    ndwi = ((data[:, :, Green] - data[:, :, NIR])
            / (data[:, :, Green] + data[:, :, NIR]))
    return np.expand_dims(ndwi,axis=-1)

def cs1(data):
    Blue, Green= BANDS.index('Blue'),BANDS.index('Green')
    Red, NIR = BANDS.index('Red'), BANDS.index('NIR')
    cs1 = ((3.0 * (data[:, :, NIR])) / (data[:, :, Blue]
        + data[:, :, Green] + data[:, :, Red]))
    return np.expand_dims(cs1,axis=-1)

def cs2(data):
    NIR, Blue = BANDS.index('NIR'), BANDS.index('Blue')
    Green, Red = BANDS.index('Green'), BANDS.index('Red')
    cs2 = ((data[: , :, Blue] + data[:, :, Green] + data[:, :, Red]
            + data[: , :, NIR]) / 4.0)
    return np.expand_dims(cs2,axis=-1)

class FeatureEngineer:
    def __init__(self,
                 project_root,
                 train_path,
                 target_path,
                 test_path,
                 output_shape
                 ):
        self.train_path = train_path
        self.target_path = target_path
        self.test_path = test_path
        self.root = project_root
        self.output_shape = output_shape
        self.outputs_path = os.path.join(self.root,'feature_engineer')
        self.train_save = os.path.join(self.outputs_path,'train_dataset')
        self.test_save = os.path.join(self.outputs_path,'test_dataset')


        if not os.path.exists(self.root):
            print(f'Create {self.root}')
            os.mkdir(self.root)

        if not os.path.exists(self.outputs_path):
            print(f'Create {self.outputs_path}')
            os.mkdir(self.outputs_path)

        else:
            print(f'Remove old {self.outputs_path}')
            shutil.rmtree(self.outputs_path)
            print(f'Create new {self.outputs_path}')
            os.mkdir(self.outputs_path)

    def _get_indices(self,img):
        # Please don't change the order
        concat = [cs1(img),cs2(img),
                  dwi(img),dvi(img),
                  ndwi(img),ndvi(img),
                  si(img),fdi(img)]

        res_indices = np.concatenate(concat,axis=-1)
        res_indices = np.nan_to_num(res_indices).astype('float32')

        return res_indices



    def _load_data(self):
        train_files = [os.path.join(self.train_path,f)
                       for f in os.listdir(self.train_path)]
        test_files = [os.path.join(self.test_path,f)
                        for f in os.listdir(self.test_path)]

        train_save = os.path.join(self.outputs_path,'train')
        test_save = os.path.join(self.outputs_path,'test')
        os.mkdir(train_save)
        print(f'Create {train_save}')
        os.mkdir(test_save)
        print(f'Create {test_save}')

        for train_file,test_file in zip(train_files,test_files):
            train_img = tifffile.imread(train_file).astype("float32") / 10_000
            test_img= tifffile.imread(test_file).astype("float32") / 10_000
            # try per image normalization ??

            generated_train_indices = self._get_indices(train_img)
            generated_test_indices = self._get_indices(test_img)

            train_indices_save = os.path.join(
                    train_save,
                    os.path.splitext(
                        os.path.split(train_file)[-1])[0]
                    )

            test_indices_save = os.path.join(
                    test_save,
                    os.path.splitext(
                        os.path.split(test_file)[-1])[0]
                    )

            np.save(train_indices_save,generated_train_indices)
            np.save(test_indices_save,generated_train_indices)


    def run(self):
        self._load_data()



class Preprocessor:
    def __init__(self,
                 project_root,
                 train_path,
                 target_path,
                 test_path,
                 output_shape
                 ):
        self.train_path = train_path
        self.target_path = target_path
        self.test_path = test_path
        self.root = project_root
        self.output_shape = output_shape
        self.outputs_path = os.path.join(self.root,'preprocessing')
        self.train_save = os.path.join(self.outputs_path,'train_dataset')
        self.test_save = os.path.join(self.outputs_path,'test_dataset')
        self.feature_engineer = os.path.exists(
                os.path.join(self.root,'feature_engineer'))



        if not os.path.exists(self.root):
            print(f'Create {self.root}')
            os.mkdir(self.root)

        if not os.path.exists(self.outputs_path):
            print(f'Create {self.outputs_path}')
            os.mkdir(self.outputs_path)

        else:
            print(f'Remove old {self.outputs_path}')
            shutil.rmtree(self.outputs_path)
            print(f'Create new {self.outputs_path}')
            os.mkdir(self.outputs_path)

        if self.feature_engineer:
            self.train_fe_path = os.path.join(
                    self.root,'feature_engineer','train')
            self.test_fe_path = os.path.join(
                    self.root,'feature_engineer','test')

    def _resize(self,img):
        '''
            Zeros fill resize
        '''
        if len(img.shape) == 3:
            res_img = np.zeros(
                    (self.output_shape[0],
                    self.output_shape[1],
                    img.shape[-1]))

            res_img[:img.shape[0],:img.shape[1],:] = img
        else:
            res_img = np.zeros(self.output_shape[:-1])
            res_img[:img.shape[0],:img.shape[1]] = img
            res_img = np.expand_dims(res_img,-1)

        return res_img

    def _get_fe_file_name(self,file,is_train):
        # remove file extention
        file_name = os.path.splitext(os.path.split(file)[-1])[0]
        if is_train:
            fe_file_name = os.path.join(self.train_fe_path,
                                        f'{file_name}.npy')
        else:
            fe_file_name = os.path.join(self.test_fe_path,
                                        f'{file_name}.npy')
        return fe_file_name




    def _load_data(self):
        train_files = [os.path.join(self.train_path,f)
                       for f in os.listdir(self.train_path)]
        target_files = [os.path.join(self.target_path,f)
                        for f in os.listdir(self.target_path)]
        test_files = [os.path.join(self.test_path,f)
                        for f in os.listdir(self.test_path)]
        train = []
        target = []
        test = []

        for train_file,target_file,test_file in zip(train_files,target_files,test_files):
            # try per image normalization ??
            train_img = tifffile.imread(train_file).astype("float32") / 10_000
            mask = tifffile.imread(target_file).astype("float32")
            test_img= tifffile.imread(test_file).astype("float32") / 10_000

            train_img_fe = np.load(self._get_fe_file_name(
                train_file,is_train=True))
            test_img_fe = np.load(self._get_fe_file_name(
                test_file,is_train=False))

            train_img = self._resize(train_img)
            train_img_fe = self._resize(train_img_fe)

            test_img = self._resize(test_img)
            test_img_fe = self._resize(test_img_fe)

            mask = self._resize(mask)

            train_img = np.concatenate(
                    (train_img,train_img_fe),axis=-1)
            test_img = np.concatenate(
                    (test_img,test_img_fe),axis=-1)

            # Some project clip their value to 0 and 1
            train_img = np.clip(train_img,0,1)
            test_img = np.clip(test_img,0,1)

            train.append(train_img)
            target.append(mask)
            test.append(test_img)

        train = np.array(train)
        target = np.array(target)
        test = np.array(test)
        return train,target,test

    def _save_data(self,train,target,test):

        train_ds = tf.data.Dataset.from_tensor_slices((train,target))
        test_ds = tf.data.Dataset.from_tensor_slices(test)

        train_ds.save(self.train_save,compression='GZIP')
        test_ds.save(self.test_save,compression='GZIP')

        print(self.train_save)
        print(self.test_save)

    def run(self):
        train,target,test = self._load_data()
        self._save_data(train,target,test)





class PerturbationRank:
    def __init__(self,
                 project_root,
                 hypermodel,
                 batch,
                 epochs,
                 ):
        '''
            hypermodel: a function return a compiled keras model
        '''
        self.root = project_root
        self.outputs_path = os.path.join(self.root,'perturbation_rank')
        self.train_dataset_path = os.path.join(self.root,'preprocessing/train_dataset')
        self.hypermodel = hypermodel
        self.batch = batch
        self.epochs = epochs

        self._verbose = 2
        self._perturbation_idx = 0

        if not os.path.exists(self.root):
            raise FileNotFoundError(f'Project {root} is not exists. Please run Preprocessor first')

        if not os.path.exists(self.outputs_path):
            print(f'Create {self.outputs_path}')
            os.mkdir(self.outputs_path)

        else:
            print(f'Delete {self.outputs_path}')
            shutil.rmtree(self.outputs_path)
            print(f'Create new {self.outputs_path}')
            os.mkdir(self.outputs_path)


    def _perturbation(self,x):
        stack = []
        for i in range(self.num_features):
            if i == self._perturbation_idx:
                stack.append(tf.zeros_like(x[:,:,:,i]))
            else:
                stack.append(tf.ones_like(x[:,:,:,i]))
        mask = tf.stack(stack,axis=-1)
        return tf.multiply(x,mask)

    def _map(self,x,y):
        x = self._perturbation(x)
        return x,y

    def _prepare_dataset(self):
        dataset = tf.data.Dataset.load(self.train_dataset_path,compression='GZIP')
        dataset_len = len(dataset)
        num_train = (dataset_len // 10) * 8

        train_dataset = dataset.take(num_train)
        valid_dataset = dataset.skip(num_train)

        train_dataset = train_dataset.batch(128).cache()
        valid_dataset = valid_dataset.batch(self.batch).cache().prefetch(
                tf.data.AUTOTUNE)

        return train_dataset,valid_dataset
    def _apply_map(self,dataset):
        return dataset.map(self._map,num_parallel_calls=tf.data.AUTOTUNE).rebatch(
                        self.batch).prefetch(tf.data.AUTOTUNE)

    def _get_num_features(self,data_sample):
        image,mask = next(iter(data_sample.unbatch()))
        num_bands = image.shape[-1]
        input_shape = image.shape
        output_shape = mask.shape
        return num_bands,input_shape,output_shape


    def _make_perturbation_rank(self,train_dataset,valid_dataset):
        results = {}
        self.num_features,self.input_shape,self.output_shape = self._get_num_features(
                train_dataset.take(1))

        for i in range(self.num_features):
            #Make baseline
            if i == 0:
                tf.keras.backend.clear_session()
                model = self.hypermodel(self.input_shape,self.output_shape)
                model.fit(train_dataset.rebatch(self.batch).prefetch(tf.data.AUTOTUNE),
                            epochs=self.epochs,
                            verbose=self._verbose)

                score = model.evaluate(valid_dataset,
                            verbose=self._verbose)
                filename = os.path.join(self.outputs_path,'baseline.txt')
                with open(filename, 'w') as file:
                    file.write(str(score))

            tf.keras.backend.clear_session()
            model = self.hypermodel(self.input_shape,self.output_shape)
            self._perturbation_idx = i

            model.fit(self._apply_map(train_dataset),
                      epochs=self.epochs,
                      verbose=self._verbose)

            score = model.evaluate(self._apply_map(valid_dataset),
                           verbose=self._verbose)
            results[i] = score


        filename = os.path.join(self.outputs_path,'results.txt')
        with open(filename, 'w') as file:
            file.write(str(results))

    def run(self):
        train_dataset,valid_dataset = self._prepare_dataset()
        self._make_perturbation_rank(train_dataset,valid_dataset)


class FeatureSelection:
    def __init__(self,
                 threshold,
                 project_root,
                 train_path,
                 target_path,
                 test_path,
                 output_shape,
                 chanels
                 ):
        '''
            threshold: float perturbation thread hold for selection  image chanels
            chanels:list of bands of light and feature engineer indices
        '''

        self.threshold = threshold
        self.train_path = train_path
        self.target_path = target_path
        self.test_path = test_path
        self.root = project_root
        self.output_shape = output_shape

        self.outputs_path = os.path.join(self.root,'feature_selection')
        self.perturbation_rank_result = os.path.join(self.root,'perturbation_rank','results.txt')
        self.train_save = os.path.join(self.outputs_path,'train_dataset')
        self.test_save = os.path.join(self.outputs_path,'test_dataset')
        self.feature_engineer = os.path.exists(
                os.path.join(self.root,'feature_engineer'))

        self.chanels = chanels
        #self.chanels = self._get_image_chanels(self.perturbation_rank_result,self.threshold)


        if not os.path.exists(self.root):
            print(f'Create {self.root}')
            os.mkdir(self.root)

        if not os.path.exists(self.outputs_path):
            print(f'Create {self.outputs_path}')
            os.mkdir(self.outputs_path)

        else:
            print(f'Remove old {self.outputs_path}')
            shutil.rmtree(self.outputs_path)
            print(f'Create new {self.outputs_path}')
            os.mkdir(self.outputs_path)

        if self.feature_engineer:
            self.train_fe_path = os.path.join(
                    self.root,'feature_engineer','train')
            self.test_fe_path = os.path.join(
                    self.root,'feature_engineer','test')

    def _get_image_chanels(self,perturbation_rank_result,threshold):
        '''
        Parse perturbation_rank_result to a dict
        and choose chanels above a thread_hold
        perturbation_rank_result: string path to perturbation rank result
        Chat GPT wrote this

        '''
        with open(perturbation_rank_result, 'r') as file:
            content = file.read()
        my_dict = eval(content)
        selected_keys = [key for key, value in my_dict.items() if value >= threshold]
        return selected_keys


    def _resize(self,img):
        '''
            Zeros fill resize
        '''
        if len(img.shape) == 3:
            res_img = np.zeros(
                    (self.output_shape[0],
                    self.output_shape[1],
                    img.shape[-1]))

            res_img[:img.shape[0],:img.shape[1],:] = img
        else:
            res_img = np.zeros(self.output_shape[:-1])
            res_img[:img.shape[0],:img.shape[1]] = img
            res_img = np.expand_dims(res_img,-1)

        return res_img

    def _chanel_selection(self,img,chanels):
        return img[:,:,chanels]

    def _feature_selection(self,img,chanels,output_shape):
        '''
            img: np.array, images and mask(label)
            chanels: list of color chanels
            output_shape: tuple preprocessing image output shape,
        '''
        '''
            Future note: feature_selection may not be aproviate name if we this step before feed to
            train module
            if new module are made for dimentional reduction, and feature engineer then leave it as it
        '''
        img = self._resize(img,output_shape)
        if img.shape[-1] >= len(chanels):
            img = self._chanel_selection(img,chanels)


        return img




    def _get_fe_file_name(self,file,is_train):
        # remove file extention
        file_name = os.path.splitext(os.path.split(file)[-1])[0]
        if is_train:
            fe_file_name = os.path.join(self.train_fe_path,
                                        f'{file_name}.npy')
        else:
            fe_file_name = os.path.join(self.test_fe_path,
                                        f'{file_name}.npy')
        return fe_file_name




    def _load_data(self):
        train_files = [os.path.join(self.train_path,f)
                       for f in os.listdir(self.train_path)]
        target_files = [os.path.join(self.target_path,f)
                        for f in os.listdir(self.target_path)]
        test_files = [os.path.join(self.test_path,f)
                        for f in os.listdir(self.test_path)]
        train = []
        target = []
        test = []

        for train_file,target_file,test_file in zip(train_files,target_files,test_files):
            train_img = tifffile.imread(train_file).astype("float32") / 10_000
            mask = tifffile.imread(target_file).astype("float32")
            test_img= tifffile.imread(test_file).astype("float32") / 10_000

            train_img_fe = np.load(self._get_fe_file_name(
                train_file,is_train=True))
            test_img_fe = np.load(self._get_fe_file_name(
                test_file,is_train=False))

            train_img = self._resize(train_img)
            train_img_fe = self._resize(train_img_fe)

            test_img = self._resize(test_img)
            test_img_fe = self._resize(test_img_fe)

            mask = self._resize(mask)

            train_img = np.concatenate(
                    (train_img,train_img_fe),axis=-1)
            test_img = np.concatenate(
                    (test_img,test_img_fe),axis=-1)

            # feature_selection here
            train_img = self._chanel_selection(train_img,self.chanels)
            test_img = self._chanel_selection(test_img,self.chanels)

            train.append(train_img)
            target.append(mask)
            test.append(test_img)


        train = np.array(train)
        target = np.array(target)
        test = np.array(test)
        return train,target,test

    def _save_data(self,train,target,test):

        train_ds = tf.data.Dataset.from_tensor_slices((train,target))
        test_ds = tf.data.Dataset.from_tensor_slices(test)

        train_ds.save(self.train_save,compression='GZIP')
        test_ds.save(self.test_save,compression='GZIP')

        print(self.train_save)
        print(self.test_save)

    def run(self):
        train,target,test = self._load_data()
        self._save_data(train,target,test)

