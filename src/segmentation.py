""" Script to train model"""

__author__ = 'Chien-Hsiang Hsu'
__create_date__ = '2019.04.24'


import os, re
import functools
import yaml
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import datetime

import matplotlib.pyplot as plt
import cv2

import data_io
import u_net

import tensorflow as tf
from tensorflow.keras import models, layers, losses


CONFIGS_FOLDER = 'configs'
COMMON_YAML = 'common.yaml'

# Limit GPU memory usage
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# print(physical_devices)
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# for g in physical_devices:
#     tf.config.experimental.set_memory_growth(g, True)

# for g in physical_devices:
#     print(tf.config.experimental.get_memory_growth(g))


class Task:
    def __init__(self, task_yaml):
        self.cfg = self.get_cfg_from_yaml(task_yaml)


    def get_yaml_path_from_name(self, fname):
        return os.path.join(CONFIGS_FOLDER, fname)


    def get_cfg_from_yaml(self, yaml_name):
        # get common paths
        with open(self.get_yaml_path_from_name(COMMON_YAML)) as f:
            common_cfg = yaml.safe_load(f)

        with open(self.get_yaml_path_from_name(yaml_name)) as f:
            cfg = yaml.safe_load(f) 

        # merge these two configs
        cfg = {**common_cfg, **cfg}

        if 'train_data' in cfg:
            cfg['img_dir'] = os.path.join(cfg['root_folder'], cfg['data_subfolder'],
                                          cfg['train_data'], cfg['img_subfolder'])
            cfg['mask_dir'] = os.path.join(cfg['root_folder'], cfg['data_subfolder'],
                                           cfg['train_data'], cfg['mask_subfolder'])

        if 'test_data' in cfg:
            cfg['test_img_dir'] = os.path.join(cfg['root_folder'], cfg['data_subfolder'],
                                               cfg['test_data']['name'], cfg['img_subfolder'])
            cfg['test_mask_dir'] = os.path.join(cfg['root_folder'], cfg['data_subfolder'],
                                               cfg['test_data']['name'], cfg['mask_subfolder'])
        
        # conversion string to number or function handle
        if 'read_cfg' in cfg:
            cfg['read_cfg']['scale'] = eval(cfg['read_cfg']['scale'])
        
        if 'metrics' in cfg:
            cfg['metrics'] = [eval(s) for s in cfg['metrics']]

        if 'test_read_cfg' in cfg:
            cfg['test_read_cfg']['scale'] = eval(cfg['test_read_cfg']['scale'])

        if 'optimizer' in cfg:
            cfg['optimizer'] = eval(cfg['optimizer'])
        else:
            cfg['optimizer'] = 'adam'
            
        return cfg  


    # For getting tf datasets
    def get_train_val_dataset(self, return_file_names=False):
        cfg = self.cfg

        x_train_fnames, x_val_fnames, y_train_fnames, y_val_fnames = \
            data_io.get_data_filenames(**cfg)

        num_train_data = len(x_train_fnames)
        num_val_data = len(x_val_fnames)


        ### Configure training and validation dataset
        read_img_fn = functools.partial(data_io._get_image_from_path, **cfg['read_cfg'])

        batch_size = cfg['batch_size']

        tr_preproc_fn = functools.partial(data_io._augment, **cfg['train_cfg'])
        val_preproc_fn = functools.partial(data_io._augment, **cfg['val_cfg'])

        train_ds = data_io.get_dataset(x_train_fnames, y_train_fnames, read_img_fn=read_img_fn,
                                       preproc_fn=tr_preproc_fn, shuffle=True, batch_size=batch_size)
        val_ds = data_io.get_dataset(x_val_fnames, y_val_fnames, read_img_fn=read_img_fn, 
                                     preproc_fn=val_preproc_fn, shuffle=False, batch_size=batch_size)

        if return_file_names:
            return train_ds, val_ds, num_train_data, num_val_data, batch_size, \
                   x_train_fnames, x_val_fnames, y_train_fnames, y_val_fnames

        else:
            return train_ds, val_ds, num_train_data, num_val_data, batch_size


    # Build the model
    def get_model(self):
        num_filters_list = self.cfg['num_filters_list']
        n_classes = self.cfg['n_classes']
        metrics = self.cfg['metrics']
        optimizer = self.cfg['optimizer']

        model = u_net.Unet(num_filters_list, n_classes=n_classes, dynamic=True)

        # loss functions
        loss_fn = self.get_loss_fn_from_name(self.cfg['loss_fn_name'])

        model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics, run_eagerly=True)

        return model


    def get_loss_fn_from_name(self, loss_fn_name):
        w_cfg = self.cfg['w_cfg']
        fn_map = {
            'weighted_bce': functools.partial(u_net.weighted_bce_loss, 
                                              w0=w_cfg['w0'], sigma=w_cfg['sigma']),
            'weighted_bce_dice': functools.partial(u_net.weighted_bce_dice_loss, 
                                                   w0=w_cfg['w0'], sigma=w_cfg['sigma']),
            'unweighted_bce': losses.binary_crossentropy,
            'unweighted_bce_dice': u_net.bce_dice_loss,
            'unweighted_dice': u_net.dice_loss
        }

        return fn_map[loss_fn_name]


    def get_trained_model(self):
        # Load the trained model
        model_dir = self.get_model_dir()
        latest = tf.train.latest_checkpoint(model_dir)
        print()
        print('Loading model from:')
        print('  ', latest)

        model = self.get_model()
        model.load_weights(latest)

        return model


    def get_init_epoch(self, latest):

        if latest is None:
            init_epoch = 0

        else:
            pattern = re.compile('-(?P<epoch>\d+)\.')
            m = pattern.search(os.path.basename(latest))
            init_epoch = int(m.group('epoch'))
            
        return init_epoch


    def train_model(self, transfer=False, src_model_yaml=''):
        cfg = self.cfg
        monitor = cfg['monitor']
        epochs = cfg['epochs']
        model_dir = self.get_model_dir()

        if not os.path.isdir(model_dir):
            os.makedirs(model_dir, exist_ok=True)

        # get training data
        train_ds, val_ds, num_train_data, num_val_data, batch_size = self.get_train_val_dataset()

        # build the model
        if transfer:
            assert src_model_yaml is not '', "Require source model yaml to transfer."
            src_model = Task(src_model_yaml)
            model = src_model.get_trained_model()
            latest = None

        else:
            # check whether to resume previous training
            latest = tf.train.latest_checkpoint(model_dir)
            if latest is not None: # previous training exists
                act = input("This model has been trained. Resume training? (R)esume | (N)ew | (A)bort: ")

                if act in ['R', 'r']:
                    model = self.get_trained_model()

                elif act in ['N', 'n']:
                    model = self.get_model()
                    latest = None
                    
                else:
                    print("Abort")
                    exit(0)
            else:
                model = self.get_model()

        initial_epoch = self.get_init_epoch(latest)

        # checkpoint callback (saving model weights)
        weights_path = os.path.join(model_dir, 'weights-{epoch:04d}.ckpt')
        cp = tf.keras.callbacks.ModelCheckpoint(filepath=weights_path, monitor=monitor, 
                                                save_best_only=True, save_weights_only=True, 
                                                verbose=1)
        # tensorboard callback
        log_dir = self.get_log_dir()
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        tb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, profile_batch=0)

        # training
        history = model.fit(train_ds, epochs=epochs, 
                            steps_per_epoch=int(np.ceil(num_train_data / batch_size)),
                            validation_data=val_ds,
                            validation_steps=int(np.ceil(num_val_data / batch_size)),
                            callbacks=[cp, tb],
                            initial_epoch=initial_epoch)


    def eval_model(self):
        x_train_fnames, x_val_fnames, y_train_fnames, y_val_fnames = \
            data_io.get_data_filenames(**self.cfg)

        # Prepare inputs
        chunk_size = self.cfg['batch_size']
        read_img_fn = functools.partial(data_io._get_image_from_path, **self.cfg['read_cfg'])
        overlay_ans = True

        # Output folders
        prediction_subfolder = self.cfg['prediction_subfolder']
        train_overlay = os.path.join(self.get_result_dir(), 'train_data', 'overlay_ans')
        train_pred = os.path.join(self.get_result_dir(), 'train_data', prediction_subfolder)

        val_overlay = os.path.join(self.get_result_dir(), 'validation_data', 'overlay_ans')
        val_pred = os.path.join(self.get_result_dir(), 'validation_data', prediction_subfolder)

        # Output prediction
        print("Outputing training data prediction...")
        self.output_predictions(x_train_fnames, read_img_fn, chunk_size=chunk_size, overlay_ans=overlay_ans, 
                                ans_files=y_train_fnames, result_folder_pred=train_pred,
                                result_folder_overlay=train_overlay)

        print("Outputing validation data prediction...")
        self.output_predictions(x_val_fnames, read_img_fn, chunk_size=chunk_size, overlay_ans=overlay_ans, 
                                ans_files=y_val_fnames, result_folder_pred=val_pred,
                                result_folder_overlay=val_overlay)


    def test_model(self, test_yaml):

        # Prepare inputs
        test_cfg = self.get_cfg_from_yaml(test_yaml)

        file_dir = test_cfg['test_img_dir']
        file_type = test_cfg['test_data']['file_type']
        filter_patter = test_cfg['test_data']['filter_patter']
        chunk_size = test_cfg['chunk_size']
        test_read_cfg = test_cfg['test_read_cfg']
        prediction_subfolder = test_cfg['prediction_subfolder']

        overlay_ans = test_cfg['output_type'] == 'overlay_ans'

        # Get img_files and ans_files
        img_files = data_io.get_filenames(file_dir, file_type, filter_patter)

        if overlay_ans:
            ans_dir = test_cfg['test_mask_dir']
            ans_files = data_io.get_filenames(ans_dir, file_type, filter_patter)
        else:
            ans_files = None

        read_img_fn = functools.partial(data_io._get_image_from_path, **test_read_cfg)

        # Output folders
        result_folder_overlay = os.path.join(self.get_result_dir(), 
                                             test_cfg['test_data']['name'], 
                                             test_cfg['output_type'])
        result_folder_pred = os.path.join(self.get_result_dir(), 
                                          test_cfg['test_data']['name'], prediction_subfolder)

        # Output prediction
        self.output_predictions(img_files, read_img_fn, chunk_size=chunk_size, overlay_ans=overlay_ans, 
                                ans_files=ans_files, result_folder_pred=result_folder_pred,
                                result_folder_overlay=result_folder_overlay)


    #-----------------------------------------------------------------------------------------------
    # Helper functions
    #-----------------------------------------------------------------------------------------------
    def get_model_dir(self):
        cfg = self.cfg
        return os.path.join(cfg['root_folder'], cfg['model_subfolder'], cfg['model_name'])

    def get_log_dir(self):
        cfg = self.cfg
        return os.path.join(cfg['root_folder'], cfg['log_subfolder'], cfg['model_name'])

    def get_result_dir(self):
        cfg = self.cfg
        return os.path.join(cfg['root_folder'], cfg['result_subfolder'], cfg['model_name'])

    def output_predictions(self, img_files, read_img_fn, chunk_size=1, overlay_ans=False, 
                           ans_files=None, result_folder_pred='', result_folder_overlay=''):
        if not os.path.isdir(result_folder_overlay):
            os.makedirs(result_folder_overlay)

        if not os.path.isdir(result_folder_pred):
            os.makedirs(result_folder_pred)

        # Chunk image_files (memory issue)
        img_files = [img_files[i:i+chunk_size] for i in range(0, len(img_files), chunk_size)]
        if overlay_ans:
            ans_files = [ans_files[i:i+chunk_size] for i in range(0, len(ans_files), chunk_size)]

        # Get trained model
        model = self.get_trained_model()
        
        # Loop thourgh chunks of images and ouput prediction and overlay 
        for i, g in enumerate(img_files): # loop through chunks
            print()
            print("Predicting chunck {}/{}...".format(i+1, len(img_files)))
            test_ds = data_io.get_dataset(g, None, read_img_fn=read_img_fn,
                                          shuffle=False, repeat=False, batch_size=1)
            y_pred = model.predict(test_ds, verbose=1)

            if overlay_ans:
                ans_ds = data_io.get_dataset(ans_files[i], None, read_img_fn=read_img_fn,
                                             shuffle=False, repeat=False, batch_size=1)
            else:
                holder = [[] for i in range(len(g))]
                ans_ds = tf.data.Dataset.from_tensor_slices(holder)

            test_ds = tf.data.Dataset.zip((test_ds, ans_ds))

            for j, (x, y) in enumerate(test_ds):
                print("Saving results {}/{}...".format(j+1, len(g)), end='\r')
                I = np.uint8(x[0]*255.)
                M_pred = np.uint8((y_pred[j,...,0] > 0.5) * 255.)

                if overlay_ans:
                    M = np.uint8((y[0,...,0].numpy() > 0.5) * 255.)
                    I = I * 0

                else:
                    M = np.zeros_like(M_pred)

                # Overlayed image
                I = data_io.overlay_mask(I, M, M_pred)
                if overlay_ans:
                    I[np.where((I==[0,0,0]).all(axis=2))] = [255,255,255]
                    I[np.where((I==[255,255,0]).all(axis=2))] = [0,0,0]
                fname = os.path.join(result_folder_overlay, os.path.basename(g[j]))
                cv2.imwrite(fname, cv2.cvtColor(I, cv2.COLOR_RGB2BGR))

                # Prediction (probability)
                pred_img = np.uint8(y_pred[j,...,0] * 255.)
                fname = os.path.join(result_folder_pred, os.path.basename(g[j]))
                cv2.imwrite(fname, pred_img)

            print()



####################################################################################################
if __name__ == '__main__':
    # Parse arguments
    parser = ArgumentParser()

    parser.add_argument("task_yaml", help="yaml file of the task")
    parser.add_argument("--mode", dest="mode", help="TRAIN, TRANSFER, EVAL or TEST", default="TRAIN")
    parser.add_argument("--gpu_id", dest="gpu", help="ID of GPU to use", default='0')
    parser.add_argument("--src_model_yaml", dest="src_model_yaml", help="Yaml of source model", default='')
    parser.add_argument("--test_yaml", dest="test_yaml", help="Yaml of test data", default='')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    MODE = args.mode

    # Limit GPU memory usage
    # physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # print(physical_devices)
    # assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Get task
    task = Task(args.task_yaml)

    if MODE in ['TRAIN', 'TRANSFER']:
        task.train_model(transfer=MODE=='TRANSFER', src_model_yaml=args.src_model_yaml)

    if MODE == 'EVAL':
        task.eval_model()

    if MODE == 'TEST':
        assert args.test_yaml is not '', "Need to specify test yaml"
        task.test_model(args.test_yaml)
        
    
