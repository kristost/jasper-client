import logging
import jasperpath

import subprocess
import pickle
import numpy as np
import pandas as pd
import arff as liacarff
import timeit
from datetime import timedelta, datetime
import os
import shutil

class Emotion(object):

    def __init__(self, sessionLogging=False, sessionRoot='~', xbow_enabled=False):

        self._logger = logging.getLogger(__name__)
        self._feature_set = 'eGeMAPSv01a'
        self._feature_map = dict({'ComParE_2016': [6376,6375,6374], 'eGeMAPSv01a': [-1, 91, 89, 88]})

        # openSMILE/sklearn classifier
        model, scaler, encoder = pickle.load(open('Classifier.pkl', 'rb'))
        self._model = model
        self._scaler = scaler
        self._encoder = encoder

        # openXBOW/sklearn classifier
        self._xbow_model, self._xbow_scaler, self._xbow_encoder = pickle.load(open('openXBOW/emodb_ComParE_2016_xbow.pkl', 'rb'))

        # TODO: Use jasper config (yaml) file to retrieve openSMILE path/bin location

        self._opensmile_path = '/home/pi/openSMILE/opensmile-2.3.0/'
        self._bin_path = 'inst/bin/SMILExtract'
        if 'GeMAPS' in self._feature_set:
            self._config_path = 'config/gemaps/' + self._feature_set + '.conf'
        else:
            self._config_path = 'config/' + self._feature_set + '.conf'

        self._openxbow_path = '/home/pi/github/openXBOW/'
        self._openxbow_bin = 'openXBOW.jar'

        self._sessionLogging = sessionLogging
        if self._sessionLogging:
            homedir = os.path.expanduser(sessionRoot)
            self._sessionRoot = homedir + '/' + 'session_' + datetime.today().strftime('%d%m%YT%H%M')

            if not os.path.exists(self._sessionRoot):
                self._logger.info('Creating session directory: "{}"'.format(self._sessionRoot))
                os.makedirs(self._sessionRoot)

        self._xbow_enabled = xbow_enabled        

    def featuriseOpenSMILE(self, input, output):

        if self._sessionLogging == True:
            timestamp = jasperpath.get_timestamp()
            output = self._sessionRoot + '/' + timestamp + '.arff'
            lld_output = self._sessionRoot + '/' + timestamp + '.lld.arff'
        
        args = [self._opensmile_path + self._bin_path, ' -noconsoleoutput 1 -appendarff 0',
                ' -C ', self._opensmile_path + self._config_path,
                ' -I ', input,
                ' -O ', output,
                ' -timestamparff 1']

        if self._xbow_enabled:
            args.extend(' -lldarffoutput ', lld_output, ' -appendarfflld 0 ')

        self._logger.debug(args)
        cmd = ''.join(args)

        start_time = timeit.default_timer()

        p = subprocess.Popen(cmd, shell=True, bufsize=-1)
        p_status = p.wait()

        elapsed = timeit.default_timer() - start_time
        self._logger.info('Total elapsed time for openSMILE feature extraction: {}'.format(str(timedelta(seconds=elapsed))))
        
        if p_status == 0 and self._sessionLogging:
            dest = self._sessionRoot + '/' + timestamp + '.wav'
            self._logger.info("Copying WAV file '{}' to '{}'".format(input, dest))
            shutil.copyfile(input, dest)
        
        return (p_status, output, lld_output)

    def featuriseOpenXBOW(self, input, output):

        if self._sessionLogging == True:
            timestamp = jasperpath.get_timestamp()
            output = self._sessionRoot + '/' + timestamp + '.xbow.arff'
        
        args = ['java -jar ', self._openxbow_path + '/' + self._openxbow_bin,
                ' -b ', 'openXBOW/emodb_ComParE_2016.codebook', #TODO: make configurable...
                ' -i ', input,
                ' -o ', output,
                ' -noLabels ',
                ' -attributes nt1[65]2[65]c'] #TODO: make configurable

        self._logger.debug(args)
        cmd = ''.join(args)

        start_time = timeit.default_timer()

        p = subprocess.Popen(cmd, shell=True, bufsize=-1)
        p_status = p.wait()

        elapsed = timeit.default_timer() - start_time
        self._logger.info('Total elapsed time for openXBOW feature extraction: {}'.format(str(timedelta(seconds=elapsed))))
        
        return (p_status, output)


    def predict(self, feature_file, event_type):
        '''
        # This function doesn't use pandas dataframes, so should be faster without it
        '''
        func_start_time = timeit.default_timer()

        start_time = timeit.default_timer()

        self._logger.info('Loading data from ARFF file {}'.format(feature_file))

        data = liacarff.load(open(feature_file))
        self._logger.info('ARFF file loaded.')

        elapsed = timeit.default_timer() - start_time
        self._logger.info('Total elapsed time to load ARFF: {}'.format(str(timedelta(seconds=elapsed))))

        start_time = timeit.default_timer()

        self._logger.info('Processing features from ARFF into list/numpy array.')
        X = data['data'][0]
        self._logger.info('Found {} features'.format(len(X)))

        if len(X) == self._feature_map[self._feature_set][0]:
            # Remove 'name', 'frameTime' and 'class'
            X = X[2:-1]
        elif len(X) == self._feature_map[self._feature_set][1]:
            # Remove 'name' and 'class'
            X = X[1:-1]
        elif len(X) == self._feature_map[self._feature_set][2]:
            # Remove 'name'
            X = X[1:]

        print('Features truncated to {}'.format(len(X)))

        X = np.reshape(X, (1,-1))
        print(X.shape)
        X = self._scaler.transform(X)
        
        elapsed = timeit.default_timer() - start_time
        self._logger.info('Total elapsed time to create/scale list/numpy array: {}'.format(str(timedelta(seconds=elapsed))))

        start_time = timeit.default_timer()

        self._logger.info('Making prediction on emotion...')
        prediction = self._model.predict(X)
        label = self._encoder.inverse_transform(prediction)
        if self._sessionLogging:
            #print(feature_file)
            #print(os.path.basename(feature_file))
            #print(os.path.splitext(os.path.basename(feature_file)))
            timestamp, _ = os.path.splitext(os.path.basename(feature_file))
            #print(timestamp)
            with open(self._sessionRoot + '/' + 'emotions.csv', 'ab') as f:
                f.write(','.join([timestamp, 'openSMILE', event_type, str(prediction[0]), label[0], '\n']))

        self._logger.info('Emotion predicted: {}'.format((prediction, label)))

        elapsed = timeit.default_timer() - start_time
        self._logger.info('Total elapsed time to make a prediction: {}'.format(str(timedelta(seconds=elapsed))))

        elapsed = timeit.default_timer() - func_start_time
        self._logger.info('Total elapsed time for function: {}'.format(str(timedelta(seconds=elapsed))))

        return (prediction, label)


    def predictXBOW(self, feature_file, event_type):
        '''
        # This function doesn't use pandas dataframes, so should be faster without it
        '''
        func_start_time = timeit.default_timer()

        start_time = timeit.default_timer()

        self._logger.info('Loading data from XBOW ARFF file {}'.format(feature_file))

        data = liacarff.load(open(feature_file))
        self._logger.info('XBOW ARFF file loaded.')

        elapsed = timeit.default_timer() - start_time
        self._logger.info('Total elapsed time to load XBOW ARFF: {}'.format(str(timedelta(seconds=elapsed))))

        start_time = timeit.default_timer()

        self._logger.info('Processing features from XBOW ARFF into list/numpy array.')
        X = data['data'][0]
        self._logger.info('Found {} features'.format(len(X)))

        #if len(X) == 6375:
        #    X = X[1:-1]
        #    print('Features truncated to {}'.format(len(X)))
        #elif len(X) == 6374:
        #    X = X[1:]
        #    print('Features truncated to {}'.format(len(X)))

        X = np.reshape(X, (1,-1))
        X = self._xbow_scaler.transform(X)
        
        elapsed = timeit.default_timer() - start_time
        self._logger.info('Total elapsed time to create/scale list/numpy array: {}'.format(str(timedelta(seconds=elapsed))))

        start_time = timeit.default_timer()

        self._logger.info('Making prediction on emotion...')
        prediction = self._xbow_model.predict(X)
        label = self._xbow_encoder.inverse_transform(prediction)
        if self._sessionLogging:
            #print(feature_file)
            #print(os.path.basename(feature_file))
            #print(os.path.splitext(os.path.basename(feature_file)))
            timestamp, _ = os.path.splitext(os.path.basename(feature_file))
            #print(timestamp)
            with open(self._sessionRoot + '/' + 'emotions.csv', 'ab') as f:
                f.write(','.join([timestamp, 'openXBOW', event_type, str(prediction[0]), label[0], '\n']))

        self._logger.info('Emotion predicted: {}'.format((prediction, label)))

        elapsed = timeit.default_timer() - start_time
        self._logger.info('Total elapsed time to make a prediction: {}'.format(str(timedelta(seconds=elapsed))))

        elapsed = timeit.default_timer() - func_start_time
        self._logger.info('Total elapsed time for function: {}'.format(str(timedelta(seconds=elapsed))))

        return (prediction, label)

