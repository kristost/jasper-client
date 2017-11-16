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

    def __init__(self, sessionLogging=False, sessionRoot='~'):

        self._logger = logging.getLogger(__name__)
        self._model = pickle.load(open('Model.pkl', 'rb'))
        self._scaler = pickle.load(open('Scaler.pkl', 'rb'))
        self._encoder = pickle.load(open('Encoder.pkl', 'rb'))
        # TODO: Use jasper config (yaml) file to retrieve openSMILE path/bin location

        self._opensmile_path = '/home/pi/openSMILE/opensmile-2.3.0/'
        self._bin_path = 'inst/bin/SMILExtract'
        self._config_path = 'config/ComParE_2016.conf'

        self._sessionLogging = sessionLogging
        if self._sessionLogging:
            homedir = os.path.expanduser(sessionRoot)
            self._sessionRoot = homedir + '/' + 'session_' + datetime.today().strftime('%d%m%YT%H%M')

            if not os.path.exists(self._sessionRoot):
                self._logger.info('Creating session directory: "{}"'.format(self._sessionRoot))
                os.makedirs(self._sessionRoot)
        

    def featurise(self, input, output):

        if self._sessionLogging == True:
            timestamp = jasperpath.get_timestamp()
            output = self._sessionRoot + '/' + timestamp + '.arff'
        
        args = [self._opensmile_path + self._bin_path, ' -noconsoleoutput 1 -appendarff 0',
                ' -C ', self._opensmile_path + self._config_path,
                ' -I ', input,
                ' -O ', output]

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
        
        return (p_status, output)

    def predict(self, feature_file):
        
        func_start_time = timeit.default_timer()

        start_time = timeit.default_timer()

        self._logger.info('Loading data from ARFF file {}'.format(feature_file))

        data = liacarff.load(open(feature_file))
        self._logger.info('ARFF file loaded.')

        elapsed = timeit.default_timer() - start_time
        self._logger.info('Total elapsed time to load ARFF: {}'.format(str(timedelta(seconds=elapsed))))

        start_time = timeit.default_timer()

        self._logger.info('Creating pandas DataFrame.')
        attributes = ['{}'.format(a[0]) for a in data['attributes']]
        df = pd.DataFrame(data=data['data'], columns=attributes)
        df.drop(['name', 'class'], axis=1, inplace=True)

        X = self._scaler.transform(df)

        elapsed = timeit.default_timer() - start_time
        self._logger.info('Total elapsed time to create pandas DataFrame: {}'.format(str(timedelta(seconds=elapsed))))
        
        start_time = timeit.default_timer()

        self._logger.info('Making prediction on emotion...')
        prediction = self._model.predict(X)
        label = self._encoder.inverse_transform(prediction)
        self._logger.info('Emotion predicted: {}'.format((prediction, label)))

        elapsed = timeit.default_timer() - start_time
        self._logger.info('Total elapsed time to make a prediction: {}'.format(str(timedelta(seconds=elapsed))))

        elapsed = timeit.default_timer() - func_start_time
        self._logger.info('Total elapsed time for function: {}'.format(str(timedelta(seconds=elapsed))))

        return (prediction, label)

    def predict_no_pandas(self, feature_file, event_type):
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

        if len(X) == 6375:
            X = X[1:-1]
            print('Features truncated to {}'.format(len(X)))
        elif len(X) == 6374:
            X = X[1:]
            print('Features truncated to {}'.format(len(X)))

        X = np.reshape(X, (1,-1))
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
                f.write(','.join([timestamp, event_type, str(prediction[0]), label[0], '\n']))

        self._logger.info('Emotion predicted: {}'.format((prediction, label)))

        elapsed = timeit.default_timer() - start_time
        self._logger.info('Total elapsed time to make a prediction: {}'.format(str(timedelta(seconds=elapsed))))

        elapsed = timeit.default_timer() - func_start_time
        self._logger.info('Total elapsed time for function: {}'.format(str(timedelta(seconds=elapsed))))

        return (prediction, label)
