import logging
#import jasperpath

import subprocess
import pickle
import numpy as np
import pandas as pd
import arff as liacarff
import timeit
from datetime import timedelta

class Emotion(object):

    def __init__(self):

        self._logger = logging.getLogger(__name__)
        self._model = pickle.load(open('Model.pkl', 'rb'))
        self._scaler = pickle.load(open('Scaler.pkl', 'rb'))
        self._encoder = pickle.load(open('Encoder.pkl', 'rb'))
        # TODO: Use jasper config (yaml) file to retrieve openSMILE path/bin location

        self._opensmile_path = '/home/pi/openSMILE/opensmile-2.3.0/'
        self._bin_path = 'inst/bin/SMILExtract'
        self._config_path = 'config/ComParE_2016.conf'
        
    def featurise(self, input, output):
        
        args = [self._opensmile_path + self._bin_path, ' -noconsoleoutput 1 -appendarff 0',
                ' -C ', self._opensmile_path + self._config_path,
                ' -I ', input,
                ' -O ', output]

        self._logger.debug(args)
        cmd = ''.join(args)
        #print(cmd)

        start_time = timeit.default_timer()

        p = subprocess.Popen(cmd, shell=True, bufsize=-1)
        (output, err) = p.communicate()
        p_status = p.wait()
        #print(p_status)
        #ret = subprocess.check_call(args, shell=True)

        elapsed = timeit.default_timer() - start_time
        self._logger.info('Total elapsed time for openSMILE feature extraction: {}'.format(str(timedelta(seconds=elapsed))))

        return p_status

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

    def predict_no_pandas(self, feature_file):
        
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
        self._logger.info('Emotion predicted: {}'.format((prediction, label)))

        elapsed = timeit.default_timer() - start_time
        self._logger.info('Total elapsed time to make a prediction: {}'.format(str(timedelta(seconds=elapsed))))

        elapsed = timeit.default_timer() - func_start_time
        self._logger.info('Total elapsed time for function: {}'.format(str(timedelta(seconds=elapsed))))

        return (prediction, label)
