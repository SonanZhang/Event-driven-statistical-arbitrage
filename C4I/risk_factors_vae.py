from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mse
from C4I import benchmarks, clustering, evaluation, investment, risk_factors2, utils
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from numpy import corrcoef
from pandas import DataFrame
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dropout,Input,Dense
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from keras import Model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mse
import keras



def CreateRiskFactors(dfs, n_pcs, gran_names=None, threshold=0.5, SEED=1):
    '''
    From datasets extract the risk factors using VAE.
    INPUT:
        data: datasets
        n_pcs: number of Principal Components to consider
        
        threshold: float, correlation threshold for the multicollinearity filter
        SEED: int, seed to set random values
    OUTPUT:
        RFs: dataset containing the extracted risk factors
    '''
    from numpy import corrcoef
    from pandas import DataFrame
    import tensorflow as tf
    from tensorflow import keras

    

    #----- Compute Encoded Representations using Autoencoders (NNPCA)
    Encoded = list()
    for df, name in zip(dfs, gran_names):
        input_dim = df.shape[1]


        epochs = 50
        epsilon_std = 1

        # Encoder
        inputs = Input(shape=(input_dim,))
        h = Dense(64, activation='relu')(inputs)
        z_mean = Dense(n_pcs)(h)
        z_log_var = Dense(n_pcs)(h)

        # Sampling from the distribution
        def sampling(args):
            z_mean, z_log_var = args
            batch = K.shape(z_mean)[0]
            dim = K.int_shape(z_mean)[1]
            epsilon = K.random_normal(shape=(batch, dim), mean=0., stddev=epsilon_std)
            return z_mean + K.exp(0.5 * z_log_var) * epsilon

        z = Lambda(sampling, output_shape=(n_pcs,))([z_mean, z_log_var])

        # Decoder
        decoder_h = Dense(64, activation='relu')
        decoder_mean = Dense(input_dim, activation='linear')
        h_decoded = decoder_h(z)
        x_decoded_mean = decoder_mean(h_decoded)

        # VAE model
        vae = Model(inputs, x_decoded_mean)

        # Loss
        reconstruction_loss = mse(inputs, x_decoded_mean)
        reconstruction_loss *= input_dim
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        vae_loss = K.mean(reconstruction_loss + kl_loss)

        vae.add_loss(vae_loss)
        vae.compile(optimizer='rmsprop')


        vae.fit(df.values, epochs=epochs)
        encoder = Model(inputs, z_mean)
        encoded_data = encoder.predict(dfs[0].values)

        Encoded.append(DataFrame(encoded_data,
                     index=dfs[0].index,
                     columns=[f'{name}_{k}th' for k in range(1, n_pcs+1)]))

    #----- Apply multicollinearity filter to obtain the final Risk Factors
    RFs = Encoded[0]
    for Enc in Encoded:
        for n, colM in enumerate(Enc.columns):
            corr = 0
            for colD in RFs.columns:
                temp = abs(corrcoef(RFs[colD].values, Enc[colM].values)[0,1])
                if temp > corr:
                    corr = temp
            if corr < threshold:
                RFs[colM] = Enc[colM]

    return RFs
