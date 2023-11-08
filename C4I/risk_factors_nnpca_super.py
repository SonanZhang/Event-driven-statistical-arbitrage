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

def CreateRiskFactors(data, n_pcs, gran_names=None, threshold=0.5, SEED=1):
    from numpy import corrcoef
    from pandas import DataFrame
    import tensorflow as tf
    from tensorflow import keras

    
    #----- Compute Encoded Representations using Autoencoders (NNPCA)
    Encoded = list()
    
    input_dim = data.shape[1]
    
    input_layer = Input(shape=(input_dim,))
    encoded_1 = Dense(1024, activation='tanh', activity_regularizer=regularizers.l1(10**(-5)))(input_layer)
    encoded_1 = Dropout(0.3)(encoded_1)
    encoded_2 = Dense(512, activation='relu', activity_regularizer=regularizers.l1(10**(-5)))(encoded_1)
    encoded_2 = Dropout(0.3)(encoded_2)
    encoded_final = Dense(n_pcs, activation='relu', activity_regularizer=regularizers.l1(10**(-5)))(encoded_2)

    decoded_1 = Dense(512, activation='relu', activity_regularizer=regularizers.l1(10**(-5)))(encoded_final)
    decoded_1 = Dropout(0.3)(decoded_1)
    decoded_2 = Dense(1024, activation='relu', activity_regularizer=regularizers.l1(10**(-5)))(decoded_1)
    decoded_2 = Dropout(0.3)(decoded_2)
    decoded_final = Dense(input_dim, activation='tanh', activity_regularizer=regularizers.l1(10**(-5)))(decoded_2)

    autoencoder = Model(input_layer, decoded_final)
    encoder = Model(input_layer, encoded_final)
    
    early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    autoencoder.fit(data.values, data.values, epochs=150, batch_size=32, shuffle=True, callbacks=[early_stopping], verbose=0)

    encoded_data = encoder.predict(data.values)
    Encoded.append(DataFrame(encoded_data,
                             index=data.index,
                             columns=[f'{k}th' for k in range(1, n_pcs+1)]))

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