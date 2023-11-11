from C4I import benchmarks, clustering, evaluation, investment, risk_factors2, utils
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from numpy import corrcoef
from pandas import DataFrame
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dropout, Input, Dense
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

from sklearn.decomposition import PCA
from numpy import corrcoef
from pandas import DataFrame





def CreateRiskFactors(dfs, n_pcs, gran_names=None, threshold=0.5, SEED=1):
    from numpy import corrcoef
    from pandas import DataFrame
    import tensorflow as tf
    from tensorflow import keras

    # ----- Compute Principal Components
    PCs = list()
    for df, name in zip(dfs, gran_names):
        pca_temp = PCA(n_components=n_pcs, random_state=SEED)
        pca_temp.fit(df.values)
        PCs.append(DataFrame(pca_temp.transform(dfs[0].values),
                             index=dfs[0].index,
                             columns=[f'{name}_{k}th' \
                                      for k in range(1, n_pcs + 1)]))

    # ----- Compute Encoded Representations using Autoencoders (NNPCA)
    Encoded = list()
    for df, name in zip(dfs, gran_names):
        input_dim = df.shape[1]

        input_layer = Input(shape=(input_dim,))
        encoded_1 = Dense(1024, activation='tanh', activity_regularizer=regularizers.l1(10 ** (-5)))(input_layer)
        encoded_1 = Dropout(0.3)(encoded_1)
        encoded_2 = Dense(512, activation='relu', activity_regularizer=regularizers.l1(10 ** (-5)))(encoded_1)
        encoded_2 = Dropout(0.3)(encoded_2)
        encoded_final = Dense(n_pcs, activation='relu', activity_regularizer=regularizers.l1(10 ** (-5)))(encoded_2)

        decoded_1 = Dense(512, activation='relu', activity_regularizer=regularizers.l1(10 ** (-5)))(encoded_final)
        decoded_1 = Dropout(0.3)(decoded_1)
        decoded_2 = Dense(1024, activation='relu', activity_regularizer=regularizers.l1(10 ** (-5)))(decoded_1)
        decoded_2 = Dropout(0.3)(decoded_2)
        decoded_final = Dense(input_dim, activation='tanh', activity_regularizer=regularizers.l1(10 ** (-5)))(decoded_2)

        autoencoder = Model(input_layer, decoded_final)
        encoder = Model(input_layer, encoded_final)

        early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        autoencoder.fit(df.values, df.values, epochs=150, batch_size=32, shuffle=True, callbacks=[early_stopping],
                        verbose=0)

        encoded_data = encoder.predict(dfs[0].values)
        Encoded.append(DataFrame(encoded_data,
                                 index=dfs[0].index,
                                 columns=[f'NN{name}_{k}th' for k in range(1, n_pcs + 1)]))

    # ----- Apply multicollinearity filter to obtain the final Risk Factors
    RFs_1 = Encoded[0]
    for Enc in Encoded:
        for n, colM in enumerate(Enc.columns):
            corr = 0
            for colD in RFs_1.columns:
                temp = abs(corrcoef(RFs_1[colD].values, Enc[colM].values)[0, 1])
                if temp > corr:
                    corr = temp
            if corr < threshold:
                RFs_1[colM] = Enc[colM]

    drop_col = list()
    for col in RFs_1.columns:
        if not (RFs_1[col] != 0).any(axis=0):
            drop_col.append(col)
    RFs_1 = RFs_1.drop(columns=drop_col)

    RFs_2 = PCs[0]
    for PC in PCs:
        for n, colM in enumerate(PC.columns):
            corr = 0
            for colD in RFs_2.columns:
                temp = abs(corrcoef(RFs_2[colD].values, PC[colM].values)[0, 1])
                if temp > corr:
                    corr = temp
            if corr < 0.5:
                RFs_2[colM] = PC[colM]

    RFs = pd.concat([RFs_1, RFs_2], axis= 1)

    return RFs