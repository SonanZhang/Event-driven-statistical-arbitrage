def CreateRiskFactors(dfs, n_pcs, gran_names=None, threshold=0.5, SEED=1):
    '''
    From datasets extract the risk factors using NNPCA.
    INPUT:
        dfs: list of datasets at different time granularities
        n_pcs: number of Principal Components to consider
        gran_names: list of strings containing the name of each granularity,
            same dimension of dfs
        threshold: float, correlation threshold for the multicollinearity filter
        SEED: int, seed to set random values
    OUTPUT:
        RFs: dataset containing the extracted risk factors
    '''
    from numpy import corrcoef
    from pandas import DataFrame
    import tensorflow as tf
    from tensorflow import keras

    if gran_names == None:
        gran_names = list()
        for df in range(len(dfs)): gran_names.append(f'{df}')
        print(gran_names)

    #----- Compute Encoded Representations using Autoencoders (NNPCA)
    Encoded = list()
    for df, name in zip(dfs, gran_names):
        input_dim = df.shape[1]

        # Define a deeper autoencoder architecture
        input_layer = keras.layers.Input(shape=(input_dim,))
        encoded_1 = keras.layers.Dense(128, activation='tanh')(input_layer)
        encoded_2 = keras.layers.Dense(64, activation='tanh')(encoded_1)
        encoded_final = keras.layers.Dense(n_pcs, activation='tanh')(encoded_2)

        # Symmetric decoder architecture
        decoded_1 = keras.layers.Dense(64, activation='tanh')(encoded_final)
        decoded_2 = keras.layers.Dense(128, activation='tanh')(decoded_1)
        decoded_final = keras.layers.Dense(input_dim, activation='linear')(decoded_2)

        autoencoder = keras.Model(input_layer, decoded_final)
        encoder = keras.Model(input_layer, encoded_final)

        autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        autoencoder.fit(df.values, df.values, epochs=100, batch_size=50, shuffle=True)
        
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
