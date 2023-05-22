from tensorflow.keras.layers import Dense,LSTM,Dropout, Concatenate, LeakyReLU
from tensorflow.keras import Sequential,Input, optimizers, losses, models, callbacks

def get_lstm_model_ensemble(inp_shape):
    '''
    This function returns model with ensemble.

    We are utilizing two models with different activation
    fucntion. These two models will learn those features
    differently and make prediction.

    Parameters
    -----------
    inp_shape : input shape 
    
    Returns
    --------
    model : lstm model
    '''
    ##building model##
    #input
    inp = Input(inp_shape)

    #model 1
    lstm = LSTM(20, input_shape=inp_shape,return_sequences=True)(inp)
    lstm = LSTM(20, input_shape=inp_shape)(lstm)
    dense = Dense(20, activation='selu')(lstm)
    dense = Dense(20, activation='selu')(dense)
    dense = Dense(20, activation='selu')(dense)
    dense_end = Dense(20, activation='selu')(dense)

    #model 2
    lstm2 = LSTM(20, input_shape=inp_shape,return_sequences=True)(inp)
    lstm2 = LSTM(20, input_shape=inp_shape)(lstm2)
    dense2 = Dense(20, activation='elu')(lstm2)
    dense2 = Dense(20, activation='elu')(dense2)
    dense2 = Dense(20, activation='elu')(dense2)
    dense2_end = Dense(20, activation='elu')(dense2)

    #model 3
    lstm3 = LSTM(20, input_shape=inp_shape,return_sequences=True)(inp)
    lstm3 = LSTM(20, input_shape=inp_shape)(lstm3)
    dense3 = Dense(20, kernel_initializer='random_normal', activation='gelu')(lstm3)
    dense3 = Dense(20, kernel_initializer='random_normal', activation='gelu')(dense3)
    dense3 = Dense(20, kernel_initializer='random_normal', activation='gelu')(dense3)
    dense3_end = Dense(20, kernel_initializer='random_normal',activation='gelu')(dense3)


    #ensembling and model output
    add = Concatenate()([dense_end,dense2_end,dense3_end])
    output = Dense(1, activation='linear')(add)
    model = models.Model(inp,output,name="LSTM_DENSE_ENSEMBLE")

    #compiling model
    model.compile(loss='mean_absolute_percentage_error', optimizer='adam')

    return model