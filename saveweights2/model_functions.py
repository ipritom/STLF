from tensorflow.keras.layers import Dense,LSTM,Dropout, Average, LeakyReLU,Concatenate
from tensorflow.keras import Sequential,Input, optimizers, losses, models, callbacks, utils


def get_lstm_model(inp_shape):
    '''
    Parameters
    -----------
    inp_shape : input shape 
    
    Returns
    --------
    model : lstm model
    '''
    n_h = 200 #number of neurons in hidden layer
    #building model
    inp = Input(inp_shape)
    lstm = LSTM(n_h, input_shape=inp_shape,return_sequences=True)(inp)
    lstm = LSTM(n_h, input_shape=inp_shape)(lstm)
    dense = Dense(n_h, activation='selu')(lstm)
    dense = Dense(n_h, activation='selu')(dense)
    dense = Dense(n_h, activation='selu')(dense)
    dense = Dense(n_h, activation='selu')(dense)
    dropout = Dropout(0.2)(dense)
    output = Dense(24, activation='linear')(dropout)
    model = models.Model(inp,output,name="LSTM_DENSE")

    #compiling model
    model.compile(loss='mean_absolute_percentage_error', optimizer='adam')

    return model


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
    n_h = 200 #number of neurons in hidden layer
    #input
    inp = Input(inp_shape)

    #model 1
    lstm = LSTM(n_h, input_shape=inp_shape,return_sequences=True)(inp)
    lstm = LSTM(n_h, input_shape=inp_shape)(lstm)
    dense = Dense(n_h, activation='selu')(lstm)
    dense = Dense(n_h, activation='selu')(dense)
    dense = Dense(n_h, activation='selu')(dense)
    dense_end = Dense(n_h, activation='selu')(dense)

    #model 2
    lstm2 = LSTM(n_h, input_shape=inp_shape,return_sequences=True)(inp)
    lstm2 = LSTM(n_h, input_shape=inp_shape)(lstm2)
    dense2 = Dense(n_h, activation='elu')(lstm2)
    dense2 = Dense(n_h, activation='elu')(dense2)
    dense2 = Dense(n_h, activation='elu')(dense2)
    dense2_end = Dense(n_h, activation='elu')(dense2)

    #model 3
    lstm3 = LSTM(n_h, input_shape=inp_shape,return_sequences=True)(inp)
    lstm3 = LSTM(n_h, input_shape=inp_shape)(lstm3)
    dense3 = Dense(n_h, kernel_initializer='random_normal', activation='gelu')(lstm3)
    dense3 = Dense(n_h, kernel_initializer='random_normal', activation='gelu')(dense3)
    dense3 = Dense(n_h, kernel_initializer='random_normal', activation='gelu')(dense3)
    dense3_end = Dense(n_h, kernel_initializer='random_normal',activation='gelu')(dense3)

    lstm4 = LSTM(n_h, input_shape=inp_shape,return_sequences=True)(inp)
    lstm4 = LSTM(n_h, input_shape=inp_shape)(lstm4)
    dense4 = Dense(n_h, kernel_initializer='random_normal', activation='softplus')(lstm4)
    dense4 = Dense(n_h, kernel_initializer='random_normal', activation='softplus')(dense4)
    dense4 = Dense(n_h, kernel_initializer='random_normal', activation='softplus')(dense4)
    dense4_end = Dense(n_h, kernel_initializer='random_normal',activation='softplus')(dense4)

    #ensembling and model output
    add = Concatenate()([dense_end,dense2_end,dense3_end,dense4_end])
    output = Dense(24, activation='linear')(add)
    model = models.Model(inp,output,name="LSTM_DENSE_ENSEMBLE")

    #compiling model
    model.compile(loss='mean_absolute_percentage_error', optimizer='adam')

    return model
