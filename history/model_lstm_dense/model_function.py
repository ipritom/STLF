from tensorflow.keras.layers import Dense,LSTM,Dropout, Concatenate
from tensorflow.keras import Sequential,Input, optimizers, losses, models, callbacks


def get_lstm_model(inp_shape):
    '''
    Parameters
    -----------
    inp_shape : input shape 
    
    Returns
    --------
    model : lstm model
    '''
    #building model
    inp = Input(inp_shape)
    lstm = LSTM(20, input_shape=inp_shape,return_sequences=True)(inp)
    lstm = LSTM(20, input_shape=inp_shape)(lstm)
    dense = Dense(20, activation='selu')(lstm)
    dense = Dense(20, activation='selu')(dense)
    dense = Dense(20, activation='selu')(dense)
    dense = Dense(20, activation='selu')(dense)
    output = Dense(1, activation='linear')(dense)
    model = models.Model(inp,output,name="LSTM_DENSE")

    #compiling model
    model.compile(loss='mean_absolute_percentage_error', optimizer='adam')

    return model