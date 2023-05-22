from tensorflow.keras.layers import Dense,LSTM,Dropout, Concatenate
from tensorflow.keras import Sequential,Input, optimizers, losses, models, callbacks


def get_ann_model(inp_shape):
    '''
    Returns ANN with dense layer only

    Parameters
    -----------
    inp_shape : input shape 
    
    Returns
    --------
    model : ANN model
    '''
    #building model
    inp = Input(inp_shape)
    dense = Dense(20, activation='selu')(inp)
    dense = Dense(20, activation='selu')(dense)
    dense = Dense(20, activation='selu')(dense)
    dense = Dense(20, activation='selu')(dense)
    dense = Dense(20, activation='selu')(dense)
    dense = Dense(20, activation='selu')(dense)
    output = Dense(1, activation='linear')(dense)
    model = models.Model(inp,output,name="ANN")

    #compiling model
    model.compile(loss='mean_absolute_percentage_error', optimizer='adam')
    
    return model