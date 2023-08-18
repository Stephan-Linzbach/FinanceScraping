import glob
import pandas as pd
import json

batch_size = 32
seq_len = 256

d_k = 256
d_v = 256
n_heads = 12
ff_dim = 256

class Time2Vector(Layer):
  def __init__(self, seq_len, **kwargs):
    super(Time2Vector, self).__init__()
    self.seq_len = seq_len

  def build(self, input_shape):
    '''Initialize weights and biases with shape (batch, seq_len)'''
    self.weights_linear = self.add_weight(name='weight_linear',
                                shape=(int(self.seq_len),),
                                initializer='uniform',
                                trainable=True)
    
    self.bias_linear = self.add_weight(name='bias_linear',
                                shape=(int(self.seq_len),),
                                initializer='uniform',
                                trainable=True)
    
    self.weights_periodic_1 = self.add_weight(name='weight_periodic_1',
                                shape=(int(self.seq_len),),
                                initializer='uniform',
                                trainable=True)

    self.bias_periodic_1 = self.add_weight(name='bias_periodic_1',
                                shape=(int(self.seq_len),),
                                initializer='uniform',
                                trainable=True)
    
    self.weights_periodic_2 = self.add_weight(name='weight_periodic_2',
                                shape=(int(self.seq_len),),
                                initializer='uniform',
                                trainable=True)

    self.bias_periodic_2 = self.add_weight(name='bias_periodic_2',
                                shape=(int(self.seq_len),),
                                initializer='uniform',
                                trainable=True)
    self.weights_periodic_3 = self.add_weight(name='weight_periodic_3',
                                shape=(int(self.seq_len),),
                                initializer='uniform',
                                trainable=True)

    self.bias_periodic_3 = self.add_weight(name='bias_periodic_3',
                                shape=(int(self.seq_len),),
                                initializer='uniform',
                                trainable=True)
  def call(self, x):
    '''Calculate linear and periodic time features'''
    x = tf.math.reduce_mean(x[:,:,:4], axis=-1) 
    time_linear = self.weights_linear * x + self.bias_linear # Linear time feature
    time_linear = tf.expand_dims(time_linear, axis=-1) # Add dimension (batch, seq_len, 1)
    
    time_periodic_1 = tf.math.sin(tf.multiply(x, self.weights_periodic_1) + self.bias_periodic_1)
    time_periodic_1 = tf.expand_dims(time_periodic_1, axis=-1) # Add dimension (batch, seq_len, 1)

    time_periodic_2 = tf.math.sin(tf.multiply(x, self.weights_periodic_2) + self.bias_periodic_2)
    time_periodic_2 = tf.expand_dims(time_periodic_2, axis=-1) # Add dimension (batch, seq_len, 1)
    
    time_periodic_3 = tf.math.sin(tf.multiply(x, self.weights_periodic_3) + self.bias_periodic_3)
    time_periodic_3 = tf.expand_dims(time_periodic_3, axis=-1) # Add dimension (batch, seq_len, 1)
    return tf.concat([time_linear, time_periodic_1, time_periodic_2, time_periodic_3], axis=-1) # shape = (batch, seq_len, 2)
   
  def get_config(self): # Needed for saving and loading model with custom layer
    config = super().get_config().copy()
    config.update({'seq_len': self.seq_len})
    return config

class SingleAttention(Layer):
  def __init__(self, d_k, d_v):
    super(SingleAttention, self).__init__()
    self.d_k = d_k
    self.d_v = d_v

  def build(self, input_shape):
    self.query = Dense(self.d_k, 
                       input_shape=input_shape, 
                       kernel_initializer='glorot_uniform', 
                       bias_initializer='glorot_uniform')
    
    self.key = Dense(self.d_k, 
                     input_shape=input_shape, 
                     kernel_initializer='glorot_uniform', 
                     bias_initializer='glorot_uniform')
    
    self.value = Dense(self.d_v, 
                       input_shape=input_shape, 
                       kernel_initializer='glorot_uniform', 
                       bias_initializer='glorot_uniform')

  def call(self, inputs): # inputs = (in_seq, in_seq, in_seq)
    q = self.query(inputs[0])
    k = self.key(inputs[1])

    attn_weights = tf.matmul(q, k, transpose_b=True)
    attn_weights = tf.map_fn(lambda x: x/np.sqrt(self.d_k), attn_weights)
    attn_weights = tf.nn.softmax(attn_weights, axis=-1)
    
    v = self.value(inputs[2])
    attn_out = tf.matmul(attn_weights, v)
    return attn_out    

#############################################################################

class MultiAttention(Layer):
  def __init__(self, d_k, d_v, n_heads):
    super(MultiAttention, self).__init__()
    self.d_k = d_k
    self.d_v = d_v
    self.n_heads = n_heads
    self.attn_heads = list()

  def build(self, input_shape):
    for n in range(self.n_heads):
      self.attn_heads.append(SingleAttention(self.d_k, self.d_v))  
    
    # input_shape[0]=(batch, seq_len, 7), input_shape[0][-1]=7 
    self.linear = Dense(input_shape[0][-1], 
                        input_shape=input_shape, 
                        kernel_initializer='glorot_uniform', 
                        bias_initializer='glorot_uniform')

  def call(self, inputs):
    attn = [self.attn_heads[i](inputs) for i in range(self.n_heads)]
    concat_attn = tf.concat(attn, axis=-1)
    multi_linear = self.linear(concat_attn)
    return multi_linear   

#############################################################################

class TransformerEncoder(Layer):
  def __init__(self, d_k, d_v, n_heads, ff_dim, dropout=0.1, **kwargs):
    super(TransformerEncoder, self).__init__()
    self.d_k = d_k
    self.d_v = d_v
    self.n_heads = n_heads
    self.ff_dim = ff_dim
    self.attn_heads = list()
    self.dropout_rate = dropout

  def build(self, input_shape):
    self.attn_multi = MultiAttention(self.d_k, self.d_v, self.n_heads)
    self.attn_dropout = Dropout(self.dropout_rate)
    self.attn_normalize = LayerNormalization(input_shape=input_shape, epsilon=1e-6)

    self.ff_conv1D_1 = Conv1D(filters=self.ff_dim, kernel_size=1, activation='relu')
    # input_shape[0]=(batch, seq_len, 7), input_shape[0][-1] = 7 
    self.ff_conv1D_2 = Conv1D(filters=input_shape[0][-1], kernel_size=1) 
    self.ff_dropout = Dropout(self.dropout_rate)
    self.ff_normalize = LayerNormalization(input_shape=input_shape, epsilon=1e-6)    
  
  def call(self, inputs): # inputs = (in_seq, in_seq, in_seq)
    attn_layer = self.attn_multi(inputs)
    attn_layer = self.attn_dropout(attn_layer)
    attn_layer = self.attn_normalize(inputs[0] + attn_layer)

    ff_layer = self.ff_conv1D_1(attn_layer)
    ff_layer = self.ff_conv1D_2(ff_layer)
    ff_layer = self.ff_dropout(ff_layer)
    ff_layer = self.ff_normalize(inputs[0] + ff_layer)
    return ff_layer 

  def get_config(self): # Needed for saving and loading model with custom layer
    config = super().get_config().copy()
    config.update({'d_k': self.d_k,
                   'd_v': self.d_v,
                   'n_heads': self.n_heads,
                   'ff_dim': self.ff_dim,
                   'attn_heads': self.attn_heads,
                   'dropout_rate': self.dropout_rate})
    return config

def load(c):
    with open(c) as f:
        data = json.load(f)
    return data

def sort_clean(df):
    df['volume'].replace(to_replace=0, method='ffill', inplace=True) 
    df.sort_values('timestamp', inplace=True)
    return df

def real(df):
  if df['open'] * 0.99 <= df['low'] and  df['open'] * 1.01 >= df['high']:
    return 0
  elif  df['open'] * 1.01 <= df['high']:
    return 1
  else:
    return 2
  return nan

def create_train_test_val(df):
    df['change'] = df.apply(real, axis = 1)
    df['open'] = df['open'].pct_change() # Create arithmetic returns column
    df['high'] = df['high'].pct_change() # Create arithmetic returns column
    df['low'] = df['low'].pct_change() # Create arithmetic returns column
    df['close'] = df['close'].pct_change() # Create arithmetic returns column
    df['volume'] = df['volume'].pct_change()

    df.dropna(how='any', axis=0, inplace=True) # Drop all rows with NaN values

    ###############################################################################
    '''Normalize price columns'''

    min_return = min(df[['open', 'high', 'low', 'close']].min(axis=0))
    max_return = max(df[['open', 'high', 'low', 'close']].max(axis=0))

    # Min-max normalize price columns (0-1 range)
    df['open'] = (df['open'] - min_return) / (max_return - min_return)
    df['high'] = (df['high'] - min_return) / (max_return - min_return)
    df['low'] = (df['low'] - min_return) / (max_return - min_return)
    df['close'] = (df['close'] - min_return) / (max_return - min_return)

    ###############################################################################
    '''Normalize volume column'''

    min_volume = df['volume'].min(axis=0)
    max_volume = df['volume'].max(axis=0)

    # Min-max normalize volume columns (0-1 range)
    df['volume'] = (df['volume'] - min_volume) / (max_volume - min_volume)

    ###############################################################################
    '''Create training, validation and test split'''

    times = sorted(df.index.values)
    last_10pct = sorted(df.index.values)[-int(0.1*len(times))] # Last 10% of series
    last_20pct = sorted(df.index.values)[-int(0.2*len(times))] # Last 20% of series

    df_train = df[(df.index < last_20pct)]  # Training data are 80% of total data
    df_val = df[(df.index >= last_20pct) & (df.index < last_10pct)]
    df_test = df[(df.index >= last_10pct)]

    # Remove date column
    df_train.drop(columns=['timestamp'], inplace=True)
    df_val.drop(columns=['timestamp'], inplace=True)
    df_test.drop(columns=['timestamp'], inplace=True)

    # Convert pandas columns into arrays
    train_data = df_train.values
    val_data = df_val.values
    test_data = df_test.values
    return (train_data, val_data, test_data)

def create_label(num):
    label = np.zeros(3)
    label[int(num)] = 1
    return label

def create_X_Y(data):
    X, y= [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i]) # Chunks of training data with a length of 128 df-rows
        y.append(create_label(data[:, 5][i])) #Value of 4th column (Close Price) of df-row 128+1
    X, y = np.array(X), np.array(y)
    return (X, y)

def fit(train, val):
    X_train, y_train = train
    X_val, y_val = val
    with tf.device('/device:GPU:0'):
        history = model.fit(X_train, y_train,
                          batch_size=batch_size,
                          epochs=35,
                          callbacks=[callback],
                          validation_data=(X_val, y_val))

all_files = glob.glob("DAY/*")
batchsize = 50

for i in range(len(batch_size, all_files, batchsize)):
    current_files = all_files[i-batchsize:i]

    dataframes = [load(c) for c in current_files]

    dataframes = [sort_clean(d) for d in dataframes]

    data = [create_train_test_val(d) for d in dataframes]

    Xytrain = [create_X_Y(d[0]) for d in data]
    Xyval  = [create_X_Y(d[1]) for d in data]
    Xytest = [create_X_Y(d[2]) for d in data]
    
    model = create_model()
    model.summary()

    callback = tf.keras.callbacks.ModelCheckpoint('Transformer+TimeEmbedding.hdf5',
                                                  monitor='val_loss',
                                                  save_best_only=True, verbose=1)

    history = [fit(t, v) for t, v in zip(Xytrain, Xyval)]
