#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 13:17:28 2024

@author: frankdietz
"""

###############################################################################
#REQUIRED PACKAGES 
###############################################################################

import sys  

import tensorflow as tf 
import tensorflow_probability as tfp
from tensorflow.keras import layers
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras import regularizers
import random

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import os


#file paths
file_path = '/Users/frankdietz/Desktop/Thesis/data/Characteristics_Returns.csv'
file_path_RF = '/Users/frankdietz/Desktop/Thesis/data/RF.csv'
path = '/Users/frankdietz/Desktop/Thesis/'

sys.path.append(path)  

###############################################################################
#DATA SET PREPARATION
###############################################################################

#Entire data set 
df_all = pd.read_csv(file_path)
df_RF = pd.read_csv(file_path_RF)
del file_path_RF, file_path

yyyymm_min = df_all['yyyymm'].min()
yyyymm_max = df_all['yyyymm'].max()

print(f"Minimum yyyymm: {yyyymm_min}")
print(f"Maximum yyyymm: {yyyymm_max}")


#Risk free rate adjustment
df_RF['DATE'] = pd.to_datetime(df_RF['DATE'])
df_RF['DATE'] = df_RF['DATE'].dt.strftime('%Y%m')
df_RF.rename(columns={'DATE': 'yyyymm'}, inplace=True)
df_RF['yyyymm'] = df_RF['yyyymm'].astype(str)

df_all['yyyymm'] = df_all['yyyymm'].astype(str)
df_merged = pd.merge(df_all, df_RF, on='yyyymm')
df_merged['monthly_risk_free_rate'] = df_merged['TB3MS'].apply(lambda x: (1 + x)**(1/12) - 1)
df_merged['ret_ad'] = df_merged['ret'] - df_merged['monthly_risk_free_rate']

pd.set_option('display.max_columns', None) 

df_all = df_merged 

cutoff_date = '201008'

#Sorting and matrix creation
df_all.sort_values(by=['yyyymm', 'permno'], inplace=True)

# Filter df_all into two datasets
df_train_val = df_all[df_all['yyyymm'] <= cutoff_date]
df_holdout = df_all[df_all['yyyymm'] > cutoff_date]

#Convert to integer
df_all['yyyymm'] = df_all['yyyymm'].astype(int)

#Get data set length
total_obs = int(len(df_all)/500)
train_val_obs = int(len(df_train_val)/500)
holdout_obs = int(len(df_holdout)/500)

del df_merged, df_RF, cutoff_date, yyyymm_max, yyyymm_min

#Returns
returns = df_train_val[['ret_ad', 'permno', 'yyyymm']]
returns = returns.drop(columns=['yyyymm', 'permno'])
returns = np.reshape(returns, (train_val_obs, 500))

returns_val = df_holdout[['ret_ad', 'permno', 'yyyymm']]
returns_val = returns_val.drop(columns=['yyyymm', 'permno'])
returns_val = np.reshape(returns_val, (holdout_obs, 500))

#Valueweighted portfolio returns 
m500 = df_train_val.groupby('yyyymm')['me'].sum().reset_index()
df_train_val = pd.merge(df_train_val, m500, on='yyyymm', suffixes=('', '_market'))
df_train_val['weight'] = df_train_val['me'] / df_train_val['me_market']
df_train_val['weighted_return'] = df_train_val['weight'] * df_train_val['ret_ad']
market_factor = df_train_val.groupby('yyyymm')['weighted_return'].sum().reset_index()
market_factor = market_factor['weighted_return']

m500 = df_holdout.groupby('yyyymm')['me'].sum().reset_index()
df_holdout = pd.merge(df_holdout, m500, on='yyyymm', suffixes=('', '_market'))
df_holdout['weight'] = df_holdout['me'] / df_holdout['me_market']
df_holdout['weighted_return'] = df_holdout['weight'] * df_holdout['ret_ad']
market_factor_val = df_holdout.groupby('yyyymm')['weighted_return'].sum().reset_index()
market_factor_val = market_factor_val['weighted_return']

#Equally weighted portfolio returns 
#market_factor = df_train_val.groupby('yyyymm')['ret_ad'].mean()
#market_factor_val = df_holdout.groupby('yyyymm')['ret_ad'].mean()

#Input values 
predictors = ['prc','me','BMdec','Investment']
nn_input = df_train_val[predictors]
nn_input = nn_input.values.reshape((train_val_obs, 500, nn_input.shape[1]))
nn_input = np.array(nn_input, dtype=np.float32)
nn_input = nn_input.reshape(train_val_obs, -1)

nn_val = df_holdout[predictors]
nn_val = nn_val.values.reshape((holdout_obs, 500, nn_val.shape[1]))
nn_val = np.array(nn_val, dtype=np.float32)
nn_val = nn_val.reshape(holdout_obs, -1)

# Impute NaN values with mean
imputer = SimpleImputer(strategy='mean')
nn_input = imputer.fit_transform(nn_input)
nn_val = imputer.fit_transform(nn_val)

# Normalize input data to have zero mean and unit variance
scaler = StandardScaler()
nn_input = scaler.fit_transform(nn_input)
nn_val = scaler.fit_transform(nn_val)
del imputer, scaler 

#Train test split 
nn_train, nn_test, ret_train, ret_test, market_train, market_test,  = train_test_split(nn_input,returns,market_factor, test_size=0.5, random_state=42, shuffle=False)

#Train and test Integer 
train_obs = len(nn_train)
test_obs = len(nn_test)

#Reshaping Returns
ret_train = tf.reshape(ret_train, (train_obs, 500))
ret_train = tf.cast(ret_train, tf.float32)

#Reshaping market factor
market_train = tf.reshape(market_train, (train_obs, 1))

#Zero response
proxy_target = tf.zeros((train_obs, 500))
proxy_target = proxy_target.numpy()

seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

def set_seed(seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def reset_session(seed=42):
    set_seed(seed)
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)


###############################################################################
#NEURAL NETWORK
###############################################################################

#FOLD ONE 
###############################################################################

shape = nn_train.shape[1]

def train_model(nn_train, proxy_target, ret_train, market_train, input_shape, batch_size=0.01, epochs=10, learning_rate=0.01, l1=0.01, l2=0.01, seed=42):
    reset_session(seed)

    # Custom loss function
    def loss(_, y_pred):
        batch_size = tf.shape(y_pred)[0]
        
        def transform_way_1(x):
            return -50 * tf.exp(-8 * x)

        def transform_way_2(x):
            return -50 * tf.exp(8 * x)
    
        def softmax(x):
            return tf.nn.softmax(x)
    
        # Transformations
        transformed_way_1 = transform_way_1(y_pred)
        transformed_way_2 = transform_way_2(y_pred)
        
        softmax_way_1 = softmax(transformed_way_1)
        softmax_way_2 = softmax(transformed_way_2)
    
        # Calculate result
        result = softmax_way_1 - softmax_way_2
        
        
        weighted_returns = result * ret_train[:batch_size]
        weighted_returns = tf.reduce_sum(weighted_returns, axis=1, keepdims=True)
        
        market_tensor_float32 = tf.cast(market_train[:batch_size], tf.float32)
        weighted_returns_float32 = tf.cast(weighted_returns, tf.float32)
        
        matrix = tf.concat([weighted_returns_float32, market_tensor_float32], axis=1)
        row_averages = tf.reduce_mean(matrix, axis=0)
        row_averages_column = tf.reshape(row_averages, shape=(-1, 1))
        row_averages_row = tf.transpose(row_averages_column)
        
        matrix_float32 = tf.cast(matrix, dtype=tf.float32)
        cov_matrix = tfp.stats.covariance(matrix_float32, sample_axis=0, event_axis=-1)
        inv_cov_matrix = tf.linalg.inv(cov_matrix)
        
        result_step1 = tf.matmul(row_averages_row, inv_cov_matrix)
        final_result = tf.matmul(result_step1, row_averages_column)
        scalar_result = tf.squeeze(final_result)
        
        l1_reg_term = tf.add_n([tf.reduce_sum(tf.abs(v)) for v in model.trainable_weights])
        l2_reg_term = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_weights])
        
        loss_value = tf.exp(-scalar_result) + 0.2 * l1_reg_term + 20 * l2_reg_term
        return loss_value

    # Define the model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(input_shape, input_shape=(input_shape,), activation='tanh', 
                              kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)),
        tf.keras.layers.Dense(300, activation='tanh', 
                              kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)),
        tf.keras.layers.Dense(64, activation='tanh', 
                              kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)),
        tf.keras.layers.Dense(500, activation='tanh', kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)),
    ])

    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss)

    # Train the model
    history = model.fit(nn_train, proxy_target, batch_size=batch_size, epochs=epochs, shuffle=False)

    return model, history

# Example usage
# Assuming nn_train, proxy_target, ret_train, and market_train are already defined
model, history = train_model(
    nn_train=nn_train,
    proxy_target=proxy_target,
    ret_train=ret_train,
    market_train=market_train,
    input_shape=nn_train.shape[1],  # Example input dimension
    batch_size=40,
    epochs=15,
    learning_rate=0.002,
    l1=0.01/100,
    l2=0.01/100,
    seed=42
)

asset_weights_f2 = None

# Define a function to perform predictions, transformations, and additional computations
def predict_transform_and_analyze(model, nn_test, ret_test, market_test, history, path):
    # Predictions with the model
    predictions_nn = model.predict(nn_test)
    
    def transform_way_1(x):
        return -50 * tf.exp(-8 * x)

    def transform_way_2(x):
        return -50 * tf.exp(8 * x)
    
    def softmax(x):
        return tf.nn.softmax(x)
    
        # Transformations
    transformed_way_1 = transform_way_1(predictions_nn)
    transformed_way_2 = transform_way_2(predictions_nn)
        
    softmax_way_1 = softmax(transformed_way_1)
    softmax_way_2 = softmax(transformed_way_2)
    
        # Calculate result
    result = softmax_way_1 - softmax_way_2
    
    asset_weights_f2 = result
    
    # Plot training loss values
    plt.plot(history.history['loss'], color='darkslateblue', linewidth=0.5, marker="+")
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    
    plt.savefig(path + '/loss__1.png', dpi=300) 
    
    plt.show()
    


    # Returns on deep factor
    deep_factor = result * ret_test
    deep_factor = np.sum(deep_factor, axis=1)

    x = np.arange(len(nn_test))

    plt.figure(figsize=(10, 6))
    plt.plot(x, market_test, color='grey', linestyle='solid', label='Market Factor', linewidth=0.5)
    plt.bar(x, deep_factor, color='darkslateblue', linestyle='-', label='Deep Factor')
    plt.title('Deep Factor and Market Factor')
    plt.xlabel('Year')
    
    plt.xticks(np.arange(0, len(nn_test), 24), np.arange(1987, 2011, 2))  # Assuming monthly data, 24 ticks per year

    plt.ylabel('Return')
    plt.grid(False)
    plt.legend()
    
    plt.savefig(path + '/factors__1.png', dpi=300) 
    
    # Calculate correlation
    correlation_matrix = np.corrcoef(deep_factor, market_test)
    correlation = correlation_matrix[0, 1]

    return deep_factor, market_test, correlation, asset_weights_f2

# Example usage:
# Assuming model, nn_test, ret_test, market_test, and history are defined elsewhere in your script

# Fold one
deep_factor_f2, market_factor_f2, correlation_f2, asset_weights_f2 = predict_transform_and_analyze(model, nn_test, ret_test, market_test, history, path)

h1 = history.history['loss']

#FOLD TWO 
###############################################################################
nn_train, nn_test, ret_train, ret_test, market_train, market_test,  = train_test_split(nn_input,returns,market_factor, test_size=0.5, random_state=42, shuffle=False)

#Renaming to perform cross-validation
store = market_test 
market_test = market_train
market_train = store
del store 

store = nn_test
nn_test = nn_train
nn_train = store 
del store 

store = ret_test 
ret_test = ret_train
ret_train = store
del store

#Train and test Integer 
train_obs = len(nn_train)
test_obs = len(nn_test)

#Reshaping Returns
ret_train = tf.reshape(ret_train, (train_obs, 500))
ret_train = tf.cast(ret_train, tf.float32)

#Reshaping market factor
market_train = tf.reshape(market_train, (train_obs, 1))


# Example usage
# Assuming nn_train, proxy_target, ret_train, and market_train are already defined
model, history = train_model(
    nn_train=nn_train,
    proxy_target=proxy_target,
    ret_train=ret_train,
    market_train=market_train,
    input_shape=nn_train.shape[1],  # Example input dimension
    batch_size=40,
    epochs=15,
    learning_rate=0.002,
    l1=0.01/100,
    l2=0.01/100,
    seed=42
)

asset_weights_f1 = None


# Define a function to perform predictions, transformations, and additional computations
def predict_transform_and_analyze2(model, nn_test, ret_test, market_test, history,path):
    # Predictions with the model
    
    predictions_nn = model.predict(nn_test)
    
    def transform_way_1(x):
        return -50 * tf.exp(-8 * x)

    def transform_way_2(x):
        return -50 * tf.exp(8 * x)
    
    def softmax(x):
        return tf.nn.softmax(x)
    
        # Transformations
    transformed_way_1 = transform_way_1(predictions_nn)
    transformed_way_2 = transform_way_2(predictions_nn)
        
    softmax_way_1 = softmax(transformed_way_1)
    softmax_way_2 = softmax(transformed_way_2)
    
        # Calculate result
    result = softmax_way_1 - softmax_way_2
    
    asset_weights_f1 = result 
    
    # Plot training loss values
    plt.plot(history.history['loss'],color='darkslateblue',linewidth=0.5,marker="+")
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    
    plt.savefig(path + '/loss__2.png', dpi=300)  
    
    plt.show()


    # Returns on deep factor
    deep_factor = result * ret_test
    deep_factor = np.sum(deep_factor, axis=1)

    x = np.arange(len(nn_test))
    
    
# Generate date range
    plt.figure(figsize=(10, 6))
    plt.plot(x, market_test, color='grey', linestyle='solid', label='Market Factor',linewidth=0.5)
    plt.bar(x, deep_factor, color='darkslateblue', linestyle='-', label='Deep Factor')
    plt.title('Deep Factor and Market Factor')
    plt.xlabel('Year')
    
    plt.xticks(np.arange(0, len(nn_test), 24), np.arange(1965, 1988, 2))  # Assuming monthly data, 24 ticks per year

    plt.ylabel('Return')
    plt.grid(False)
    plt.legend()
    
    plt.savefig(path + '/factors__2.png', dpi=300) 
    
    # Calculate correlation
    correlation_matrix = np.corrcoef(deep_factor, market_test)
    correlation = correlation_matrix[0, 1]

    return deep_factor, market_test, correlation, asset_weights_f1

deep_factor_f1, market_factor_f1, correlation_f1, asset_weights_f1 = predict_transform_and_analyze2(model, nn_test, ret_test, market_test, history,path)

h2 = history.history['loss']

h3 = [(a + b) / 2 for a, b in zip(h1, h2)]
average_last_5_h3_1 = np.mean(h3[-5:])

#VALIDATION
###############################################################################

asset_weights_v = None

# Define a function to perform predictions, transformations, and additional computations
def predict_transform_and_analyze3(model, nn_val, returns_val, market_factor_val, history):
    # Predictions with the model
    predictions_nn = model.predict(nn_val)
    
    def transform_way_1(x):
        return -50 * tf.exp(-8 * x)

    def transform_way_2(x):
        return -50 * tf.exp(8 * x)
    
    def softmax(x):
        return tf.nn.softmax(x)
    
        # Transformations
    transformed_way_1 = transform_way_1(predictions_nn)
    transformed_way_2 = transform_way_2(predictions_nn)
        
    softmax_way_1 = softmax(transformed_way_1)
    softmax_way_2 = softmax(transformed_way_2)
    
        # Calculate result
    result = softmax_way_1 - softmax_way_2
    
    asset_weights_v = result


    # Returns on deep factor
    deep_factor = result * returns_val
    deep_factor = np.sum(deep_factor, axis=1)

    x = np.arange(len(nn_val))

    
# Generate date range
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, market_factor_val, color='grey', linestyle='solid', label='Market Factor',linewidth=0.5)
    plt.bar(x, deep_factor, color='darkslateblue', linestyle='-', label='Deep Factor')
    plt.title('Deep Factor and Market Factor')
    plt.xlabel('Year')
    
    plt.xticks(np.arange(0, len(nn_val), 24), np.arange(2010, 2022, 2))
    
    plt.xlabel('Year')

    plt.ylabel('Return')
    plt.grid(False)
    plt.legend()
    
    plt.savefig(path + '/holdout', dpi=300) 
    
    x = np.arange(len(nn_val))
    
    # Calculate correlation
    correlation_matrix = np.corrcoef(deep_factor, market_factor_val)
    correlation = correlation_matrix[0, 1]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(deep_factor, market_factor_val, c='darkslateblue', marker='+')
    plt.xlabel('Deep Factor')
    plt.ylabel('Market Factor')
    plt.title('Scatter Plot of Deep Factor vs Market Factor')
    
    plt.savefig(path + '/scatter', dpi=300) 
    
    x = np.arange(len(nn_val))

    return deep_factor, market_factor_val, correlation, asset_weights_v

deep_factor_v1, market_test_v1, correlation_fv, asset_weights_v = predict_transform_and_analyze3(model, nn_val, returns_val, market_factor_val, history)



###############################################################################
###Echo State Network
###############################################################################

#Data reshaping
input_data = np.expand_dims(nn_train, axis=0)  # Add batch dimension 
target_data = np.expand_dims(proxy_target, axis=0)  # Add batch dimension
test_data = np.expand_dims(nn_test, axis=0)  # Add batch dimension 

#Define Echo State Network 

class EchoStateNetwork(tf.keras.Model):
    def __init__(self, n_inputs, n_reservoir, n_outputs, spectral_radius=0.95, l1=0.01, l2=0.01):
        reset_session()
        super(EchoStateNetwork, self).__init__()
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.n_outputs = n_outputs
        
        self.W_in = tf.Variable(tf.random.uniform([self.n_reservoir, self.n_inputs], -0.5, 0.5))
        self.W_res = tf.Variable(tf.random.uniform([self.n_reservoir, self.n_reservoir], -0.5, 0.5))
        eigvals = tf.linalg.eigvals(self.W_res)
        self.W_res.assign(self.W_res / tf.reduce_max(tf.abs(eigvals)) * spectral_radius)

        self.W_out = layers.Dense(self.n_outputs,activation="tanh", kernel_regularizer=regularizers.L1L2(l1=l1, l2=l2))

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        time_steps = tf.shape(inputs)[1]
        state = tf.zeros([batch_size, self.n_reservoir])

        def step(t, state, states): 
            u = tf.tanh(inputs[:, t, :]) 
            state = tf.tanh(tf.matmul(u, self.W_in, transpose_b=True) + tf.matmul(state, self.W_res, transpose_b=True))
            states = states.write(t, state)
            return t + 1, state, states

        states = tf.TensorArray(dtype=tf.float32, size=time_steps)
        t = tf.constant(0)

        _, state, states = tf.while_loop(
            cond=lambda t, *_: t < time_steps,
            body=step,
            loop_vars=[t, state, states]
        )

        states = states.stack()
        states = tf.transpose(states, [1, 0, 2])

        outputs = self.W_out(states[:, -1, :])
        return outputs

class TrainingLossRecorder(Callback):
    def on_train_begin(self, logs=None):
        self.losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        
        
def loss_function(esn, ret_train, market_train, a1=-1, a2=0.01):
    def loss(_, y_pred):
        def transform_way_1(x):
            return a1 * tf.exp(a2 * x)
        
        def transform_way_2(x):
            return a1 * tf.exp(-a2 * x)
        
        def softmax(x):
            e_x = tf.exp(x - tf.reduce_max(x, axis=1, keepdims=True))
            return e_x / tf.reduce_sum(e_x, axis=1, keepdims=True)
        
        transformed_way_1 = transform_way_1(y_pred)
        transformed_way_2 = transform_way_2(y_pred)
        
        softmax_way_1 = softmax(transformed_way_1)
        softmax_way_2 = softmax(transformed_way_2)
        
        result = softmax_way_1 - softmax_way_2
        
        weighted_returns = result * ret_train
        weighted_returns = tf.reduce_sum(weighted_returns, axis=1, keepdims=True)
            
        market_tensor_float32 = tf.cast(market_train, tf.float32)
        weighted_returns_float32 = tf.cast(weighted_returns, tf.float32)
            
        matrix = tf.concat([weighted_returns_float32, market_tensor_float32], axis=1)
        row_averages = tf.reduce_mean(matrix, axis=0)
        row_averages_column = tf.reshape(row_averages, shape=(-1, 1))
        row_averages_row = tf.transpose(row_averages_column)
            
        matrix_float32 = tf.cast(matrix, dtype=tf.float32)
        cov_matrix = tfp.stats.covariance(matrix_float32, sample_axis=0, event_axis=-1)
        inv_cov_matrix = tf.linalg.inv(cov_matrix)
            
        result_step1 = tf.matmul(row_averages_row, inv_cov_matrix)
        final_result = tf.matmul(result_step1, row_averages_column)
        scalar_result = tf.squeeze(final_result)
            
        l1_reg_term = tf.add_n([tf.reduce_sum(tf.abs(v)) for v in esn.trainable_weights])
        l2_reg_term = tf.add_n([tf.nn.l2_loss(v) for v in esn.trainable_weights])
            
        loss = tf.exp(-scalar_result) + 0.06 * l1_reg_term + 0.06 * l2_reg_term
        return loss
    return loss


n_inputs = 2000
n_reservoir = 50
n_outputs = 500
spectral_radius = 0.7
l1 = 0.01/100
l2 = 0.01/100
a1 = 8
a2 = 8

esn = EchoStateNetwork(n_inputs, n_reservoir, n_outputs, spectral_radius, l1, l2)
loss_fn = loss_function(esn, ret_train, market_train, a1, a2)   

def create_esn_model():
    reset_session()
    esn = EchoStateNetwork(n_inputs, n_reservoir, n_outputs, spectral_radius, l1, l2)
    loss_fn = loss_function(esn, ret_train, market_train, a1, a2)
    esn.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.002*50), loss=loss_fn)
    return esn   
        
        
      # Create the loss recorder callback
loss_recorder = TrainingLossRecorder()

# Train the model
esn = create_esn_model()
esn.fit(input_data, target_data, epochs=20, callbacks=[loss_recorder]) 

asset_weights_esn_f2 = None


def analyze_and_plot_results4(esn, test_data, ret_test, market_test, test_obs, loss_recorder):
    # Predict using the ESN model
    predictions_esn = esn.predict(test_data)
    
    def transform_way_1(x):
        return -8* tf.exp(-8 * x)

    def transform_way_2(x):
        return -8 * tf.exp(8 * x)
    
    def softmax(x):
        return tf.nn.softmax(x)
    
        # Transformations
    transformed_way_1 = transform_way_1(predictions_esn)
    transformed_way_2 = transform_way_2(predictions_esn)
        
    softmax_way_1 = softmax(transformed_way_1)
    softmax_way_2 = softmax(transformed_way_2)
    
        # Calculate result
    result = softmax_way_1 - softmax_way_2
    
    asset_weights_esn_f2 = result
    
    plt.plot(loss_recorder.losses,marker="+",linewidth=0.5)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    
    plt.savefig(path + '/rnn_loss1', dpi=300) 
    
    plt.show()

    # Returns on deep factor
    deep_factor = result * ret_test
    deep_factor = np.sum(deep_factor, axis=1)

    # Returns on market factor
    x = np.arange(test_obs)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x, market_test, color='grey', linestyle='solid', label='Market Factor',linewidth=0.5)
    plt.bar(x, deep_factor, color='darkslateblue', linestyle='-', label='Deep Factor')
    plt.title('Deep Factor and Market Factor')
    plt.xlabel('Year')
    
    plt.xticks(np.arange(0, len(nn_test), 24), np.arange(1987, 2011, 2))  # Assuming monthly data, 24 ticks per year
    
    
    plt.ylabel('Return')
    plt.grid(False)
    plt.legend()
    
    plt.savefig(path + '/rnn_result1', dpi=300) 

    # Compute correlation
    correlation_matrix_esn = np.corrcoef(deep_factor, market_test)
    correlation = correlation_matrix_esn[0, 1]

    return correlation, asset_weights_esn_f2, deep_factor


#Compute final results
correlation_fold1, asset_weights_esn_f2, deep_factor_r2 = analyze_and_plot_results4(esn, test_data, ret_test, market_test, test_obs, loss_recorder)

h1 = loss_recorder.losses


#Fold 2 
#FOLD TWO 
###############################################################################

nn_train, nn_test, ret_train, ret_test, market_train, market_test,  = train_test_split(nn_input,returns,market_factor, test_size=0.5, random_state=42, shuffle=False)

#Train and test Integer 
train_obs = len(nn_train)
test_obs = len(nn_test)

#Reshaping Returns
ret_train = tf.reshape(ret_train, (train_obs, 500))
ret_train = tf.cast(ret_train, tf.float32)

#Reshaping market factor
market_train = tf.reshape(market_train, (train_obs, 1))

#Zero response
proxy_target = tf.zeros((train_obs, 500))
proxy_target = proxy_target.numpy()

#Rename for cross-validation
input_data = np.expand_dims(nn_test, axis=0)  # Add batch dimension 
target_data = np.expand_dims(proxy_target, axis=0)  # Add batch dimension 
test_data = np.expand_dims(nn_train, axis=0)  # Add batch dimension 

esn = create_esn_model()
esn.fit(input_data, target_data, epochs=20, callbacks=[loss_recorder])  

asset_weights_esn_f1 = None

def analyze_and_plot_results5(esn, test_data, ret_test, market_test, test_obs, loss_recorder):
    # Predict using the ESN model
    predictions_esn = esn.predict(test_data)
    
    def transform_way_1(x):
        return -8 * tf.exp(-8 * x)

    def transform_way_2(x):
        return -8 * tf.exp(8 * x)
    
    def softmax(x):
        return tf.nn.softmax(x)
    
        # Transformations
    transformed_way_1 = transform_way_1(predictions_esn)
    transformed_way_2 = transform_way_2(predictions_esn)
        
    softmax_way_1 = softmax(transformed_way_1)
    softmax_way_2 = softmax(transformed_way_2)
    
        # Calculate result
    result = softmax_way_1 - softmax_way_2
    
    asset_weights_esn_f1 = result
    
    plt.plot(loss_recorder.losses,marker="+",linewidth=0.5)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    
    plt.savefig(path + '/rnn_loss2', dpi=300) 

    # Returns on deep factor
    deep_factor = result * ret_test
    deep_factor = np.sum(deep_factor, axis=1)

    # Returns on market factor
    x = np.arange(test_obs)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x, market_test, color='grey', linestyle='solid', label='Market Factor',linewidth=0.5)
    plt.bar(x, deep_factor, color='darkslateblue', linestyle='-', label='Deep Factor')
    plt.title('Deep Factor and Market Factor')
    plt.xlabel('Year')
    
    plt.xticks(np.arange(0, len(nn_test), 24), np.arange(1965, 1988, 2))  # Assuming monthly data, 24 ticks per year
    
    plt.ylabel('Return')
    plt.grid(False)
    plt.legend()
    
    plt.savefig(path + '/rnn_result2', dpi=300) 

    # Compute correlation
    correlation_matrix_esn = np.corrcoef(deep_factor, market_test)
    correlation = correlation_matrix_esn[0, 1]

    return correlation, asset_weights_esn_f1, deep_factor


#Compute final results
correlation_fold2, asset_weights_esn_f1, deep_factor_r1 = analyze_and_plot_results5(esn, test_data, ret_test, market_test, test_obs, loss_recorder)

h2 = loss_recorder.losses

h3 = [(a + b) / 2 for a, b in zip(h1, h2)]
average_last_5_h3_1 = np.mean(h3[-5:])


#VALIDATION
###############################################################################

test_data = np.expand_dims(nn_val, axis=0)  # Add batch dimension 

asset_weights_esn_v = None

def analyze_and_plot_results6(esn, test_data, returns_val, market_factor_val, test_obs, loss_recorder):
    # Predict using the ESN model
    predictions_esn = esn.predict(test_data)
    
    def transform_way_1(x):
        return -8 * tf.exp(-8 * x)

    def transform_way_2(x):
        return -8 * tf.exp(8 * x)
    
    def softmax(x):
        return tf.nn.softmax(x)
    
        # Transformations
    transformed_way_1 = transform_way_1(predictions_esn)
    transformed_way_2 = transform_way_2(predictions_esn)
        
    softmax_way_1 = softmax(transformed_way_1)
    softmax_way_2 = softmax(transformed_way_2)
    
        # Calculate result
    result = softmax_way_1 - softmax_way_2
    
    asset_weights_esn_v = result 
    
    plt.plot(loss_recorder.losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model loss')
    plt.show()

    # Returns on deep factor
    deep_factor = result * returns_val
    deep_factor = np.sum(deep_factor, axis=1)

    # Returns on market factor
    x = np.arange(len(nn_val))

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x, market_factor_val, color='grey', linestyle='solid', label='Market Factor',linewidth=0.5)
    plt.bar(x, deep_factor, color='darkslateblue', linestyle='-', label='Deep Factor')
    plt.title('Deep Factor and Market Factor')
    plt.xlabel('Year')
    
    plt.xticks(np.arange(0, len(nn_val), 24), np.arange(2010, 2022, 2))  # Assuming monthly data, 24 ticks per year
    
    plt.ylabel('Return')
    plt.grid(False)
    plt.legend()
    
    plt.savefig(path + '/rnn_val', dpi=300) 

    # Compute correlation
    correlation_matrix_esn = np.corrcoef(deep_factor, market_factor_val)
    correlation = correlation_matrix_esn[0, 1]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(deep_factor, market_factor_val, c='darkslateblue', marker='+')
    plt.xlabel('Deep Factor')
    plt.ylabel('Market Factor')
    plt.title('Scatter Plot of Deep Factor vs Market Factor')
    
    plt.savefig(path + '/scatter', dpi=300) 

    return correlation, asset_weights_esn_v, deep_factor


#Compute final results
correlation_v, asset_weights_esn_v, deep_factor_rv = analyze_and_plot_results6(esn, test_data, returns_val, market_factor_val, test_obs, loss_recorder)


###############################################################################
#SUMMARY STATISTICS
###############################################################################

#Extract relevant variables for data summary

df_summary_f1 = df_holdout[['ret', 'melag', 'prc','me','me_20','Size','BMdec','OperProf','Investment',
                     'Mom12m','STreversal','me500','ret_ad']]

df_summary_f2 = df_train_val[['ret', 'melag', 'prc','me','me_20','Size','BMdec','OperProf','Investment',
                     'Mom12m','STreversal','me500','ret_ad']]

df_summary_f3 = df_all[['ret', 'melag', 'prc','me','me_20','Size','BMdec','OperProf','Investment',
                     'Mom12m','STreversal','me500','ret_ad']]

start_date_f1 = '196501'
end_date_f1 = '198710'
start_date_f2 = '198711'

df_summary_f4 = df_train_val.loc[(df_train_val['yyyymm'] >= start_date_f1) & (df_train_val['yyyymm'] <= end_date_f1),
                                 ['ret', 'melag', 'prc', 'me', 'me_20', 'Size', 'BMdec', 'OperProf', 'Investment',
                                  'Mom12m', 'STreversal', 'me500', 'ret_ad']]
df_summary_f5 = df_train_val.loc[df_train_val['yyyymm'] >= start_date_f2,
                                 ['ret', 'melag', 'prc', 'me', 'me_20', 'Size', 'BMdec', 'OperProf', 'Investment',
                                  'Mom12m', 'STreversal', 'me500', 'ret_ad']]

def my_summary_stats(data):
    """
    Summary stats: mean, variance, standard deviation, maximum and minimum.

    Parameters
    ----------
    data : TYPE: pd.DataFrame
        DESCRIPTION: dataframe for which descriptives will be computed
    Returns

    -------
    None. Prints descriptive table od the data
    """
    # generate storage for the stats as an empty dictionary
    my_descriptives = {}
    # loop over columns
    for col_id in data.columns:
        # fill in the dictionary with descriptive values by assigning the
        # column ids as keys for the dictionary
        my_descriptives[col_id] = [data[col_id].mean(),                  # mean
                                   data[col_id].var(),               # variance
                                   data[col_id].std(),                # st.dev.
                                   data[col_id].max(),                # maximum
                                   data[col_id].min(),                # minimum
                                   sum(data[col_id].isna()),          # missing
                                   len(data[col_id].unique()),  # unique values
                                   data[col_id].shape[0]]      # number of obs.
    # convert the dictionary to dataframe for a nicer output and name rows
    # pandas dataframe will automatically take the keys as columns if the
    # data input is a dictionary. Transpose for having the stats as columns
    my_descriptives = pd.DataFrame(my_descriptives,
                                   index=['mean', 'var', 'std', 'max', 'min',
                                          'na', 'unique', 'obs']).transpose()
    # define na, unique and obs as integers such that no decimals get printed
    ints = ['na', 'unique', 'obs']
    # use the .astype() method of pandas dataframes to change the type
    my_descriptives[ints] = my_descriptives[ints].astype(int)
    # print the descriptives, (\n inserts a line break)
    print('Descriptive Statistics:', '-' * 80,
          round(my_descriptives, 2), '-' * 80, '\n\n', sep='\n')

my_summary_stats(df_summary_f1)
my_summary_stats(df_summary_f2)
my_summary_stats(df_summary_f3)
my_summary_stats(df_summary_f4)
my_summary_stats(df_summary_f5)


def my_hist(data, varname, path, nbins=70):
    """
    Plot histograms.

    Parameters
    ----------
    data : TYPE: pd.DataFrame
        DESCRIPTION: dataframe containing variables of interest
    varname : TYPE: string
        DESCRIPTION: variable name for which histogram should be plotted
    path : TYPE: string
        DESCRIPTION: path where the plot will be saved
    nbins : TYPE: integer
        DESCRIPTION. Number of bins. The default is 10.

    Returns
    -------
    None. Prints and saves histogram.
    """
    # produce nice histograms
    data[varname].plot.hist(grid=False, bins=nbins, rwidth=0.75, color='lightsteelblue')
    # add labels
    plt.title('Histogram of ' + varname)
    plt.xlabel(varname)
    plt.ylabel('Frequency')
    # save the plot
    plt.savefig(path + '/histogram_of_' + varname + '.png')
    # print the plot
    plt.show()
    
my_hist(df_summary_f1,'ret_ad',path)
my_hist(df_summary_f1,'Size',path)

my_hist(df_summary_f2,'ret_ad',path)
my_hist(df_summary_f2,'Size',path)

my_hist(df_summary_f3,'ret_ad',path)
my_hist(df_summary_f3,'Size',path)

my_hist(df_summary_f4,'ret_ad',path)
my_hist(df_summary_f4,'Size',path)

my_hist(df_summary_f5,'ret_ad',path)
my_hist(df_summary_f5,'Size',path)






del start_date_f1, end_date_f1, start_date_f2
del df_summary_f1, df_summary_f2, df_summary_f3, df_summary_f4, df_summary_f5
#del df_all, df_holdout, df_train_val

###############################################################################
#Benchmarking
###############################################################################



def calculate_sharpe_ratio(data):
    average_return = data.mean()
    std_dev_return = data.std()
    sharpe_ratio = average_return / std_dev_return
    return sharpe_ratio

#Market Factor 
sharpe_ratio_m3 = calculate_sharpe_ratio(market_factor_val)

########## Sharpe Ratio Deep Tangency Portfolio Neural Network 

#Deep Tangency Portfolio NN 
theta_values = np.arange(-7, 7, 0.5)
sharpe_ratios_t3 = []

for theta in theta_values:
    t3 = market_factor_val + theta * deep_factor_v1
    sharpe_ratio_t3 = calculate_sharpe_ratio(t3)
    sharpe_ratios_t3.append(sharpe_ratio_t3)
    
plt.figure(figsize=(12, 6))
plt.plot(theta_values, sharpe_ratios_t3, marker='+', linestyle='-', color='darkslateblue')
plt.title('Sharpe Ratios Tangency Portfolio')
plt.axhline(y=sharpe_ratio_m3, color='grey', linestyle='-', label='MF SR')
plt.axvline(x=0, color='grey', linestyle='--')
plt.text(1, sharpe_ratio_m3 + 0.005, 'Sharpe Ratio Market Factor', color='grey', fontsize=9)
plt.xlabel('Theta')
plt.ylabel('Sharpe Ratio')
plt.grid(False)
plt.savefig(path + '/SRNN', dpi=300)
plt.show()  


#VAR Analysis

#VAR market portfolio 
confidence_level = 0.05  # 5% VaR (95% confidence level)

#Sort returns
sorted_market = market_factor_val.sort_values().reset_index(drop=True)

# Step 2: Determine the index for the confidence level
index = int(confidence_level * len(sorted_market))

# Step 3: Calculate VaR
VaR = sorted_market.iloc[index]

# Step 4: Determine returns beyond the VaR
returns_beyond_var_nn_f1 = sorted_market[sorted_market <= VaR]

# Step 5: Calculate the Expected Shortfall (average of returns beyond VaR)
expected_shortfall = returns_beyond_var_nn_f1.mean()



VaR_NN_results = []
ES_NN_results = []

for theta in theta_values:
    portfolio = market_factor_val + theta * deep_factor_v1
    
    # Sort returns
    sorted_portfolio = portfolio.sort_values().reset_index(drop=True)
    
    # Determine the index for the confidence level
    index = int(confidence_level * len(sorted_portfolio))
    
    # Calculate VaR
    VaR_S = sorted_portfolio.iloc[index]
    VaR_NN_results.append(VaR_S)
    
    # Determine returns beyond the VaR
    returns_beyond_var = sorted_portfolio[sorted_portfolio <= VaR]
    
    # Calculate the Expected Shortfall (average of returns beyond VaR)
    expected_shortfall_s = returns_beyond_var.mean()
    ES_NN_results.append(expected_shortfall_s)
    
plt.subplot(1, 2, 1)
plt.plot(theta_values, VaR_NN_results, marker='+', linestyle='-', color='darkslateblue')
plt.axhline(y=VaR, color='grey', linestyle='-', label='MF SR')
plt.axvline(x=0, color='grey', linestyle='--')
plt.text(1, VaR + 0.15, 'VaR Market Factor', color='grey', fontsize=9)
plt.title('VaR Neural Network')
plt.xlabel('Theta')
plt.ylabel('VaR')
plt.grid(False)
plt.savefig(path + '/VaRNN', dpi=300)
plt.show()  

plt.subplot(1, 2, 1)
plt.plot(theta_values, ES_NN_results, marker='+', linestyle='-', color='darkslateblue')
plt.axhline(y=expected_shortfall, color='grey', linestyle='-', label='MF SR')
plt.axvline(x=0, color='grey', linestyle='--')
plt.text(2, expected_shortfall + 0.15, 'ES Market Factor', color='grey', fontsize=9)
plt.title('ES Neural Network')
plt.xlabel('Theta')
plt.ylabel('ES')
plt.grid(False)
plt.savefig(path + '/ES_NN', dpi=300)
plt.show()  



















########## Sharpe Ratio Deep Tangency Portfolio RNN 

#Deep Tangency Portfolio NN 
theta_values = np.arange(-7, 7, 0.5)
sharpe_ratios_r3 = []

for theta in theta_values:
    r3 = market_factor_val + theta * deep_factor_rv
    sharpe_ratio_t3 = calculate_sharpe_ratio(r3)
    sharpe_ratios_r3.append(sharpe_ratio_t3)
    
plt.figure(figsize=(12, 6))
plt.plot(theta_values, sharpe_ratios_r3, marker='+', linestyle='-', color='darkslateblue')
plt.title('Sharpe Ratios Tangency Portfolio')
plt.axhline(y=sharpe_ratio_m3, color='grey', linestyle='-', label='MF SR')
plt.axvline(x=0, color='grey', linestyle='--')
plt.text(1, sharpe_ratio_m3 + 0.003, 'Sharpe Ratio Market Factor', color='grey', fontsize=9)
plt.xlabel('Theta')
plt.ylabel('Sharpe Ratio')
plt.grid(False)
plt.savefig(path + '/SRRNN', dpi=300)
plt.show()  


VaR_RNN_results = []
ES_RNN_results = []

for theta in theta_values:
    portfolio = market_factor_val + theta * deep_factor_rv
    
    # Sort returns
    sorted_portfolio = portfolio.sort_values().reset_index(drop=True)
    
    # Determine the index for the confidence level
    index = int(confidence_level * len(sorted_portfolio))
    
    # Calculate VaR
    VaR_M = sorted_portfolio.iloc[index]
    VaR_RNN_results.append(VaR_M)
    
    # Determine returns beyond the VaR
    returns_beyond_var = sorted_portfolio[sorted_portfolio <= VaR]
    
    # Calculate the Expected Shortfall (average of returns beyond VaR)
    expected_shortfall_s = returns_beyond_var.mean()
    ES_RNN_results.append(expected_shortfall_s)
    
plt.subplot(1, 2, 1)
plt.plot(theta_values, VaR_RNN_results, marker='+', linestyle='-', color='darkslateblue')
plt.axhline(y=VaR, color='grey', linestyle='-', label='MF SR')
plt.axvline(x=0, color='grey', linestyle='--')
plt.text(1, VaR + 0.3, 'VaR Market Factor', color='grey', fontsize=9)
plt.title('VaR Echo State Network')
plt.xlabel('Theta')
plt.ylabel('VaR')
plt.grid(False)
plt.savefig(path + '/VaRRNN', dpi=300)
plt.show()  

plt.subplot(1, 2, 1)
plt.plot(theta_values, ES_RNN_results, marker='+', linestyle='-', color='darkslateblue')
plt.axhline(y=expected_shortfall, color='grey', linestyle='-', label='MF SR')
plt.axvline(x=0, color='grey', linestyle='--')
plt.text(2, expected_shortfall - 0.25, 'ES Market Factor', color='grey', fontsize=9)
plt.title('ES Echo State Network')
plt.xlabel('Theta')
plt.ylabel('ES')
plt.grid(False)
plt.savefig(path + '/ES_RNN', dpi=300)
plt.show()  







