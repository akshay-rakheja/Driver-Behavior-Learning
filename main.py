import tensorflow
import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
# import sklearn
from datetime import datetime
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
# from sklearn.preprocessing import StandardScaler
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import CSVLogger
# from keras_visualizer import visualizer
import scipy
from scipy import stats
from statistics import mean
import os
import re
import json


def extract_features(df): #extract the necessary features from the file
    df2_time = df.loc[:, 'Timestamp']  # timestamp
    df2_image_index = df.loc[:, 'Image_Index']
    df2_rv = df.loc[:, 'object relative velocity 01':'object relative velocity 12']  # relative velocity
    df2_long_d = df.loc[:,
                 'Object Long distance to rear bumper 01':'Object Long distance to rear bumper 12']  # longitude distance to rear bumper
    df2_ttc = df.loc[:,
              'Object Time to Collision_inverse 01':'Object Time to Collision_inverse 12']  # time to collision inverse
    df2_ang_t = df.loc[:,
                'Angle to cntr target vehicle bumper 01':'Angle to cntr target vehicle bumper 12']  # angle to target vehicle bumper
    df2_polar_d = df.loc[:, 'object  polar distance 01':'object  polar distance 12']  # polar distance
    df2_veh_id = df.loc[:, 'Closest in Path Vehicle ID']  # vehicle id
    df2_obj_id = df.loc[:, 'Object ID 01':'Object ID 12']  # object id
    df2_lane = df.loc[:, 'Object Lane Location 01':'Object Lane Location 12']  # lane location
    df2_detect_v = df.loc[:, 'detected vehicle ahead']  # detected vehicle
    df2_latitude = df.loc[:, 'Latitude']  # latitude
    df2_longitude = df.loc[:, 'Longitude']  # longitude
    df2_ped_id = df.loc[:, 'Most Critical Ped ID']  # critical ped id
    df2_confidence = df.loc[:,
                     'Object level of Confidence 01':'Object level of Confidence 12']  # object confidence level
    df2_turnsignal = df.loc[:,
                     'TurnSignal on preceding vehicles 01':'TurnSignal on preceding vehicles 12']  # turn signal and brake lights on preceding vehicle
    df2_brakelights = df.loc[:,
                      'BrakeLights on preceding vehicles 01':'BrakeLights on preceding vehicles 12']  # turn signal and brake lights on preceding vehicle
    df2_distoIntersect = df.loc[:, 'DisToIntersection']  # distance to intersection
    df2_typeofObject = df.loc[:, 'Type of object 01':'Type of object 12']  # type of object
    df2_avg_speed = df.loc[:, 'Vehicle Speed Average Driven']

    cols_to_concat = [df2_time, df2_image_index, df2_rv, df2_long_d, df2_ttc, df2_ang_t, df2_polar_d, df2_veh_id,
                      df2_obj_id, df2_lane,
                      df2_detect_v, df2_latitude, df2_longitude, df2_ped_id, df2_confidence, df2_turnsignal,
                      df2_brakelights, df2_distoIntersect, df2_typeofObject, df2_avg_speed]
    df2 = pd.concat(cols_to_concat, axis=1)

    return df2


def get_highway_2(df2): #get highway segment data for drives >= 15 (different route compared to drived <15)
    df2 = df2[df2['Latitude'] != 0]
    df2 = df2.reset_index()

    seg_flag_1 = []
    seg_flag_2 = []
    seg_flag_3 = []
    seg_flag_4 = []
    seg_flag_5 = []

    lat_highway_1 = []
    long_highway_1 = []

    lat_highway_2 = []
    long_highway_2 = []

    lat_highway_3 = []
    long_highway_3 = []

    lat_highway_4 = []
    long_highway_4 = []

    lat_highway_5 = []
    long_highway_5 = []

    highway_index = []

    in_highway_1 = False
    highway_ended_1 = False

    in_highway_2 = False
    highway_ended_2 = False

    in_highway_3 = False
    highway_ended_3 = False

    in_highway_4 = False
    highway_ended_4 = False

    in_highway_5 = False
    highway_ended_5 = False

    for index, row in df2.iterrows():

        if np.sqrt(
                (row['Latitude'] - 43.421139) ** 2 + (
                        row['Longitude'] + 80.508611) ** 2) < 0.003 and not highway_ended_1:   #starting coordinates for the 1st highway segment
            in_highway_1 = True     #indicates highway has started
        if np.sqrt((row['Latitude'] - 43.432538) ** 2 + (row['Longitude'] + 80.459034) ** 2) < 0.003 and in_highway_1: #ending coordinates for the 2nd highway segment
            in_highway_1 = False    
            highway_ended_1 = True  #indicates that highway has ended

        if in_highway_1:
            lat_highway_1.append(row['Latitude'])
            long_highway_1.append(row['Longitude'])
            highway_index.append(index) #adding index of the row belonging to highway segment 1
            seg_flag_1.append(1)        #adding segment 1
        else:
            lat_highway_1.append(np.NaN)
            long_highway_1.append(np.NaN)


        if np.sqrt(
                (row['Latitude'] - 43.433598) ** 2 + (
                        row['Longitude'] + 80.448804) ** 2) < 0.003 and not highway_ended_2:  #starting coordinates for the 2nd highway segment
            in_highway_2 = True
        if np.sqrt((row['Latitude'] - 43.404753) ** 2 + (row['Longitude'] + 80.380763) ** 2) < 0.003 and in_highway_2: #ending coordinates for the 2nd highway segment
            in_highway_2 = False
            highway_ended_2 = True

        if in_highway_2:
            lat_highway_2.append(row['Latitude'])
            long_highway_2.append(row['Longitude'])
            highway_index.append(index)  #adding index of the row belonging to highway segment 2
            seg_flag_2.append(2)    #adding segment 2
        else:
            lat_highway_2.append(np.NaN)
            long_highway_2.append(np.NaN)


        if np.sqrt(
                (row['Latitude'] - 43.435639) ** 2 + (
                        row['Longitude'] + 80.455639) ** 2) < 0.003 and not highway_ended_3:  #starting coordinates for the 3rd highway segment
            in_highway_3 = True
        if np.sqrt((row['Latitude'] - 43.416328) ** 2 + (row['Longitude'] + 80.409406) ** 2) < 0.003 and in_highway_3:  #ending coordinates for the 3rd highway segment
            in_highway_3 = False
            highway_ended_3 = True

        if in_highway_3:
            lat_highway_3.append(row['Latitude'])
            long_highway_3.append(row['Longitude'])
            highway_index.append(index) #adding index of the row belonging to highway segment 3
            seg_flag_3.append(3)     #adding segment 3
        else:
            lat_highway_3.append(np.NaN)
            long_highway_3.append(np.NaN)



        if np.sqrt(
                (row['Latitude'] - 43.403363) ** 2 + (
                        row['Longitude'] + 80.375839) ** 2) < 0.003 and not highway_ended_4: #starting coordinates for the 4th highway segment
            in_highway_4 = True
        if np.sqrt((row['Latitude'] - 43.418240) ** 2 + (row['Longitude'] + 80.289940) ** 2) < 0.003 and in_highway_4: #ending coordinates for the 3rd highway segment
            in_highway_4 = False
            highway_ended_4 = True

        if in_highway_4:
            lat_highway_4.append(row['Latitude'])
            long_highway_4.append(row['Longitude'])
            highway_index.append(index)
            seg_flag_4.append(4)
        else:
            lat_highway_4.append(np.NaN)
            long_highway_4.append(np.NaN)



        if np.sqrt(
                (row['Latitude'] - 43.418496) ** 2 + (
                        row['Longitude'] + 80.290089) ** 2) < 0.003 and not highway_ended_5:
            in_highway_5 = True
        if np.sqrt((row['Latitude'] - 43.414397) ** 2 + (row['Longitude'] + 80.320906) ** 2) < 0.003 and in_highway_5:
            in_highway_5 = False
            highway_ended_5 = True

        if in_highway_5:
            lat_highway_5.append(row['Latitude'])
            long_highway_5.append(row['Longitude'])
            highway_index.append(index)
            seg_flag_5.append(5)
        else:
            lat_highway_5.append(np.NaN)
            long_highway_5.append(np.NaN)


    df_highway = df2.iloc[highway_index, :]
    df_highway.insert(len(df_highway.columns), 'Seg#', seg_flag_1 + seg_flag_2 + seg_flag_3 + seg_flag_4 + seg_flag_5)

    return df_highway


def get_highway(df2):     # get highway segment data for drives <15 (different route compared to drived >=15)
    df2 = df2[df2['Latitude'] != 0]
    df2 = df2.reset_index()

    seg_flag_1 = []
    seg_flag_2 = []
    seg_flag_3 = []

    lat_highway_1 = []
    long_highway_1 = []

    lat_highway_2 = []
    long_highway_2 = []

    lat_highway_3 = []
    long_highway_3 = []

    highway_index = []

    in_highway_1 = False
    highway_ended_1 = False

    in_highway_2 = False
    highway_ended_2 = False

    in_highway_3 = False
    highway_ended_3 = False

    for index, row in df2.iterrows():

        if np.sqrt(
                (row['Latitude'] - 43.435639) ** 2 + (
                        row['Longitude'] + 80.455639) ** 2) < 0.0003 and not highway_ended_1:
            in_highway_1 = True
        if np.sqrt((row['Latitude'] - 43.416328) ** 2 + (row['Longitude'] + 80.409406) ** 2) < 0.0003 and in_highway_1:
            in_highway_1 = False
            highway_ended_1 = True

        if in_highway_1:
            lat_highway_1.append(row['Latitude'])
            long_highway_1.append(row['Longitude'])
            highway_index.append(index)
            seg_flag_1.append(1)
        else:
            lat_highway_1.append(np.NaN)
            long_highway_1.append(np.NaN)

        if np.sqrt(
                (row['Latitude'] - 43.403363) ** 2 + (
                        row['Longitude'] + 80.375839) ** 2) < 0.0003 and not highway_ended_2:
            in_highway_2 = True
        if np.sqrt((row['Latitude'] - 43.418240) ** 2 + (row['Longitude'] + 80.289940) ** 2) < 0.0003 and in_highway_2:
            in_highway_2 = False
            highway_ended_2 = True

        if in_highway_2:
            lat_highway_2.append(row['Latitude'])
            long_highway_2.append(row['Longitude'])
            highway_index.append(index)
            seg_flag_2.append(2)
        else:
            lat_highway_2.append(np.NaN)
            long_highway_2.append(np.NaN)

        if np.sqrt(
                (row['Latitude'] - 43.418496) ** 2 + (
                        row['Longitude'] + 80.290089) ** 2) < 0.0003 and not highway_ended_3:
            in_highway_3 = True
        if np.sqrt((row['Latitude'] - 43.414397) ** 2 + (row['Longitude'] + 80.320906) ** 2) < 0.0003 and in_highway_3:
            in_highway_3 = False
            highway_ended_3 = True

        if in_highway_3:
            lat_highway_3.append(row['Latitude'])
            long_highway_3.append(row['Longitude'])
            highway_index.append(index)
            seg_flag_3.append(3)
        else:
            lat_highway_3.append(np.NaN)
            long_highway_3.append(np.NaN)

    df_highway = df2.iloc[highway_index, :]
    df_highway.insert(len(df_highway.columns), 'Seg#', seg_flag_1 + seg_flag_2 + seg_flag_3)

    return df_highway


def target_car_features(df2_2):   # consider data only for the car in front of the subject vehicle (disregard the cars in other lanes)

    ttc_object_2 = []
    relative_velocity_2 = []
    angle_2_target_veh_2 = []
    turnsignal_2 = []
    brakelights_2 = []
    long_dist_2 = []
    confidence_2 = []
    polar_dist_2 = []
    flag_ahead_2 = []
    latitude_2 = []
    longitude_2 = []

    #add the necessary features for the car in front to their respective lists and later add them to dataframe
    for index, row in df2_2.iterrows():
        if row['detected vehicle ahead'] == 1:
            if np.where(row['Object Lane Location 01':'Object Lane Location 12'] == 2)[0].size > 0: #when the target car(in front) is in the same lane as subject vehicle
                vehicle_id = row['Closest in Path Vehicle ID']
                vehicle_num = np.where(row['Object ID 01':'Object ID 12'] == vehicle_id)[0][0] + 1
                ttc_object_2.append(row['Object Time to Collision_inverse {:02d}'.format(vehicle_num)])
                relative_velocity_2.append(row['object relative velocity {:02d}'.format(vehicle_num)])
                angle_2_target_veh_2.append(row['Angle to cntr target vehicle bumper {:02d}'.format(vehicle_num)])
                turnsignal_2.append(row['TurnSignal on preceding vehicles {:02d}'.format(vehicle_num)])
                brakelights_2.append(row['BrakeLights on preceding vehicles {:02d}'.format(vehicle_num)])
                long_dist_2.append(1 / row['Object Long distance to rear bumper {:02d}'.format(vehicle_num)])
                confidence_2.append(row['Object level of Confidence {:02d}'.format(vehicle_num)])
                polar_dist_2.append(1 / row['object  polar distance {:02d}'.format(vehicle_num)])
                latitude_2.append(row['Latitude'])
                longitude_2.append(row['Longitude'])
                flag_ahead_2.append(1)

            else:
                ttc_object_2.append(0)  # appending 0's for time to collision when vehicle isnt detected
                relative_velocity_2.append(row['Vehicle Speed Average Driven'])
                angle_2_target_veh_2.append(90)  # considering the angle to target vehicle is 90 degrees 
                turnsignal_2.append(0)
                brakelights_2.append(0)
                long_dist_2.append(0)
                confidence_2.append(1)
                polar_dist_2.append(0)
                latitude_2.append(np.NaN)
                longitude_2.append(np.NaN)
                flag_ahead_2.append(0)

        else:
            ttc_object_2.append(0)
            relative_velocity_2.append(row['Vehicle Speed Average Driven'])
            angle_2_target_veh_2.append(90)
            turnsignal_2.append(0)
            brakelights_2.append(0)
            long_dist_2.append(0)
            confidence_2.append(1)
            polar_dist_2.append(0)
            latitude_2.append(np.NaN)
            longitude_2.append(np.NaN)
            flag_ahead_2.append(0)

    #adding the necessarry features for the target vehicle (Car in front) to the dataframe
    df2_2['ttc_object'] = ttc_object_2
    df2_2['relative_velocity'] = relative_velocity_2
    df2_2['angle_2_target_veh'] = angle_2_target_veh_2
    df2_2['turnsignal'] = turnsignal_2
    df2_2['brakelights'] = brakelights_2
    df2_2['long_dist'] = long_dist_2
    df2_2['confidence'] = confidence_2
    df2_2['polar_dist'] = polar_dist_2
    df2_2['Latitude'] = latitude_2
    df2_2['Longitude'] = longitude_2
    df2_2['flag_ahead'] = flag_ahead_2

    return df2_2


def Image_merge(df2):   #merge the columns by Image_Index -> take mean of the numerical data type columns that have the same Image_index  and mode of the categorical ones
    df_new = df2
    cols_mean = ['ttc_object', 'relative_velocity', 'long_dist', 'confidence', 'polar_dist',
                 'Vehicle Speed Average Driven', 'Latitude', 'Longitude', 'Timestamp']
    # # # cols grouped by means
    df_mean = df_new.groupby('Image_Index', as_index=False)[cols_mean].mean()

    cols_mode = ['turnsignal', 'brakelights', 'flag_ahead', 'Seg#']
    # # # cols grouped by mode
    df_mode = df_new.groupby('Image_Index', as_index=False)[cols_mode].agg(lambda x: scipy.stats.mode(x)[0])

    # # cols merged
    df_new = pd.merge(df_mean, df_mode, how="outer", on=['Image_Index'])

    return df_new


def scale_highway(df_highway): #scaling the numerical data types
    df_highway_scaled = df_highway.loc[:, ['relative_velocity', 'ttc_object', 'long_dist', 'polar_dist',
                                           'Vehicle Speed Average Driven']].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    df_highway_scaled.loc[:,
    ['Image_Index', 'confidence', 'Latitude', 'Longitude', 'turnsignal', 'brakelights', 'flag_ahead',
     'Seg#']] = df_highway.loc[:,
                ['Image_Index', 'confidence', 'Latitude', 'Longitude', 'turnsignal', 'brakelights', 'flag_ahead',
                 'Seg#']]
    return df_highway_scaled


def remove_0s(df2): #removes rows where we do not have a car infront (checks for 1000 rows at once)
    stretch = 1000  # number of contiguous rows with 0s to remove
    to_remove = [] # list of indices that will be removed
    for i in range(len(df2)):
        if (df2['ttc_object'][i:i + stretch] == 0).all(): # if from i th index to i th  + stretch (1000) all the ttc values are 0 (default value when no car is in front), then add those indices to to_remove
            to_remove.extend(j for j in range(i, i + stretch))

    to_remove = list(set(to_remove))  # removes duplicate indices from the list
    to_remove = list(filter(lambda x: x < len(df2), to_remove)) # index should not exceed the length of the dataframe

    df2.drop(df2.index[to_remove], inplace=True, axis=0) #drop the indices from the dataframe

    return df2


def plot_route(df, df_highway, filename, mode): #plots training and test paths and saves them in /plots directory; mode is passed to see if we are plotting training or test path
    if mode == 1:  # Training
        map_ax = df[df['Longitude'] != 0].plot('Longitude', 'Latitude', figsize=(12, 8), legend=False)
        map_ax.set_facecolor('black')
        map_ax.set_ylabel('Latitude')
        map_ax.set_xlabel('Longitude')
        map_ax.set_title('Map')
        map_ax.scatter(df_highway['Longitude'], df_highway['Latitude'], c='lime')
        plt.savefig('plots/Training_{}.png'.format(filename))
        plt.close()

    else:  # testing
        map_ax = df[df['Longitude'] != 0].plot('Longitude', 'Latitude', figsize=(12, 8), legend=False)
        map_ax.set_facecolor('black')
        map_ax.set_ylabel('Latitude')
        map_ax.set_xlabel('Longitude')
        map_ax.set_title('Map')
        map_ax.scatter(df_highway['Longitude'], df_highway['Latitude'], c='red')
        plt.savefig('plots/Testing_{}.png'.format(filename))
        plt.close()


def train_df(df2, n_past, n_future):    #creates the data structure to be fed into the model; creates 3d tensor for X and 2d one for y

    Image_index = df2['Image_Index']
    df2 = df2.drop('Image_Index', axis=1)
    df2 = df2.drop(['Latitude', 'Longitude'], axis=1)
    df_train = df2

    df_train_target = df_train['ttc_object']
    # df_train_target.astype('category')
    # df_train_target = df_train_target.cat.codes

    # dropping ttc_object (target) from train
    #df_train.drop('ttc_object', axis=1, inplace=True)

    # converting df's to np array
    df_train = np.array(df_train)
    df_train_target = np.array(df_train_target).reshape(-1, 1)
    df_train_target = np.array(df_train_target)

    # Creating a data structure with 500 GSC"s and 1 output
    X_train = []
    y_train = []

    for i in range(n_past, len(df_train) - n_future + 1):
        X_train.append(df_train[i - n_past: i, 0:df_train.shape[1]]) #appends(200,10) slices of data to X_train n # of times
        y_train.append(df_train_target[i:i + n_future , -1]) #appends (1,40) slices of target variable to y_train n # of times

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    return X_train, y_train


def rm_disc(X_train, y_train, n_past):  #remove discontinuous/overlapping highway segments
    index = []
    for i in range(len(X_train)):
        if (np.count_nonzero(X_train[i][:, 9] == 1) < n_past and np.count_nonzero(X_train[i][:, 9] == 1) != 0): # here 9 refers to index of column 'Seg#'; (looping over X_train indices)instances when the segment # for the nth sample contains 1 as the segment # but its occurence is less than n_past samples (second dimension of X_train = n_past). Also there should be non zero instances when the segement # is 1. 
            index.append(i) # append the index of X_train staisfying the above condition
        if (np.count_nonzero(X_train[i][:, 9] == 2) < n_past and np.count_nonzero(X_train[i][:, 9] == 2) != 0): # here 9 refers to index of column 'Seg#'; (looping over X_train indices)instances when the segment # for the nth sample contains 2 as the segment # but its occurence is less than n_past samples (second dimension of X_train = n_past). Also there should be non zero instances when the segement # is 2. 
            index.append(i)
        if (np.count_nonzero(X_train[i][:, 9] == 3) < n_past and np.count_nonzero(X_train[i][:, 9] == 3) != 0): # here 9 refers to index of column 'Seg#'; (looping over X_train indices)instances when the segment # for the nth sample contains 3 as the segment # but its occurence is less than n_past samples (second dimension of X_train = n_past). Also there should be non zero instances when the segement # is 3.  
            index.append(i)
        if (np.count_nonzero(X_train[i][:, 9] == 4) < n_past and np.count_nonzero(X_train[i][:, 9] == 4) != 0):
            index.append(i)
        if (np.count_nonzero(X_train[i][:, 9] == 5) < n_past and np.count_nonzero(X_train[i][:, 9] == 5) != 0):
            index.append(i)

    index = list(set(index))    #consists of only unique values of indices 
    X_train = np.delete(X_train, index, axis=0) #remove the above indices (indicies where each nth sample does not have a full set of segments being the same value [1,2,3,4,5])
    y_train = np.delete(y_train, index, axis=0)

    print("Discontinuous frames removed: {}".format(index))
    return X_train, y_train


def create_model():
    # Initializing the Neural Network based on LSTM
    model = Sequential()

    # Adding 1st LSTM layer
    model.add(LSTM(units=10, return_sequences=True, input_shape=(200, 10)))
    # adding l2 regularizer
    regularizers.l2(0.01)

    # Adding 2nd LSTM layer
    model.add(LSTM(units=10, return_sequences=False))
    # adding l2 regularizaer
    regularizers.l2(0.01)

    # Output layer
    model.add(Dense(units=40, activation='linear'))

    # Compiling the Neural Network
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model


def run_model(X, y, model, X_v, y_v):
    es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=10, verbose=1) #early stopping the training if no improvement is found after 10 epochs
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1) # reducing learning rate after 10 epochs of no improvement
    mcp = ModelCheckpoint(filepath='weights.h5', monitor='val_loss', verbose=1, save_best_only=True,
                          save_weights_only=True)
    date = datetime.now().strftime("%Y_%m_%d_%I_%M_%S_%p")
    csv_logger = CSVLogger("model_history_log_{}.csv".format(date), append=True)

    tb = TensorBoard(log_dir='logs_1')

    #training
    history = model.fit(X, y, shuffle=True, epochs=50, callbacks=[es, rlr, mcp, tb, csv_logger], verbose=1,
                        batch_size=256, validation_data=(X_v, y_v))
    return history


def myprint(s): #saves model summary to a file
    date = datetime.now().strftime("%Y_%m_%d_%I_%M_%S_%p")
    with open('plots/modelsummary_{}.txt'.format(date), 'w+') as f:
        print(s, file=f)


def get_rmse(Y, pred):  #gets root mean square value
    
    rmse_sum_error = 0.0
    for i in range(len(Y)):
        prediction_error = pred[i] - Y[i]
        rmse_sum_error += (prediction_error ** 2)
    mean_error = rmse_sum_error / float(len(Y))
    rmse = mean_error ** (0.5)

    return rmse


def get_mae(Y, pred):   #gets mean absolute error value

    mae_sum_error = 0.0
    for i in range(len(Y)):
        mae_sum_error += abs(pred[i] - Y[i])
    mae = mae_sum_error / float(len(Y))

    return mae


'''
def metrics(X, Y, model):   #print the metrics including prediction vs actual value plots 

    predictions = model.predict(X)
    
    print("Predictions shape is {} ".format(predictions.shape[0]))
    print("Y_test shape is {} ".format(Y.shape[0]))

    rmse_list = []
    mae_list = []

    for i in random.sample(range(0, Y.shape[0]), 10): #plots 10 random instances (5sec + 1sec) of prediction and actual values

        rmse_list.append(get_rmse(Y[i],predictions[i])) #rmse for the 10 random samples
        mae_list.append(get_mae(Y[i],predictions[i]))   #mae for the 10 random samples

        plt.plot(np.arange(0,200,1),X[i,:,1],label = 'ttc_previous 5seconds')   #Xaxis :0-200, yaxis: X[random sample, 200 points (5sec), 1-represnets time to collision]
        plt.plot(list(np.arange(200,240,1)),predictions[i], label = 'Predicted ttc values') #plots 40 predictions for a random samlple
        plt.plot(list(np.arange(200,240,1)),Y[i], label ='Actual ttc values') #plots actual values for the ransom sample
        plt.title('Predicted vs Actual Time to collision values (scaled)')
        plt.legend()
        plt.savefig('plots/PredictvsActual_{}.png'.format(i))   #save in plots directory
        plt.close()
    
    print("RMSE for 10 random samples is {}".format(rmse_list))
    print("MAE for 10 random samples is {}".format(mae_list))


    rmse = get_rmse(Y,predictions) #rmse for all the predictions combined
    mae = get_mae(Y, predictions) #mae for all the predictions
    
    return rmse, mae

'''



def metrics(X, Y, model):

    predictions = model.predict(X)

    print("Predictions shape is {} ".format(predictions.shape[0]))
    print("Y_test shape is {} ".format(Y.shape[0]))

    rmse_list = []
    mae_list = []

    test = X[0,:,1].reshape(-1,1)
    test_pred = predictions[0,:].reshape(-1,1)

    for i in range(1,2201):
        test = np.append(test,X[i,-1,1].reshape(-1,1) , axis=0)

    for i in range(1,2161):
        test_pred = np.append(test_pred,predictions[i,-1].reshape(-1,1), axis =0)


    for i in random.sample(range(0, Y.shape[0]), 10):

        rmse_list.append(get_rmse(Y[i],predictions[i]))
        mae_list.append(get_mae(Y[i],predictions[i]))

        plt.plot(np.arange(0,200,1),X[i,:,1],label = 'previous 5seconds')
        plt.plot(list(np.arange(200,240,1)),predictions[i], label = 'Predicted ttc values')
        plt.plot(list(np.arange(200,240,1)),Y[i], label ='Actual ttc values')
        plt.legend()
        plt.savefig('plots/PredictvsActual_{}.png'.format(i))
        plt.close()

    print("RMSE for 10 random samples is {}".format(rmse_list))
    print("MAE for 10 random samples is {}".format(mae_list))



    plt.plot(np.arange(0,2400,1), test.reshape(2400,-1), label = '1 min interval as 5 secs')
    #plt.plot(np.arange(200, 2600, 1), Y[0:60,:].reshape(2400,-1), label = 'Actual values')
    plt.plot(np.arange(200, 2400, 1), test_pred.reshape(2200,-1), label = 'predicted values')
    plt.legend()
    plt.savefig('plots/PredictvsActual_1min.png')
    plt.close()

    rmse  = get_rmse(Y,predictions)
    mae = get_mae(Y, predictions)

    return rmse, mae


def plot_metrics(history): #plots lossvs epochs and saves it in /plots directory
    plt.title('Loss vs Epochs')
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.legend()
    plt.show()
    plt.savefig('plots/LossvsEpochs.png')
    plt.close()

def get_filepaths(directory):
    odd = []  # list of odd file paths
    even = []  # list of even file paths

    # Walk the tree.
    for root, directories, files in os.walk(directory):  #going through directory to look for files then search using regex
        for filename in files:

            odd_match = re.search("P[1-2]*[13579]{1,2}_route[1-2]_VLT.csv", filename)  # using regex to filer odd files
            even_match = re.search("P[1-2]*[02468]{1,2}_route[1-2]_VLT.csv", filename)  # same for even files

            if odd_match:
                odd.append(os.path.join(root, filename))
            elif even_match:
                even.append(os.path.join(root, filename))
            else:
                print("Filename {} does not match regex format".format(filename))

            odd.sort(key=lambda f: int(re.sub('\D', '', f)))  # sorts them in ascending order
            even.sort(key=lambda f: int(re.sub('\D', '', f)))

    return odd, even  # Self-explanatory.


def main():
    # change this path to where your parquet/csv files are stored.
    odd_files, even_files = get_filepaths(
        "/home/arakheja/scratch/DBL_Script/Data/csvs")

    OutputFile = open('output.txt', 'w')

    n_future = 40  # Number of Image indices we want top predict into the future (roughly 1s)
    n_past = 200  # Number of Image indices we want to use to predict the future (roughly 5s)

    # initial dimensions of the training and testing sets, the second dimension (5 here) depends on n_past(previous timesteps used)
    X_train = np.empty((0, 200, 10))
    X_test = np.empty((0, 200, 10))
    y_train = np.empty((0, 40))
    y_test = np.empty((0, 40))

    for i in range(len(odd_files)):
        df = pd.read_csv(odd_files[i])
        filename = os.path.basename(odd_files[i]).split('.')[0]
        print("Adding {} to training".format(filename))
        print("Adding {} to training".format(filename), file=OutputFile)
        df2 = extract_features(df)  # extract only relevant features (ttc,relative velocity, distance,etc.)
        print("Feature extraction done", file=OutputFile)

        if int(filename.split('_')[0][1:]) < 15:
            print("File# <15",file = OutputFile)
            df_highway = get_highway(df2)  # Get Data corresponding to highway coordinates
        else:
            print("File # >=15",file =OutputFile)
            df_highway = get_highway_2(df2)  # Get Data corresponding to highway coordinates

        print("Got highway datapoints", file=OutputFile)
        df_highway = target_car_features(df_highway)  # append features only for the target vehicle
        print("Appended target vehicle datapoints to dataframe", file=OutputFile)
        df_highway = Image_merge(
            df_highway)  # Merge Image indices of same value into one (mean of numerical data, mode of categorical ones)
        print("Merged Image Indices", file=OutputFile)
        df_highway = scale_highway((df_highway))
        print("Scaled highway", file=OutputFile)
        df_highway = remove_0s(df_highway)  # Remove a stretch of 0s (stretch = 1000 here)
        print("Removed stretch of 0's", file=OutputFile)
        plot_route(df2, df_highway, filename, mode=1)   #plots the highway segment along with original route
        print("Plotted training highway route", file=OutputFile)
        X_h, y_h = train_df(df_highway, n_past, n_future)   #creates the datastructure for X_train and y_train 
        print("Constructed 3D tensor to be fed to the model", file=OutputFile)
        X_h, y_h = rm_disc(X_h, y_h, n_past)    #removes discontinuties between highway segments
        print("Removed discontinuity between drive segments from the same drive", file=OutputFile)

        X_train = np.append(X_train, X_h, axis=0) #appends training data created from an invididual file to X_train
        y_train = np.append(y_train, y_h, axis=0) 
        print(X_train.shape, y_train.shape, file=OutputFile)

    print("Appended Odd training sets to X_train and y_train", file=OutputFile)

    print("Starting to append even files to training and test sets randomly.", file=OutputFile)

    train_count_even = 0
    for i in range(len(even_files)):

        # Append Even Training set
        if np.random.randint(0, 2) == 0 and train_count_even <= (len(
                even_files) / 2):  # Append even files to training set randomly at max 12 times (since we have 24 even files the other 12 go to test)
            df = pd.read_csv(even_files[i])
            filename = os.path.basename(even_files[i]).split('.')[0]
            print("Adding {} to training".format(filename))
            print("Adding {} to training".format(filename), file=OutputFile)
            df2 = extract_features(df)  # extract only relevant features (ttc,relative velocity, distance,etc.)
            print("Feature extraction done", file=OutputFile)

            if int(filename.split('_')[0][1:]) < 15:
                print("File# <15",file = OutputFile)
                df_highway = get_highway(df2)  # Get Data corresponding to highway coordinates
            else:
                print("File # >=15",file =OutputFile)
                df_highway = get_highway_2(df2)
            
            print("Got highway datapoints", file=OutputFile)
            df_highway = target_car_features(df_highway)  # append features only for the target vehicle
            print("Appended target vehicle datapoints to dataframe", file=OutputFile)
            df_highway = Image_merge(
                df_highway)  # Merge Image indices of same value into one (mean of numerical data, mode of categorical ones)
            print("Merged Image Indices", file=OutputFile)
            df_highway = scale_highway((df_highway))
            print("Scaled highway", file=OutputFile)
            df_highway = remove_0s(df_highway)  # Remove a stretch of 0s (stretch = 1000 here)
            print("Removed stretch of 0's", file=OutputFile)
            plot_route(df2, df_highway, filename, mode=1)
            print("Plotted training highway route", file=OutputFile)
            X_h, y_h = train_df(df_highway, n_past, n_future)
            print("Constructed 3D tensor to be fed to the model", file=OutputFile)
            X_h, y_h = rm_disc(X_h, y_h, n_past)
            print("Removed discontinuity between drive segments from the same drive", file=OutputFile)

            X_train = np.append(X_train, X_h, axis=0)   #appending training set for the file to X_train
            y_train = np.append(y_train, y_h, axis=0)   #appending training set for the file to y_train

            train_count_even += 1

            print(X_train.shape, y_train.shape, file=OutputFile)

        # Append Even Test set
        else:
            df = pd.read_csv(even_files[i])
            filename = os.path.basename(even_files[i]).split('.')[0]
            print("Adding {} to testing".format(filename))
            print("Adding {} to testing".format(filename, file=OutputFile))
            df2 = extract_features(df)  # extract only relevant features (ttc,relative velocity, distance,etc.)
            print("Feature extraction done", file=OutputFile)

            if int(filename.split('_')[0][1:]) < 15:
                print("File# <15",file = OutputFile)
                df_highway = get_highway(df2)  # Get Data corresponding to highway coordinates
            else:
                print("File # >=15",file =OutputFile)
                df_highway = get_highway_2(df2)
            
            print("Got highway datapoints", file=OutputFile)
            df_highway = target_car_features(df_highway)  # append features only for the target vehicle
            print("Appended target vehicle datapoints to dataframe", file=OutputFile)
            df_highway = Image_merge(
                df_highway)  # Merge Image indices of same value into one (mean of numerical data, mode of categorical ones)
            print("Merged Image Indices", file=OutputFile)
            df_highway = scale_highway((df_highway))
            print("Scaled highway", file=OutputFile)
            df_highway = remove_0s(df_highway)  # Remove a stretch of 0s (stretch = 1000 here)
            print("Removed stretch of 0's", file=OutputFile)
            plot_route(df2, df_highway, filename, mode=0)
            print("Plotted testing highway route", file=OutputFile) 
            X_h, y_h = train_df(df_highway, n_past, n_future)   #creating test data structure (similar to training set) for X_test and y_test
            print("Constructed 3D tensor to be fed to the model", file=OutputFile)
            X_h, y_h = rm_disc(X_h, y_h, n_past)        #remove discontinuities between driving segments
            print("Removed discontinuity between testing drive segments from the same drive", file=OutputFile)

            X_test = np.append(X_test, X_h, axis=0)
            y_test = np.append(y_test, y_h, axis=0)

            print(X_test.shape, y_test.shape, file=OutputFile)

    # Saving X_train, y_train, X_test, y_test
    with open('X_train.npy', 'wb') as f:
        np.save(f, X_train)
    with open('y_train.npy', 'wb') as f:
        np.save(f, y_train)
    with open('X_test.npy', 'wb') as f:
        np.save(f, X_test)
    with open('y_test.npy', 'wb') as f:
        np.save(f, y_test)

    print('Final X_train shape == {}.'.format(X_train.shape), file=OutputFile)
    print('Final y_train shape == {}.'.format(y_train.shape), file=OutputFile)

    print('Final X_test shape == {}.'.format(X_test.shape), file=OutputFile)
    print('Final y_test shape == {}.'.format(y_test.shape), file=OutputFile)

    #creating training and val sets to feed to the model
    indices = np.random.permutation(X_train.shape[0])
    training_idx, val_idx = indices[:int(0.8 * X_train.shape[0])], indices[int(0.8 * X_train.shape[0]):]   # training set is 80% of X_train, 20% is val set
    X_train, X_val = X_train[training_idx, :], X_train[val_idx, :]
    y_train, y_val = y_train[training_idx, :], y_train[val_idx, :]

    #creates a model
    model = create_model()  # also saves the model as my_model.h5

    #runs the model
    history = run_model(X_train, y_train, model, X_val, y_val)

    # Makes predictions using X,y test sets on the above model
    rmse, mae = metrics(X_test, y_test, model)

    print("RMSE is {:0.3f} \t Mean absolute error is {:0.3f}".format(rmse[0], mae[0]), file=OutputFile)

    # plotting validation and training loss
    plot_metrics(history)

    # saving model summary
    model.summary(print_fn=myprint)

    OutputFile.close()


if __name__ == "__main__":
    main()


