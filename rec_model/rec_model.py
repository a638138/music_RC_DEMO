import warnings
import os
warnings.simplefilter(action='ignore', category=FutureWarning)
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import requests
import pydot
import pydotplus
import pandas as pd
import numpy as np
import argparse
import keras
import random
import tensorflow as tf
import time
from rec_model.neg_sampling import neg_sampling
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from keras.utils.vis_utils import model_to_dot
from tqdm import tqdm
from keras.activations import softmax, tanh, relu, sigmoid
from keras.models import Model
from keras.utils import to_categorical, plot_model
from keras.callbacks import ModelCheckpoint
from keras.backend.tensorflow_backend import set_session
from keras.losses import categorical_crossentropy
from keras.layers.core import Activation
from keras.layers import Input, Dense, Dropout, Concatenate, Reshape, Bidirectional, Embedding, GRU, Dot, Lambda, Multiply, LeakyReLU, BatchNormalization, Add, CuDNNGRU

# 請修改此處!!
# 歌曲資訊
# 過濾後為 song_art_mapping(filtered).csv
# 重複消費事件為 song_art_mapping(repeat).csv
# 將其從data_process\data_preprocessing中放入此資料夾中
song_mapping = pd.read_csv('rec_model/song_art_mapping.csv')

# 請修改此處!!
# 填入要放置的資料集
# 過濾後為 output.hdf
# 重複消費事件為 output_repeat.hdf
PATH_DATASET = 'rec_model/dataset.hdf'
train = pd.read_hdf(PATH_DATASET, key='train')
valid = pd.read_hdf(PATH_DATASET, key='valid')
test = pd.read_hdf(PATH_DATASET, key='test')
data = pd.concat([train, valid, test])
item_pop = data['item_id'].value_counts(sort=False) / len(data)
user_item_pop = data.item_id.value_counts(sort=False).sort_index() / data.user_id.nunique()

class UserDataIterator:

    def __init__(self, dataset, song_num, newthing_threshold=3):
        self.dataset = dataset  
        self.user_key = 'user_id'
        self.item_key = 'item_id'
        self.n_items = song_num
        self.session_key = 'session_id'
        self.newthing_threshold = newthing_threshold
        self.session_num = self.dataset.session_id.nunique()

    def __iter__(self):

        df = self.dataset
        user_item_recency = np.ones((1, self.n_items+1), dtype=float)*self.newthing_threshold
        user_item_selected = np.zeros((1, self.n_items+1), dtype=bool)
        session_set = sorted(df.session_id.unique())

        sstart = np.ones((1, 1))
        ustart = np.ones((1, 1))

        for session_id in tqdm(session_set, total=len(session_set)):
            s_data = df[df.session_id==session_id].copy()
            s_data.sort_values('counter', inplace=True, ignore_index=True)
            for i in range(len(s_data)):
                id_input = np.array([s_data.iat[i, 4]])
                id_input_oh = to_categorical(id_input, num_classes=self.n_items+1)
                id_target = np.array([s_data.iat[i, 4]])

                # Update recency
                selected_song_idx = id_input
                user_item_recency[0, selected_song_idx] = 0
                user_item_selected[0, selected_song_idx] = True
                user_item_recency[user_item_selected] += 1
                
                # Clean out of windows
                out_of_windows = user_item_recency > self.newthing_threshold
                user_item_selected[out_of_windows] = False
                user_item_recency[out_of_windows] = self.newthing_threshold
                
                # Get user-item recency
                user_info = np.array([user_item_selected[0]])
                
                yield id_input_oh, id_target, sstart, ustart, user_info

                # Reset sstart and ustart
                sstart = sstart * 0
                ustart = ustart * 0

            sstart = np.ones((1, 1))

class HRNNre:

    def __init__(self, kwargs):
        self.songs_size = kwargs['song_num']
        self.batch_size = kwargs['batch_size']
        self.h_size = kwargs['h_size']
        self.mode = kwargs['mode']
        self.newthing_layer_size = kwargs['newthing_layer_size']
        self.model_path = kwargs['model_path']
        self.model = self.create_model()

    def cross_entropy(self, y_true, y_pred):
        loss = tf.math.reduce_mean(-tf.math.log(y_pred[:, 0]+1e-15))
        return loss

    def userGRU_handler(self, x):
        pre_hu = x[0]
        new_hu = x[1]
        sstart = x[2]
        ustart = x[3]
        new_hu1 = pre_hu*(1-sstart)+new_hu*sstart
        new_hu2 = K.zeros_like(new_hu1)*ustart+new_hu1*(1-ustart)
        return new_hu2

    def sessionGRU_handler(self, x):
        pre_hs = x[0]
        sstart = x[1]
        h_s_init = x[2]
        ustart = x[3]
        h_s_init1 = pre_hs*(1-sstart)+h_s_init*sstart
        h_s_init2 = h_s_init1*(1-ustart)+K.zeros_like(h_s_init1)*ustart
        return h_s_init2

    def create_model(self):

        songs_size = self.songs_size
        batch_size = self.batch_size
        h_size = self.h_size
        mode = self.mode
        newthing_layer_size = self.newthing_layer_size

        sstart = Input(batch_shape=(batch_size, 1), name='sstart')
        ustart = Input(batch_shape=(batch_size, 1), name='ustart')
        pre_hs = Input(batch_shape=(batch_size, h_size), name='pre_hs')
        pre_hu = Input(batch_shape=(batch_size, h_size), name='pre_hu')

        # User GRU
        pre_hs_ = Lambda(K.expand_dims, arguments={'axis':1})(pre_hs)
        new_hu = GRU(h_size, name='userGRU')(pre_hs_, initial_state=pre_hu)
        new_hu_ = Lambda(self.userGRU_handler, name='userGRU_handler')([pre_hu, new_hu, sstart, ustart])
        h_s_init = Dense(h_size, activation='tanh')(new_hu_)
        h_s_init_ = Lambda(self.sessionGRU_handler, name='sessionGRU_handler')([pre_hs, sstart, h_s_init, ustart])

        # song model
        track_id_inputs = Input(batch_shape=(batch_size, songs_size+1), name='track_id') 
        track_id_inputs_ = Lambda(K.expand_dims, arguments={'axis':1})(track_id_inputs)
        new_hs = GRU(h_size, name='idGRU')(track_id_inputs_, initial_state=h_s_init_)

        # User repeat info
        user_info_input = Input(batch_shape=(batch_size, songs_size+1), name='user_info')
        user_info_dense2 = Dense(newthing_layer_size)(user_info_input)
        user_info_dense2_ = Lambda(Activation(tanh))(user_info_dense2)
        concat = Concatenate()([new_hs, user_info_dense2_])

        # Concatenate info
        dropout_song_ep = Lambda(K.expand_dims, arguments={'axis':1})(concat)

        # Negative sampling
        target_input = Input(batch_shape=(batch_size, None), name='target_input')
        neg_sampling_layer = neg_sampling(songs_size+1, h_size+newthing_layer_size,  mode=mode)([target_input, dropout_song_ep])
        predictions = Lambda(Activation(softmax))(neg_sampling_layer)
        pred_model = Model(inputs=[sstart, ustart, pre_hs, pre_hu, track_id_inputs, user_info_input, target_input], outputs=[predictions, new_hu_, new_hs])

        pred_model.load_weights(self.model_path)

        return pred_model

    def get_states(self, model):
        return [K.get_value(s) for s, _ in model.state_updates]

    def get_songs(self, user_data, alpha, recall_k=[5], mrr_k=[5]):
        test_generator = UserDataIterator(user_data, 99681)
        model_to_predict = self.model
        pre_hu = np.zeros((1, self.h_size))
        pre_hs = np.zeros((1, self.h_size))
        rec_idx = None
        for ids, target, sstart, ustart, user_info in test_generator:
            input = {
                'sstart':sstart,
                'ustart':ustart,
                'pre_hu':pre_hu,
                'pre_hs':pre_hs,
                'track_id':ids,
                'user_info':user_info,
                'target_input':target,
            }

            # 開始預測
            pred, pre_hu, pre_hs = model_to_predict.predict(input, batch_size=1)
            pred[:, 1:] = pred[:, 1:] *(1-alpha) + (1-user_item_pop.values)*alpha
            data = pred.copy()[0]
            ind = np.argpartition(data, -5)[-5:]
            rec_idx = ind[np.argsort(data[ind])][::-1][:5]

        return rec_idx

# 利用google youtube data api根據method抓取相關資料
def youtube_api(method, args):

    API_KEY_list = []
    num = random.randint(0, len(API_KEY_list)-1)
    API_KEY = API_KEY_list[num]
    payload = dict()
    # Header & URL
    headers = {'user-agent': 'DataRequest', 'Connection' : 'close'}
    # Add API key and format to the payload
    payload['key'] = API_KEY
    payload.update(args)
    url = f'https://www.googleapis.com/youtube/v3/{method}'
    response = requests.get(url, headers=headers, params=payload)
    return response

# 取得歌曲youtube的資訊
def get_movie_info(rec_songs_ids):
    movie_info_list = list()
    for song_id in rec_songs_ids:
        song_info = song_mapping[song_mapping.new_song_id == song_id]
        song_name =  song_info['song_name'].values
        art_name =  song_info['art_name'].values

        if len(song_name) != 0:
            args = dict()
            args['q'] = f'{song_name} {art_name}'
            args['part'] = 'snippet'
            args['type'] = 'video'
            args['maxResults'] = 1
            movie_info_response = youtube_api('search', args)
            if movie_info_response.status_code != 200:
                return None

            movie_info_response = movie_info_response.json()
            movie_info = dict()
            video_id = movie_info_response['items'][0]['id']['videoId']          
            movie_info['song_id'] = song_id
            movie_info['movie_id'] = video_id
            movie_info['title'] = movie_info_response['items'][0]['snippet']['title']
            movie_info['desc'] = movie_info_response['items'][0]['snippet']['description']
            movie_info['thumbnail'] = movie_info_response['items'][0]['snippet']['thumbnails']['high']['url']
            movie_info_list.append(movie_info)

    return movie_info_list

# 根據熱門程度進行推薦的模型
def pop_predict():

    predict_song_id = np.random.choice(item_pop.index, size=5, replace=False, p=item_pop)
    movie_info_list = get_movie_info(predict_song_id)
    return movie_info_list

# 利用HRNNre模型進行推薦
def HRNN_predict(user_data):

    # 主要參數的部分
    kwargs = {
        'song_num':99681,
        'batch_size':1,
        'h_size':100,
        'mode':'test',
        'newthing_layer_size':100,
        'model_path':'rec_model/model.h5'
    }
    alpha = 0.1
    model = HRNNre(kwargs)
    rec_songs = model.get_songs(user_data, alpha)
    movie_info_list = get_movie_info(rec_songs)
    K.clear_session()
    return movie_info_list


        



        





