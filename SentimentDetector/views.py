from django.shortcuts import render
from keras.datasets import imdb
from keras import models 
import numpy as np
from keras.utils import pad_sequences



def vectorization(data, number_of_features = 10000): 
  res = np.zeros((len(data), number_of_features))
  for i, j in enumerate(data): 
    res[i, j] = 1 
  return res

def sentiment_detector(request): 
    context = {}
    context['predict_res'] = False 
    context['empty_comment'] = False 
    context['short_comment'] = False 
    if request.method == 'POST':
        print("====" * 5)
        print(request.POST['review_content'])
        print("====" * 5)
        fcn_imdb_model = models.load_model('IMDB_REVIEW_FCN.h5')
        rnn_imdb_model = models.load_model('RNN_IMDB_REVIEW.h5')
        lstm_imdb_model = models.load_model('LSTM_IMDB_REVIEW_MODEL.h5')
        comment_string = request.POST['review_content'].lower()
        if comment_string == "":
            context['empty_comment'] = True 
        # elif len(comment_string.split(" ")) < 10:
        #     context['short_comment'] = True
        else: 
            imdb_data_dict = imdb.get_word_index()
            seq_data = []
            string_stream_lst = comment_string.split(" ")
            not_found_in_dict_lst = []
            for seq in string_stream_lst: 
                if seq in imdb_data_dict: 
                    seq_data.append(imdb_data_dict.get(seq))
                else: 
                    not_found_in_dict_lst.append(seq)
            seq_ndarry = np.array(seq_data, dtype = 'float32')
            fcn_vector_data = vectorization(seq_data)
            seq_ndarry = pad_sequences([seq_ndarry], maxlen=500)
            # seq_data = seq_ndarry.reshape(1, seq_ndarry.shape[0])
            print(seq_ndarry)
            rnn_res = rnn_imdb_model.predict(seq_ndarry)[0]
            lstm_res = lstm_imdb_model.predict(seq_ndarry)[0]
            fcn_res = fcn_imdb_model.predict(fcn_vector_data)[0]
            rnn_res = True if rnn_res[0] > 0.5 else False 
            lstm_res = True if lstm_res[0] > 0.5 else False 
            fcn_res = True if fcn_res[0] > 0.5 else False 
            context['predict_res'] = True 
            context['rnn_res'] = rnn_res 
            context['lstm_res'] = lstm_res
            context['fcn_res'] = fcn_res
    return render(request,'sentiment_detector.html', context)