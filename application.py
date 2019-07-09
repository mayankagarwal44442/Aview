

from __future__ import unicode_literals

import time

from flask import Flask, redirect, url_for, request, render_template

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import time
import pandas as pd
import glob
from bs4 import BeautifulSoup

#Packages for preprocessing
#remember to set keras to theano backend
import pandas as pd
import random
import numpy as np
from sklearn.utils import shuffle
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical


#Packages for model traning and testing
from keras import models
from keras import layers
from keras import regularizers
import matplotlib.pyplot as plt

import os
import pickle

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

import youtube_dl

from deepsegment import DeepSegment
app = Flask(__name__)

def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def predict_if_question(sentence,tk,glove_model,MAX_LEN):
    sentence_nstop = [sentence]
    sentence_nstop = np.array(sentence_nstop)
    sentence_seq = tk.texts_to_sequences(sentence_nstop)
    #print(sentence_seq)
    sentence_seq_trunc = pad_sequences(sentence_seq, maxlen=MAX_LEN)
    return glove_model.predict(sentence_seq_trunc)


def identify_questions(youtube_chats):

	MAX_LEN = 44  # Maximum number of words in a sequence. We truncate if more than this.


	if(os.path.exists('identify_questions.h5') and os.path.exists("obj/tk.pkl")):
		print("Loading model...")
		glove_model = models.load_model('identify_questions.h5')
		tk = load_obj("tk")
		print("Model loaded")

	else:
		questions_df = pd.read_csv('question-pairs-dataset/questions.csv')     #reading questions from questions.csv
		questions_data = list(questions_df['question1'])                       #taking only the column of questions
		questions_data = questions_data[:10664]                                #taking a slice as the other dataset has only 10664 entries

		with open('positive-and-negative-sentences/positive.txt','rb') as file:
		    data_pos = file.read().decode(errors='replace')     #positive comments
		data_pos = data_pos.split('\n')                         #split based on new line
		with open('positive-and-negative-sentences/negative.txt','rb') as file:
		    data_neg = file.read().decode(errors='replace')     #negative comments
		data_neg = data_neg.split('\n')                         #split based on new line

		non_questions_data = data_pos+data_neg                  #combine
		random.shuffle(non_questions_data)                      #shuffle

		questions_data_labels = list(np.ones(len(questions_data)))                  #target vectors for question and non question data
		non_questions_data_labels = list(np.zeros(len(non_questions_data)))

		X = questions_data + non_questions_data
		Y = questions_data_labels + non_questions_data_labels
		X,Y = shuffle(X,Y,random_state=1)                      #shuffling X and Y 

		X = np.array(X)
		Y = np.array(Y)
		X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=31) #split testing and training set
		print('# Train data samples:', X_train.shape[0])
		print('# Test data samples:', X_test.shape[0])
		assert X_train.shape[0] == y_train.shape[0]
		assert X_test.shape[0] == y_test.shape[0]

		NB_WORDS = 10000  # Parameter indicating the number of words we'll put in the dictionary

		tk = Tokenizer(num_words=NB_WORDS,                            #defining tokenizer which tokenizes sentences into
		            filters='!"#$%&()*+,-./:;<=>@[\\]^_`{|}~\t\n', #words. Note that we do not filter out ''?'.
		            lower=True,
		            split=" ")
		tk.fit_on_texts(X_train)

		X_train_seq = tk.texts_to_sequences(X_train)                  #convert texts to sequences(based on incices)
		X_test_seq = tk.texts_to_sequences(X_test)


		save_obj(tk, "tk")

		X_train_seq_trunc = pad_sequences(X_train_seq, maxlen=MAX_LEN)   #pad zeros
		X_test_seq_trunc = pad_sequences(X_test_seq, maxlen=MAX_LEN)

		X_train_seq_trunc[10]  # Example of padded sequence

		le = LabelEncoder()
		y_train_oh = to_categorical(y_train)
		y_test_oh = to_categorical(y_test)

		X_train_emb, X_valid_emb, y_train_emb, y_valid_emb = train_test_split(X_train_seq_trunc, y_train_oh, test_size=0.1, random_state=37)

		#separating validation set from training set

		assert X_valid_emb.shape[0] == y_valid_emb.shape[0]
		assert X_train_emb.shape[0] == y_train_emb.shape[0]

		GLOVE_DIM = 100  # Number of dimensions of the GloVe word embeddings

		if os.path.exists("obj/embedding_dict.pkl"):
			print("loading dict")
			emb_dict = load_obj("embedding_dict")
			print("done loading dict")
		else:
    
			print("Reading glove vectors...")

			glove_file = 'glove.twitter.27B.100d.txt'    #glove file containing words and glove vectors
			emb_dict = {}


			glove = open('glove.twitter.27B/'+ glove_file)
			for line in glove:
			    values = line.split()
			    word = values[0]
			    vector = np.asarray(values[1:], dtype='float32')
			    emb_dict[word] = vector
			glove.close()

			save_obj(emb_dict,"embedding_dict")

			print("Done reading glove vectors...")

        #Constructing embedding mastrix

		emb_matrix = np.zeros((NB_WORDS, GLOVE_DIM))

		for w, i in tk.word_index.items():
			# The word_index contains a token for all words of the training data so we need to limit that
			if i < NB_WORDS:
			    vect = emb_dict.get(w)
			    # Check if the word from the training data occurs in the GloVe word embeddings
			    # Otherwise the vector is kept with only zeros
			    if vect is not None:
			        emb_matrix[i] = vect
			else:
			    break

		glove_model = models.Sequential()
		glove_model.add(layers.Embedding(NB_WORDS, GLOVE_DIM, input_length=MAX_LEN))
		glove_model.add(layers.LSTM(200, dropout_U = 0.2, dropout_W = 0.2))
		glove_model.add(layers.Dense(2, activation='softmax'))

		glove_model.layers[0].set_weights([emb_matrix])
		glove_model.layers[0].trainable = False             #we are using pretrained weights. No need to train the initial layer

		NB_START_EPOCHS = 10  # Number of epochs we usually start to train with
		BATCH_SIZE = 512  # Size of the batches used in the mini-batch gradient descent

		glove_model.compile(optimizer='rmsprop'                   #defining the model's training configs
		                    , loss='categorical_crossentropy'
		                    , metrics=['accuracy'])


		print("training")
		glove_history = glove_model.fit(X_train_emb                               #training the model
		                        , y_train_emb
		                        , epochs=NB_START_EPOCHS
		                        , batch_size=BATCH_SIZE
		                        , validation_data=(X_valid_emb, y_valid_emb)
		                        , verbose=1)

		print("done")

		glove_model.save('identify_questions.h5')
		print("Saved identify_questions.h5")

	youtube_questions = []
	 
	for i in youtube_chats:
	    if(predict_if_question(i,tk,glove_model,MAX_LEN)[0][1]>0.7):
	        youtube_questions.append(i)

	return youtube_questions


def identify_operational(youtube_questions):


        df = pd.read_csv("chats.csv", header = None)
        df.columns = ['index','sentence', 'label', 'chat_session_id', 'created_by_user_id', '_', 'modified_by_user_id','__','status_id']

        df['label'].fillna('N', inplace = True) 

        # df.sort_values(by = ['index'], inplace=True)
        df.reset_index(inplace =True)
        df.drop(columns = ['level_0', 'index', '_', '__'], inplace = True)


        X_tr, X_te, y_train, y_test = train_test_split(df['sentence'], df['label'], random_state=100)

        vector = CountVectorizer(stop_words = 'english')
        X_train_cv = vector.fit_transform(X_tr)
        tfidf_transformer = TfidfTransformer()
        X_train = tfidf_transformer.fit_transform(X_train_cv)

        clf = MultinomialNB().fit(X_train, y_train)
        X_test_cv = vector.transform(X_te)
        X_test = tfidf_transformer.transform(X_test_cv)

        tst = vector.transform(youtube_questions)
        values = tfidf_transformer.transform(tst)
        res = clf.predict(values)

        youtube_operational = []

        for i in range(len(youtube_questions)):
            if(res[i]=='O'):
                youtube_operational.append(youtube_questions[i])

        return youtube_operational
  


def selenium(link):
    

    chromeOptions = Options()
    chromeOptions.add_argument("--ignore-certificate-errors")
    chromeOptions.add_argument("--incognito")
    chromeOptions.add_argument("--headless")

    driver = webdriver.Chrome("./chromedriver_linux64/chromedriver_73", options=chromeOptions)

    driver.get(str(link))
    time.sleep(30)
    driver.find_element_by_tag_name('body').send_keys(Keys.NUMPAD9)
    time.sleep(5)
    driver.find_element_by_tag_name('body').send_keys('k')    # for video pause
    time.sleep(2)
    page_source_overview = driver.page_source
    try:
        iframe = driver.find_element_by_xpath("//iframe[@id='chatframe']")
        driver.switch_to.frame(iframe)
        chats = []
        innerHTML = driver.execute_script("return document.body.innerHTML")
    
        soup = BeautifulSoup(page_source_overview, 'lxml')
        title = soup.title.get_text()
        title = title[:-10]


        for chat in driver.find_elements_by_css_selector('yt-live-chat-text-message-renderer'):
                #author_name = list(chat.find_element_by_css_selector("#author-name").get_attribute('innerHTML').split('<'))
                message = chat.find_element_by_css_selector("#message").get_attribute('innerHTML')
                chats.append(message)
        return chats
    except Exception as e:
        
        return e
    finally:
        driver.quit()

def operational(youtube_chats):
    youtube_questions = identify_questions(youtube_chats)
    youtube_operational = identify_operational(youtube_questions)
    return youtube_operational


""" Route for Processing YouTube live chat files """
@app.route('/process', methods = ['GET', 'POST'])
def process():

    if request.method == 'POST':
        link = request.form.get('chat_link')

        youtube_chats = selenium(link)

        if(isinstance(youtube_chats,(list,))):
            youtube_operational = operational(youtube_chats)
            
            length_chats = len(youtube_chats)
            length_operational = len(youtube_operational)
            length_non_operational = length_chats - length_operational

        return render_template("showgraph.html", length_operational = length_operational, length_non_operational = length_non_operational)
    else:
        return redirect(url_for("index"))

    return render_template("process.html", link = readfiles)

""" Route for Processing Chat files """
@app.route('/chatfile', methods = ['GET', 'POST'])
def chatfile():

    if request.method == "POST":
        chat_path = request.form.get('chat_path')


        with open(chat_path) as file:
            readfiles = file.read().split("\n")

        operational_1 = operational(readfiles)

        # Get total number of Messages, Operational and Non-Operational messages
        length_chat_file = len(readfiles)
        length_operational = len(operational_1)
        length_non_operational = length_chat_file - length_operational

        # questions = identify_questions(readfiles)
        # operational_questions = identify_operational(questions)

        # print("Operational questions: ")
        # print(operational_questions)



        # Call function to create a pie chart showing Operational vs Non-Operational Problems
        return render_template("showgraph.html", length_operational = length_operational, length_non_operational = length_non_operational)

    else:
        return redirect(url_for("index"))


""" Route for Processing video """
@app.route('/video', methods = ['GET', 'POST'])
def video():

    if request.method == 'POST':
        link = request.form.get('video_link')

    ydl_opts = {
     #   'verbose': True,
        'format': 'bestaudio/best',
        'channel': 'mono',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '320',
        }],
        'postprocessor_args': [
            '-ar', '16000',
            '-ac', '1',
            # '-t', '7:00',
            # '-acodec', 'pcm_s16le'
        ],
        'prefer_ffmpeg': True,
        'outtmpl':  'static/audio' + '.%(ext)s',
        # 'keepvideo': True,
        'extractaudio' : True,      # only keep the audio
    }

    print('here')
    if(os.path.exists('static/audio.wav')):
        print('here')
        pass
    else:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([link])



    return render_template("audio.html")


""" Route for Processing transcript files """
@app.route('/transcript', methods = ['GET', 'POST'])
def transcript():

    if request.method == "POST":
        transcript_path = request.form.get('transcript_path')

        with open(transcript_path) as file:
            text = file.read()

        segmenter = DeepSegment('deepsegment_eng_v1/config.json')

        var = segmenter.segment(text)

        operational_1 = operational(var)

        length_chat_file = len(var)
        length_operational = len(operational_1)
        length_non_operational = length_chat_file - length_operational

   
        # Call function to create a pie chart showing Operational vs Non-Operational Problems
        # draw_figure(length_operational, length_non_operational)

        return render_template("showgraph.html", length_operational = length_operational, length_non_operational = length_non_operational)
        
    else:
        return redirect(url_for("index"))    


@app.route('/')
def index():
    return render_template("index.html")

if __name__ == '__main__':
   app.run()
