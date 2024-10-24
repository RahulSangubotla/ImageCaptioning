from flask import Flask, render_template, request
import os
import pickle
import string
import tensorflow
import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.callbacks import History
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from PIL import Image
import os
import string
import matplotlib.pyplot as plt
from pickle import dump
from pickle import load
from keras.applications.xception import Xception #to get pre-trained model Xception
from keras.applications.xception import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.text import Tokenizer #for text tokenization
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import add
from keras.models import Model, load_model
from keras.layers import Input, Dense#Keras to build our CNN and LSTM
from keras.layers import LSTM, Embedding, Dropout
from tqdm import tqdm_notebook as tqdm #to check loop progress
import tensorflow as tf
tqdm().pandas()

app = Flask(__name__)

@app.route('/')
def student():
   return render_template('index.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      """ def load_description(filename):
         mappings = {}
         file = open(filename,'r')
         content = file.readlines()
         file.close()
         for lines in content:
            tokens = lines.split()
            if len(lines)<2:
               continue
            image_id,image_desc = tokens[0].split('.')[0],tokens[1:]
            image_desc = ' '.join(image_desc)
            if image_id not in mappings:
               mappings[image_id] = []
            mappings[image_id].append(image_desc)
         return mappings


      def clean_description(descriptions):
         table = str.maketrans('','',string.punctuation)
         for k,image_descriptions in descriptions.items():
            for i in range(len(image_descriptions)):
               desc = image_descriptions[i]
               desc = desc.split()
               desc = [x.lower() for x in desc]
               desc = [w.translate(table) for w in desc]
               desc = [x for x in desc if len(x)>1]
               desc = [x for x in desc if x.isalpha()]
               image_descriptions[i] = ' '.join(desc)

      def create_corpus(descriptions):
         corpus = set()
         for k in descriptions.keys():
            [corpus.update(x.split()) for x in descriptions[k]]
         return corpus

      def save_descriptions(desc,filename):
         lines = []
         for k,v in desc.items():
            for description in v:
               lines.append(k+' '+description)
         data = '\n'.join(lines)
         file = open(filename,'w')
         file.write(data)
         file.close()

      # load all descriptions
      filename = '/home/siddharth/Documents/Projects/captions/ImaceCaptioning/DataSets/Flickr8k_text/Flickr8k.token.txt'
      descriptions = load_description(filename)
      print('Descriptions loaded: ',len(descriptions))

      # clean the loaded descriptions
      clean_description(descriptions)

      # check the vocabulary length
      vocabulary = create_corpus(descriptions)
      print('Vocabulary length: ',len(vocabulary))
      save_descriptions(descriptions,'/home/siddharth/Documents/Projects/captions/ImaceCaptioning/Image-Captioning-main/descriptions.txt')

      print('SAVED !!!')

      def load_set_of_image_ids(filename):
         file = open(filename,'r')
         lines = file.readlines()
         file.close()
         image_ids = set()
         for line in lines:
            if len(line)<1:
               continue
            image_ids.add(line.split('.')[0])
         return image_ids

      def load_clean_descriptions(all_desc,train_desc_names):
         file = open(all_desc,'r')
         lines = file.readlines()
         descriptions = {}
         for line in lines:
            tokens = line.split()
            image_id,image_desc = tokens[0].split('.')[0],tokens[1:]
            if image_id in train_desc_names:
               if image_id not in descriptions:
                  descriptions[image_id] = []
               desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
               descriptions[image_id].append(desc)
         return descriptions

      def load_image_features(filename,dataset):
         all_features = pickle.load(open(filename,'rb'))
         features = {k:all_features[k] for k in dataset}
         return features

      # load train image ids
      train = '/home/siddharth/Documents/Projects/captions/ImaceCaptioning/DataSets/Flickr8k_text/Flickr_8k.trainImages.txt'
      train_image_ids = load_set_of_image_ids(train)
      print('Training images found: ',len(train_image_ids))

      # load training descriptions
      train_descriptions = load_clean_descriptions('/home/siddharth/Documents/Projects/captions/ImaceCaptioning/Image-Captioning-main/descriptions.txt',train_image_ids)
      print('training descriptions loaded: ',len(train_descriptions))

      # load training image features
      train_features = load_image_features('/home/siddharth/Documents/Projects/captions/ImaceCaptioning/Image-Captioning-main/Flicker_dataset_image_features.pkl',train_image_ids)
      print('training features loaded: ',len(train_features))

      def to_list(descriptions):
         all_desc_list = []
         for k,v in descriptions.items():
            for desc in v:
               all_desc_list.append(desc)
         return all_desc_list

      def tokenization(descriptions):
         # list of all the descriptions
         all_desc_list = to_list(descriptions)  
         tokenizer = Tokenizer()
         tokenizer.fit_on_texts(all_desc_list)
         return tokenizer

      # create tokenizer
      tokenizer = tokenization(train_descriptions)

      # word index is the dictionary /mappings of word-->integer
      vocab_size = len(tokenizer.word_index)+1
      print('Vocab size: ',vocab_size)

      def max_length(descriptions):
         all_desc_list = to_list(descriptions)
         return (max(len(x.split()) for x in all_desc_list))


      def create_sequences(tokenizer,desc_list,max_len,photo):
         X1,X2,y = [],[],[]
         # X1 will contain photo
         # X2 will contain current sequence
         # y will contain one hot encoded next word

         for desc in desc_list:
            # tokenize descriptions
            seq = tokenizer.texts_to_sequences([desc])[0]
            for i in range(1,len(seq)):
            # out seq is basically the next word in the sentence
               in_seq,out_seq = seq[:i],seq[i]
            # pad input sequence
               in_seq = pad_sequences([in_seq],maxlen=max_len)[0]
            # one hot encode output sequence
               out_seq = to_categorical([out_seq],num_classes=vocab_size)[0]
               X1.append(photo)
               X2.append(in_seq)
               y.append(out_seq)
         return np.array(X1),np.array(X2),np.array(y)

      # maximum length that a description can have OR the biggest description we are having
      max_len = max_length(train_descriptions)
      print(max_len)

      def int2word(tokenizer,integer):
         for word,index in tokenizer.word_index.items():
            if index==integer:
               return word
         return None

      def predict_desc(model,tokenizer,photo,max_len):
         in_seq = 'startseq'
         for i in range(max_len):
            seq = tokenizer.texts_to_sequences([in_seq])[0]
            seq = pad_sequences([seq],maxlen=max_len)
            y_hat = model.predict([photo,seq],verbose=0)
            y_hat = np.argmax(y_hat)
            word = int2word(tokenizer,y_hat)
            if word==None:
               break
            in_seq = in_seq+' '+word
            if word=='endseq':
               break
         return in_seq
 """
      img_to_test = request.files['image']
      img = cv2.imdecode(np.frombuffer(img_to_test.read(), np.uint8), 1)
      #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      img_to_test = "static/assets/img/user.jpeg"
      cv2.imwrite(img_to_test,img)
      #plt.imshow(img)
      def extract_features(filename, model):
         try:
                  image = Image.open(filename)
         except:
                  print("ERROR: Can't open image! Ensure that image path and extension is correct")
         image = image.resize((299,299))
         image = np.array(image)
         # for 4 channels images, we need to convert them into 3 channels
         if image.shape[2] == 4:
                  image = image[..., :3]
         image = np.expand_dims(image, axis=0)
         image = image/127.5
         image = image - 1.0
         feature = model.predict(image)
         return feature
      def word_for_id(integer, tokenizer):
         for word, index in tokenizer.word_index.items():
            if index == integer:
                  return word
         return None
      def generate_desc(model, tokenizer, photo, max_length):
         in_text = ''
         for i in range(32):
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=max_length)
            pred = model.predict([photo,sequence], verbose=0)
            pred = np.argmax(pred)
            word = word_for_id(pred, tokenizer)
            #print(word)
            if word is None:
                  break
            in_text += ' ' + word
            if word == 'end':
                  break
         return in_text
      max_length = 32
      tokenizer = load(open("tokenizer2.p","rb"))
      model = load_model('models/model_9.h5')
      xception_model = Xception(include_top=False, pooling="avg")
      photo = extract_features(img_to_test, xception_model)
      img = Image.open(img_to_test)
      description = generate_desc(model, tokenizer, photo, max_length)
      print(description)
      hsh = []
      stopwords = "a, an, and, are, as, at, be, but, by, others, for, if, in, into, while, is, it, no, not, down, of, on, or, such, that, the, their, then, there, blowing, these, they, this, to, was, will, with, in, his, its"
      stopwords = stopwords.split(", ")
      def remove_dup_word(string):
         # Used to split string around spaces.
         words = string.split()
         
         # To store individual visited words
         
         #print("Caption:")
         # Traverse through all words
         i=0
         for word in words:
            # If current word is not seen before.
            if word not in hsh and i<5:
                  #print(word, end=" ")
                  hsh.append(word)
                  i+=1
                  continue
            
            #print(word,end=" ")
            hsh.append(word)
            if word in stopwords:
                  continue
            else: break
      remove_dup_word(description)
      print(hsh)
      text = ' '.join([str(element) for element in hsh])
      """ def extract_features(filename):
         # load the model
         model = VGG16()
         # re-structure the model
         model.layers.pop()
         model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
         # load the photo
         image = load_img(filename, target_size=(224, 224))
         # convert the image pixels to a numpy array
         image = img_to_array(image)
         # reshape data for the model
         image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
         # prepare the image for the VGG model
         image = preprocess_input(image)
         # get features
         feature = model.predict(image, verbose=0)
         confidence = round(100*(np.max(feature[0])), 2)
         print('confi',confidence)
         return feature

      # pre-define the max sequence length (from training)
      maxLength = 34
      # load the model
      model = load_model('models/model_9.h5')
      # load and prepare the photograph
      photo = extract_features(img_to_test)
      # generate description
      description = predict_desc(model, tokenizer, photo, maxLength)
 """
      description = "The generated caption is: "+ "   " + text.capitalize()
   return render_template("index2.html",result = description)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)