import img_cap_text as t
import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
import string
import os
from PIL import Image
import glob
#from pickle import dump, load
import pickle
from time import time
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,\
                         Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization, Concatenate
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import add
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras import Input, layers
from keras import optimizers
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras import backend as K
from nltk.translate.bleu_score import corpus_bleu
from keras import regularizers

def load_set(filename):
    doc = t.load_doc(filename)
    dataset = list()
    for line in doc.split('\n'):
        if(len(line)<1):
            continue

        identifier = line.split('.')[0]
        dataset.append(identifier)
    return (dataset)

def load_clean_desc(filename,dataset):
    doc = t.load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        image_id, image_desc = tokens[0], tokens[1:]
        if image_id in dataset:
            if image_id not in descriptions:
                descriptions[image_id] = list()

            desc = 'startseq '+' '.join(image_desc)+ ' endseq'
            descriptions[image_id].append(desc)
    
    return descriptions
             
train_img = []

def preprocess(image_path):
    img_preprocess = image.load_img(image_path, target_size=(299, 299))
    
    x = image.img_to_array(img_preprocess)
    
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x;

def preproces_test(image_path):
    img_preprocess = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img_preprocess)
    img_preprocess.save('resized_image.jpg')
    x = np.expand_dims(x, axis=0)
    return x;

model = InceptionV3(weights='imagenet')
model_new = Model(model.input, model.layers[-2].output)
def encode(image):
    image = preprocess(image) 
    #print("image shape {}".format(image.shape))
    fea_vec = model_new.predict(image) 
    #print("feature vector shpe: {}".format(fea_vec.shape))
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1]) 
    #print("feature vector shpe aft reshape: {}".format(fea_vec.shape))
    return fea_vec
    
def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc

def max_length(descriptions):
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)

def data_generator(descriptions, photos, wordtoix, max_length, num_photos_per_batch,vocab_size):
    X1, X2, y = list(), list(), list()
    n=0
    # loop for ever over images
    while 1:
        for key, desc_list in descriptions.items():
            n+=1
            # retrieve the photo feature
            photo = photos[key+'.jpg']
            for desc in desc_list:
                # encode the sequence
                seq = [wordtoix[word] for word in desc.split(' ') if word in wordtoix]
                # split one sequence into multiple X, y pairs
                for i in range(1, len(seq)):
                    # split into input and output pair
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    # encode output sequence
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    # store
                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(out_seq)
            # yield the batch data
            if n==num_photos_per_batch:
                yield [[array(X1), array(X2)], array(y)]
                X1, X2, y = list(), list(), list()
                n=0

def greedySearch(model,photo,wordtoix,ixtoword,max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequenc = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        
        sequenc = pad_sequences([sequenc], maxlen=max_length)
        
        yhat = model.predict([photo,sequenc], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        
        if word == 'endseq':
            break
    #final = in_text.split()
    #final = final[1:-1]
    #final = ' '.join(final)
    print(in_text)
    return in_text

def evaluate_model(model, descriptions, photos, wordtoix,ixtoword, max_length):
    actual, predicted = list(), list()
    for key, desc_list in descriptions.items():
        key=key+'.jpg'
        image = photos[key].reshape((1,2048))
        yhat = greedySearch(model,  image,wordtoix,ixtoword, max_length)
        references = [d.split() for d in desc_list]
        actual.append(references)
        predicted.append(yhat.split())
    	
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

def beam_search_predictions(image,max_len,model,wordtoix,ixtoword, beam_index):
    start = [wordtoix["startseq"]]
    
    start_word = [[start, 0.0]]
    
    while len(start_word[0][0]) < max_len:
        temp = []
        for s in start_word:
            par_caps = sequence.pad_sequences([s[0]], maxlen=max_len, padding='post')
            #e = encoding_test[image[len(images):]]
            preds = model.predict([image, np.array(par_caps)])
            
            word_preds = np.argsort(preds[0])[-beam_index:]
            
            # Getting the top <beam_index>(n) predictions and creating a 
            # new list so as to put them via the model again
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])
                    
        start_word = temp
        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Getting the top words
        start_word = start_word[-beam_index:]
    
    start_word = start_word[-1][0]
    intermediate_caption = [ixtoword[i] for i in start_word]

    final_caption = []
    
    for i in intermediate_caption:
        if i != 'endseq':
            final_caption.append(i)
        else:
            break
    
    final_caption = ' '.join(final_caption[1:])
    return final_caption

def main():
    filename = 'Flicker8k_text\Flickr_8k.trainImages.txt'
    filename_test = 'Flicker8k_text\Flickr_8k.testImages.txt'
    train = load_set(filename)
    test = load_set(filename_test) 
    print('Dataset: {}'.format(len(train)))
    
    images = 'Flicker8k_Dataset/'
    img = glob.glob(images+'*.jpg')
    
    train_images = set(open(filename,'r').read().strip().split('\n'))
    
    
    for i in img:
        if i[(len(images)):] in train_images:
            train_img.append(i)
    
    
    test_images_file = 'Flicker8k_text\Flickr_8k.testImages.txt'
    
    test_images = set(open(test_images_file, 'r').read().strip().split('\n'))
    
    # Create a list of all the test images with their full path names
    test_img = []
    
    for i in img: 
        if i[len(images):] in test_images:
            test_img.append(i) 
    
    #print("Train Images: {}".format(train_img))
    '''image_test=[]
    
    image_test.append(preproces_test("Flicker8k_Dataset\\10815824_2997e03d76.jpg"))
    print("Size after resize: {}".format(image_test[0][0].shape))'''
        
    
    
    encoding_train = {}
    for img in train_img:
        encoding_train[img[len(images):]] = encode(img)
    
    with open("encoded_train_images.pkl", "wb") as encoded_pickle:
        pickle.dump(encoding_train, encoded_pickle)
        
    start = time()
    encoding_test = {}
    for img in test_img:
        encoding_test[img[len(images):]] = encode(img)
    print("Time taken in seconds =", time()-start)
    
    with open("encoded_test_images.pkl", "wb") as encoded_pickle:
        pickle.dump(encoding_test, encoded_pickle)
    
    train_features = pickle.load(open("encoded_train_images.pkl", "rb"))
    print('Photos: train=%d' % len(train_features))
    test_features = pickle.load(open("encoded_test_images.pkl", "rb"))
    print('Photos: test=%d' % len(test_features))
    
    filename = 'Flicker8k_text\Flickr_8k.testImages.txt'
    test_images = set(open(filename,'r').read().strip().split('\n'))
    test_img = []

    for i in img:
        if i[len(images):] in test_images:
            test_img.append(i)
    #print("Test Images: {}".format(test_img))
    train_descriptions = load_clean_desc('descriptions.txt',train)
    #print(train_descriptions)
    test_descriptions = load_clean_desc('descriptions.txt', test)
    print('Descriptions: test=%d' % len(test_descriptions))
    print('Descriptions: training={}'.format(len(train_descriptions)))
    all_train_captions = []
    for key,val in train_descriptions.items():
        for cap in val:
            all_train_captions.append(cap)
    print(len(all_train_captions))

    word_count_threshold = 5
    word_counts = {}

    for caption in all_train_captions:
        
        for w in caption.split(' '):
            word_counts[w]=word_counts.get(w,0)+1
            
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print("Length of Vocab with 10 threshold: {}".format(len(vocab)))
    
    
    '''word_counts = set()        
    for caption in all_train_captions:
        
        for w in caption.split(' '):
            word_counts.add(w)
            
    #vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    vocab = [w for w in word_counts]
    print("Length of Vocab with 10 threshold: {}".format(len(vocab)))'''
    
    
    
    ixtoword = {}
    wordtoix = {}
    
    ix = 1
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1
    
    vocab_size = len(ixtoword) + 1 # one for appended 0's
    #print(vocab_size)
    #print(wordtoix)
    
    max_len = max_length(train_descriptions)
    print('Description Length: %d' % max_len)
    
    glove_dir='Glove_Dataset'
    embeddings_index = {} # empty dictionary
    f = open(os.path.join(glove_dir, 'glove.6B.200d.txt'), encoding="utf-8")
    
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))
    
    embedding_dim = 200

    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    
    for word, i in wordtoix.items():
    
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    
    #print(embedding_matrix.shape)
    #print(embedding_matrix)
    
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.01))(fe1)
    inputs2 = Input(shape=(max_len,))
   
    se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256,kernel_regularizer=regularizers.l2(0.01))(se2)
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.01))(decoder1)
    outputs = Dense(vocab_size, activation='softmax',kernel_regularizer=regularizers.l2(0.01))(decoder2)
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    #model.summary()

    #print(model.layers[2])

    model.layers[2].set_weights([embedding_matrix])
    model.layers[2].trainable = False
    
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    epochs = 20
    number_pics_per_bath = 60
    steps = len(train_descriptions)//number_pics_per_bath
    print(steps)
    print(len(train_descriptions))
    
    for i in range(epochs):
        generator = data_generator(train_descriptions, train_features, wordtoix, max_len, number_pics_per_bath,vocab_size)
        model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
    
    for i in range(epochs):
        generator = data_generator(train_descriptions, train_features, wordtoix, max_len, number_pics_per_bath,vocab_size)
        model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
    
    K.set_value(model.optimizer.lr,0.0001)
    epochs = 20
    number_pics_per_bath = 60
    steps = len(train_descriptions)//number_pics_per_bath
    
    for i in range(epochs):
        generator = data_generator(train_descriptions, train_features, wordtoix, max_len, number_pics_per_bath,vocab_size)
        model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
    
    model.save_weights('model_weights/model_weights.h5')
    
    model.load_weights('model_weights/model_weights.h5')
    images = "Flicker8k_Dataset"
    
    with open("encoded_test_images.pkl", "rb") as encoded_pickle:
        encoding_test = pickle.load(encoded_pickle)
    
    
    '''pic = list(encoding_test.keys())[202]
    print(pic)
    pic = '1394368714_3bc7c19969.jpg'
    encoding_test={}
    #pic='cats.jpg'
    x = plt.imread(pic)
    plt.imshow(x)
    plt.show()
    encoding_test[pic] = encode(pic)
    image = encoding_test[pic].reshape((1,2048))
    
    image = encoding_test[pic].reshape((1,2048))
    x=plt.imread(images+'/'+pic)
    plt.imshow(x)
    plt.show()
    print(image.shape)
    print(image)
    print("Greedy:",greedySearch(model,image,wordtoix,ixtoword,max_len))
    print("BEAM:",beam_search_predictions(image,max_len,model,wordtoix,ixtoword,3))
    print("BEAM:",beam_search_predictions(image,max_len,model,wordtoix,ixtoword,5))
    print("BEAM:",beam_search_predictions(image,max_len,model,wordtoix,ixtoword,7))
    #print(test_features)'''
    evaluate_model(model,test_descriptions,test_features,wordtoix,ixtoword,max_len)
if __name__ == "__main__":
    main()
