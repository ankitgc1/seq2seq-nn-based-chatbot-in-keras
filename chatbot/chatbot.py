from keras.preprocessing.text import Tokenizer
from keras.layers import Input, LSTM, Dense, Embedding, TimeDistributed
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.preprocessing.sequence import pad_sequences
# from keras.utils import plot_model
import numpy as np

with open("encoder_inputs.txt", 'r') as f:
    encoder_inputs = f.readlines()
    encoder_inputs = encoder_inputs[0:15000]

with open("decoder_inputs.txt", 'r') as f:
    decoder_inputs = f.readlines()
    decoder_inputs = decoder_inputs[0:15000]

VOCAB_SIZE = 17737
MAX_LEN = 10
HIDDEN_DIM=300
BATCH_SIZE = 128
EPOCHS = 50
EMBEDDING_DIM = 100

def vocab_creater(encoder_inputs, decoder_inputs, VOCAB_SIZE):

    tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(encoder_inputs)
    tokenizer.fit_on_texts(decoder_inputs)
    dictionary = tokenizer.word_index

    word2idx = {}
    idx2word = {}
    for k, v in dictionary.items():
        if v < VOCAB_SIZE:
            word2idx[k] = v
            idx2word[v] = k
        if v >= VOCAB_SIZE-1:
            continue

    return word2idx, idx2word

word2idx, idx2word = vocab_creater(encoder_inputs, decoder_inputs, VOCAB_SIZE=VOCAB_SIZE)

def text2seq(encoder_text, decoder_text, VOCAB_SIZE):

    tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(encoder_text)
    tokenizer.fit_on_texts(decoder_text)
    encoder_sequences = tokenizer.texts_to_sequences(encoder_text)
    decoder_sequences = tokenizer.texts_to_sequences(decoder_text)

    return encoder_sequences, decoder_sequences

encoder_sequences, decoder_sequences = text2seq(encoder_inputs, decoder_inputs, VOCAB_SIZE)

def padding(encoder_sequences, decoder_sequences, MAX_LEN):

    encoder_input_data = pad_sequences(encoder_sequences, maxlen=MAX_LEN, dtype='int32', padding='post', truncating='post')
    decoder_input_data = pad_sequences(decoder_sequences, maxlen=MAX_LEN, dtype='int32', padding='post', truncating='post')

    return encoder_input_data, decoder_input_data

encoder_input_data, decoder_input_data = padding(encoder_sequences, decoder_sequences, MAX_LEN)

num_samples = len(encoder_sequences)
print("encoder_input_data:- ", encoder_input_data.shape)
print("decoder_input_data:- ", decoder_input_data.shape)
print("num_samples:- ", num_samples)

# GLOVE_DIR = path for glove.6B.100d.txt
embeddings_index = {}
f = open("glove.6B.100d.txt")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

# this time: embedding_dimention = 100d
def embedding_matrix_creater(embedding_dimention, embeddings_index, word_index):
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dimention))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

embedding_matrix = embedding_matrix_creater(EMBEDDING_DIM, embeddings_index, word2idx)

def embedding_layer_creater(VOCAB_SIZE, EMBEDDING_DIM, MAX_LEN, embedding_matrix):

    embedding_layer = Embedding(input_dim = VOCAB_SIZE,
                                output_dim = EMBEDDING_DIM,
                                input_length = MAX_LEN,
                                weights = [embedding_matrix],
                                trainable = False)
    return embedding_layer

embed_layer = embedding_layer_creater(VOCAB_SIZE, EMBEDDING_DIM, MAX_LEN, embedding_matrix)
embed_layer.build((None,))
embed_layer.set_weights([embedding_matrix])

VOCAB_SIZE = 17738

def decoder_output_creater(decoder_input_data, num_samples, MAX_LEN, VOCAB_SIZE):

    decoder_output_data = np.zeros((num_samples, MAX_LEN, VOCAB_SIZE), dtype="float32")

    for i, seqs in enumerate(decoder_input_data):
        for j, seq in enumerate(seqs):
            if j > 0:
                decoder_output_data[i][j][seq] = 1.
    print(decoder_output_data.shape)

    return decoder_output_data

decoder_output_data = decoder_output_creater(decoder_input_data, num_samples, MAX_LEN, VOCAB_SIZE)

from sklearn.model_selection import train_test_split
en_train, en_test, de_train, de_test = train_test_split(encoder_input_data, decoder_input_data, test_size=0.2)
#en_train, en_val, de_train, de_val = train_test_split(en_train, de_train, test_size=test_size2)



def seq2seq_model_builder(HIDDEN_DIM, MAX_LEN, vocab_size, embed_layer):

    encoder_inputs = Input(shape=(MAX_LEN, ), dtype='int32')
    encoder_embedding = embed_layer(encoder_inputs)
    # encoder_embedding = Embedding(input_dim = vocab_size, output_dim=HIDDEN_DIM, input_length = MAX_LEN)(encoder_inputs)
    encoder_LSTM = LSTM(HIDDEN_DIM, return_state=True)
    encoder_outputs, state_h, state_c = encoder_LSTM(encoder_embedding)

    decoder_inputs = Input(shape=(MAX_LEN, ), dtype='int32',)
    decoder_embedding = embed_layer(decoder_inputs)
    # decoder_embedding = Embedding(input_dim = vocab_size, output_dim=HIDDEN_DIM, input_length = MAX_LEN)(decoder_inputs)
    decoder_LSTM = LSTM(HIDDEN_DIM, return_state=True, return_sequences=True)
    decoder_outputs, _, _ = decoder_LSTM(decoder_embedding, initial_state=[state_h, state_c])

    outputs = TimeDistributed(Dense(VOCAB_SIZE, activation='softmax'))(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], outputs)
    # plot_model(model, to_file='mk1.png')

    return model

model = seq2seq_model_builder(HIDDEN_DIM, MAX_LEN, VOCAB_SIZE, embed_layer)
model.summary()
model.compile(loss="categorical_crossentropy", optimizer='Adam', metrics=['accuracy'])
model.fit([encoder_input_data, decoder_input_data], decoder_output_data, batch_size=BATCH_SIZE, epochs=EPOCHS)
model.save("model.h5")
print("model saved succesfully")
