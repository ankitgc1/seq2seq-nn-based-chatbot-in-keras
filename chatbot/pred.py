from keras.preprocessing.text import Tokenizer
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np

VOCAB_SIZE = 17737
MAX_LEN = 10

with open("encoder_inputs.txt", 'r') as f:
    encoder_inputs = f.readlines()
    encoder_inputs = encoder_inputs[0:15000]

with open("decoder_inputs.txt", 'r') as f:
    decoder_inputs = f.readlines()
    decoder_inputs = decoder_inputs[0:15000]

print("encoder_inputs:- ", encoder_inputs[:5])
print("decoder_inputs:- ", decoder_inputs[:5])

que = ["how are you"]

# def vocab_creater(text_lists, VOCAB_SIZE, que):
# text_lists = encoder_inputs + decoder_inputs
# print("text_lists:- ", text_lists[:5])
tokenizer = Tokenizer(num_words=VOCAB_SIZE)
tokenizer.fit_on_texts(encoder_inputs)
tokenizer.fit_on_texts(decoder_inputs)
encoder_sequences = tokenizer.texts_to_sequences(que)
encoder_input_data = pad_sequences(encoder_sequences, maxlen=MAX_LEN, dtype='int32', padding='post', truncating='post')
# encoder_input_data = np.array(encoder_input_data).reshape((10,))
dictionary = tokenizer.word_index

word2idx = {}
idx2word = {}
for k, v in dictionary.items():
    if v < VOCAB_SIZE:
        word2idx[k] = v
        idx2word[v] = k
    if v >= VOCAB_SIZE-1:
        continue


print("en_input:- ", que)
# print("de_input:- ", start_word)
print("en_input sequences:- ", encoder_input_data)
# print("de_input sequences:- ", de_input)
print("en_input sequences:- ", np.array(encoder_input_data).shape)
# print("de_input sequences:- ", np.array(de_input).shape)
print("total lenght of word2idx:- ", len(word2idx))

# load the model we saved
model = load_model('model.h5')
model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])
# prediction = model.predict([np.array(en_input), np.array(de_input)])
# print("prediction:- ", prediction)
# print("real output:- ", output_data)
def predicts(encoder_input_data):
    start_word = ['bos']
    while 1:
        now_caps = [word2idx[i] for i in start_word]
        print("en_input sequences:- ", np.array(encoder_input_data).shape)
        now_caps = pad_sequences([now_caps], maxlen=MAX_LEN, padding='post')
        # now_caps = now_caps.reshape((10, ))
        preds = model.predict([encoder_input_data, now_caps])
        word_pred = idx2word[np.argmax(preds[0])]
        start_word.append(word_pred)

        if word_pred == 'eos' or len(start_word) > MAX_LEN:
    #keep on predicting next word unitil word predicted is <end> or caption lenghts is greater than max_lenght(40)
            break

    return ' '.join(start_word[1:-1])

print('Greedy search:', predicts(encoder_input_data))
