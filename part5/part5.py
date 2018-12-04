import numpy as np
from tqdm import tqdm, trange

import word2vec
import rnn
from preprocess import clean_word


def predict(entity_mapping, entity_type_mapping, y1, y2):
    idx1 = np.argmax(y1, axis=1)
    idx2 = np.argmax(y2, axis=1)
    
    e_mapping = list(entity_mapping)
    e = [e_mapping[int(i)] for i in idx1]
    
    e_t_mapping = list(entity_type_mapping)
    e_t = [e_t_mapping[int(i)] for i in idx2]
    
    output = []
    for entity, entity_type in zip(e, e_t):
        if entity == 'O' or entity_type == 'O':
            output.append('O')
        else:
            output.append(f'{entity}-{entity_type}')
    return output


def write_predictions(W, U, model,
                      token_mapping, entity_mapping, entity_type_mapping,
                      val_filename, out_filename):
    # read val file
    with open(val_filename, 'r') as f:
        lines = f.readlines()

    sentence, sentences = [], []
    for word in lines:
        if word == '\n':
            sentences.append(sentence)
            sentence = []
        else:
            sentence.append(clean_word(word))
            
    # convert inputs via word2vec
    X = rnn.prepare_inputs(W, U, token_mapping, sentences)
    # get and store predictions
    result = []
    for i in trange(len(X), leave=False):
        y1, y2 = rnn.forward(*model, x=X[i])
        y_pred = predict(entity_mapping, entity_type_mapping, y1, y2)
        result.append(y_pred)
    # write predictions to file
    with open(out_filename, 'w') as f:
        for i in range(len(sentences)):
            for j in range(len(sentences[i])):
                word = sentences[i][j]
                pred = result[i][j]
                f.write(word + ' ' + pred + '\n')
            f.write('\n')


def train_and_predict():
    # W, U = word2vec.train_word2vec(train_filename)
    # word2vec.save_model(W, U, f'weights/word2vec/{dataset}')
    W, U = word2vec.load_model(f'weights/word2vec/{dataset}')
    
    mappings, model = rnn.train_rnn(f'../data/{dataset}/train', f'weights/word2vec/{dataset}')
    rnn.save_model(*model, f'weights/rnn/{dataset}')
    rnn.save_mappings(*mappings, f'mappings/{dataset}')
    
    write_predictions(W, U, model, *mappings, val_filename, out_filename)

    
def load_and_predict():
    W, U = word2vec.load_model(f'weights/word2vec/{dataset}')
    model = rnn.load_model(f'weights/rnn/{dataset}')
    mappings = rnn.load_mappings(f'mappings/{dataset}')
    write_predictions(W, U, model, *mappings, val_filename, out_filename)


if __name__ == '__main__':
    dataset = 'EN'
    train_filename = f'../data/{dataset}/train'
    val_filename = f'../data/{dataset}/dev.in'
    out_filename = f'../data/{dataset}/dev.p5.out'

    train_and_predict()
    print('Done')

