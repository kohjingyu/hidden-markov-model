"""
EN training for RNN:
    - word2vec
    - MTL
    - regularization
    - no count weighting
"""
import numpy as np
import pickle
import os
from os.path import join
from collections import defaultdict
from random import shuffle
from tqdm import tqdm, trange

from file_io import parse, read_file
from preprocess import get_token_mapping, tokenize
from utils import softmax, one_hot_encode
from word2vec import load_model as load_word2vec, word2vec

lr = 1e-4
batch_size = 32
n_epochs = 44
ld = 1e-3


def convert_state(state):
    """
    Converts a state into a (entity, entity_type) tuple.
    """
    if state == 'O':
        return (state, state)
    return tuple(state.split('-'))


def count_states(states):
    """
    Given a dictionary with the counts of each state,
    obtain counts for entities and entity types separately.
    """
    entities = defaultdict(int)
    entity_types = defaultdict(int)
    
    for state, counts in states.items():
        e, e_type = convert_state(state)
        entities[e] += counts
        entity_types[e_type] += counts
    return entities, entity_types


def prepare_inputs(w2v_W, w2v_U, token_mapping, sentences):
    """
    Converts a 2-D list of sentences (list of list of words)
    to one-hot encoded tokens of shape [n_sentences, n_words, len(token_mapping), 1].
    """
    tokens = [tokenize(token_mapping, sentence) for sentence in sentences] 
    
    depth = len(token_mapping)
    one_hot_tokens = []
    for sentence in tokens:
        one_hot_sentence = []
        for i, token in enumerate(sentence):
            if token != token_mapping['#UNK#']:
                one_hot_sentence.append(one_hot_encode(token, depth))
            else:
                if i <= 2:
                    context_tokens = sentence[:i] + sentence[i+1:i+3]
                else:
                    context_tokens = sentence[i-2:i] + sentence[i+1:i+3]
                context_one_hot = [one_hot_encode(token, depth) for token in context_tokens]
                context_mean = np.mean(np.asarray(context_one_hot), axis=0)
                one_hot_sentence.append(context_mean)
        one_hot_tokens.append(one_hot_sentence)
    
    # squeeze to convert from [len(sentence), n_tokens, 1] to [len(sentence), n_tokens]
    one_hot_tokens = [np.asarray(ls) for ls in one_hot_tokens]
    vec_tokens = [word2vec(w2v_W, w2v_U, sentence) for sentence in tqdm(one_hot_tokens, desc='Vectorizing sentences')]
    return vec_tokens


def prepare_labels(entity_mapping, entity_type_mapping, sequences):
    """
    Convert each state from str to its corresponding int value.
    Convert the int to a one-hot vector.
    """
    # convert each str label to its corresponding int value
    encoded_entities = [[entity_mapping[convert_state(state)[0]] for state in label] for label in sequences]
    encoded_e_types = [[entity_type_mapping[convert_state(state)[1]] for state in label] for label in sequences]
    # perform one hot encoding on int values
    depth = len(entity_mapping)
    one_hot_entities = [[one_hot_encode(label, depth) for label in sequence] for sequence in encoded_entities]
    one_hot_entities = [np.asarray(ls) for ls in one_hot_entities]
    
    depth = len(entity_type_mapping)
    one_hot_e_types = [[one_hot_encode(label, depth) for label in sequence] for sequence in encoded_e_types]
    one_hot_e_types = [np.asarray(ls) for ls in one_hot_e_types]
    return one_hot_entities, one_hot_e_types


def init_vars(n_entities, n_entity_types, latent_size=300, rnn_size=128, seed=0):
    np.random.seed(0)
    # input
    U = np.random.normal(0, 0.1, size=[rnn_size, latent_size])
    # hidden layer
    W = np.random.normal(0, 0.1, size=[rnn_size, rnn_size])
    b = np.ones(shape=[rnn_size, 1]) * 0.1
    # output - entity
    V1 = np.random.normal(0, 0.1, size=[n_entities, rnn_size])
    c1 = np.ones(shape=[n_entities, 1]) * 0.1
    # output - entity type
    V2 = np.random.normal(0, 0.1, size=[n_entity_types, rnn_size])
    c2 = np.ones(shape=[n_entity_types, 1]) * 0.1
    # time for some unpacking magic
    model = (U, W, b, V1, c1, V2, c2)
    return model


def forward(U, W, b, V1, c1, V2, c2, x):
    """
    Compute the outputs at each time step.
    Note that in the forward function, we only store y_ at each time step.
    """
    y1_, y2_ = [], []
    # initialize h_1
    h = np.tanh(b + U.dot(x[0]))
    o1 = c1 + V1.dot(h)
    o2 = c2 + V2.dot(h)
    y1_.append(softmax(o1))
    y2_.append(softmax(o2))
    # iterate for [h_t, .., h_n]
    for i in range(1, len(x)):
        h = np.tanh(b + W.dot(h) + U.dot(x[i]))
        o1 = c1 + V1.dot(h)
        o2 = c2 + V2.dot(h)
        y1_.append(softmax(o1))
        y2_.append(softmax(o2))
    return np.array(y1_), np.array(y2_)


def jacobian(h):
    """
    Returns the Jacobian of tanh(h).
    """
    diag_elems = (1 - h**2).flatten()
    return np.diag(diag_elems)
    

def backward(U, W, b, V1, c1, V2, c2, x, y1, y2):
    assert len(x) == len(y1) == len(y2)
    n = len(x)
    if n <= 1:
        return np.nan, (0, 0, 0, 0, 0, 0, 0)
    
    # feedforward
    h_1 = np.tanh(b + U.dot(x[0]))
    h = [h_1]
    for i in range(1, n):
        h_t = np.tanh(b + W.dot(h[-1]) + U.dot(x[i]))
        h.append(h_t)

    o1 = [c1 + V1.dot(h_t) for h_t in h]
    o2 = [c2 + V2.dot(h_t) for h_t in h]
    
    # backprop
    do1 = [softmax(o1[i]) - y1[i] for i in range(n)]
    do2 = [softmax(o2[i]) - y2[i] for i in range(n)]
    
    dh_n = 0.5 * (V1.T.dot(do1[-1]) + V2.T.dot(do2[-1]))  # h_n has no (t+1) gradient
    dh = [dh_n]
    for i in range(n-2, -1, -1):
        dh_t = W.T.dot(jacobian(h[i+1])).dot(dh[0]) + 0.5 * (V1.T.dot(do1[i]) + V2.T.dot(do2[i]))
        dh.insert(0, dh_t)
        
    dc1 = np.sum(do1, axis=0)
    dV1 = np.sum([do1[i].dot(dh[i].T) for i in range(n)], axis=0)  # TODO: dubious
    
    dc2 = np.sum(do2, axis=0)
    dV2 = np.sum([do2[i].dot(dh[i].T) for i in range(n)], axis=0)  # TODO: dubious
    
    delta_h = [jacobian(h[i]).dot(dh[i]) for i in range(n)]  # propagated error term of h
    db = np.sum(delta_h, axis=0)
    dW = np.sum([delta_h[i].dot(h[i-1].T) for i in range(1, n)], axis=0)  # t=1 has no prev
    dU = np.sum([delta_h[i].dot(x[i].T) for i in range(n)], axis=0)
    
    dV1 += ld * np.abs(V1)
    dV2 += ld * np.abs(V2)
    dW += ld * np.abs(W)
    dU += ld * np.abs(U)
    
    assert dc1.shape == c1.shape
    assert dV1.shape == V1.shape
    assert dc2.shape == c2.shape
    assert dV2.shape == V2.shape
    assert db.shape == b.shape
    assert dW.shape == W.shape
    assert dU.shape == U.shape

    grad = (dU, dW, db, dV1, dc1, dV2, dc2)
    
    # compute loss
    y1_ = [softmax(o1_t) for o1_t in o1]
    xent1 = [-np.log(y1_[i] + 1e-8) * y1[i] for i in range(n)]
    loss1 = np.mean(np.sum(xent1, axis=1))  # sum softmax CE for each word, then take mean across all words
    
    y2_ = [softmax(o2_t) for o2_t in o2]
    xent2 = [-np.log(y2_[i] + 1e-8) * y2[i] for i in range(n)]
    loss2 = np.mean(np.sum(xent2, axis=1))  # sum softmax CE for each word, then take mean across all words
    return loss1 + loss2, grad
        
        
def train_epoch(model, X, y1, y2):
    n_iters = len(X) // batch_size
    for j in trange(n_iters, desc='Training sentence', leave=False):
        batch_indices = np.random.choice(np.arange(len(X)), size=batch_size)
        batch_grads = tuple(np.zeros_like(weight) for weight in model)

        for k in batch_indices:
            _x, _y1, _y2 = X[k], y1[k], y2[k]
            loss, grad = backward(*model, x=_x, y1=_y1, y2=_y2)

            for error, batch_error in zip(grad, batch_grads):
                batch_error += error  # accumulate all grads in batch
        # average gradients gradients
        for errors, weight in zip(batch_grads, model):
            weight -= lr * errors / batch_size
    return model


def save_model(U, W, b, V1, c1, V2, c2, root_dir):
    os.makedirs(root_dir, exist_ok=True)
    np.save(join(root_dir, 'U'), U)
    np.save(join(root_dir, 'W'), W)
    np.save(join(root_dir, 'b'), b)
    np.save(join(root_dir, 'V1'), V1)
    np.save(join(root_dir, 'c1'), c1)
    np.save(join(root_dir, 'V2'), V2)
    np.save(join(root_dir, 'c2'), c2)
    print('RNN model saved in:', root_dir)
    
    
def load_model(root_dir):
    U = np.load(join(root_dir, 'U.npy'))
    W = np.load(join(root_dir, 'W.npy'))
    b = np.load(join(root_dir, 'b.npy'))
    V1 = np.load(join(root_dir, 'V1.npy'))
    c1 = np.load(join(root_dir, 'c1.npy'))
    V2 = np.load(join(root_dir, 'V2.npy'))
    c2 = np.load(join(root_dir, 'c2.npy'))
    return (U, W, b, V1, c1, V2, c2)
    
    
def save_mappings(token_mapping, entity_mapping, entity_type_mapping, root_dir):
    os.makedirs(root_dir, exist_ok=True)
    with open(join(root_dir, 'token_mapping'), 'wb') as f_t, \
         open(join(root_dir, 'entity_mapping'), 'wb') as f_e, \
         open(join(root_dir, 'entity_type_mapping'), 'wb') as f_e_t:
        pickle.dump(token_mapping, f_t)
        pickle.dump(entity_mapping, f_e)
        pickle.dump(entity_type_mapping, f_e_t)
    print('Label mappings saved in :', root_dir)
    
    
def load_mappings(root_dir):
    with open(join(root_dir, 'token_mapping'), 'rb') as f:
        token_mapping = pickle.load(f)
    with open(join(root_dir, 'entity_mapping'), 'rb') as f:
        entity_mapping = pickle.load(f)
    with open(join(root_dir, 'entity_type_mapping'), 'rb') as f:
        entity_type_mapping = pickle.load(f)
    return token_mapping, entity_mapping, entity_type_mapping


def train_rnn(filename, word2vec_dir):
    # read input file
    observations, states = parse(filename)
    entities, entity_types = count_states(states)
    sentences, labels = read_file(filename)
    # load word2vec
    w2v_W, w2v_U = load_word2vec(word2vec_dir)
    # prepare inputs
    token_mapping = get_token_mapping(observations)
    X = prepare_inputs(w2v_W, w2v_U, token_mapping, sentences)
    # prepare labels
    entity_mapping = {entity: i for i, entity in enumerate(entities)}
    entity_type_mapping = {e_type: i for i, e_type in enumerate(entity_types)}
    y1, y2 = prepare_labels(entity_mapping, entity_type_mapping, labels)
    # init model
    model = init_vars(n_entities=len(entity_mapping),
                      n_entity_types=len(entity_type_mapping))
    # train
    for i in trange(n_epochs, desc='Training RNN'):
        model = train_epoch(model, X, y1, y2)
    mappings = (token_mapping, entity_mapping, entity_type_mapping)
    return mappings, model
    

if __name__ == '__main__':
    dataset = 'EN'
    mappings, model = train_rnn(f'../data/{dataset}/train', f'weights/word2vec/{dataset}')
    save_model(*model, f'weights/rnn/{dataset}')
    save_mappings(*mappings, f'mappings/{dataset}')
