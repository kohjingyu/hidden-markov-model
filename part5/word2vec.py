import numpy as np
import os
from os.path import join
from tqdm import tqdm, trange

from file_io import parse, read_file
from preprocess import get_token_mapping, tokenize

lr = 1e-4


def softmax(x):
    """
    Computes softmax on a vector of shape (n, 1)
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def one_hot_encode(n, depth):
    """
    Return one-hot encoded vector of shape (depth, 1)
    """
    a = np.zeros([depth, 1])
    a[n, 0] = 1
    return a


def prepare_inputs(token_mapping, sentences):
    """
    Converts a 2-D list of sentences (list of list of words) to 
        one-hot encoded tokens of shape 
        [n_sentences, n_words, len(token_mapping), 1].
    """
    tokens = [tokenize(token_mapping, sentence) for sentence in sentences] 
    
    depth = len(token_mapping)
    one_hot_tokens = [[one_hot_encode(token, depth) for token in sentence] for sentence in tokens]
    one_hot_tokens = [np.asarray(ls) for ls in one_hot_tokens]  # list of [n_words, len(token_mapping), 1]
    return one_hot_tokens

    
    
def init_vars(vocab_size, latent_size=300, seed=0):
    np.random.seed(0)
    
    W = np.random.normal(0, 0.1, size=[latent_size, vocab_size])
    U = np.random.normal(0, 0.1, size=[vocab_size, latent_size])
    return W, U


def forward(W, U, x):
    h = W.dot(x)
    o = U.dot(h)
    y_ = softmax(o)
    return y_


def backward(W, U, x, y):
    h = W.dot(x)
    o = U.dot(h)
    y_ = softmax(o)
    
    do = y_ - y
    dU = do.dot(h.T)
    dW = U.T.dot(do).dot(x.T)
    grad = (dW, dU)
    
    assert dU.shape == U.shape
    assert dW.shape == W.shape
    
    xent = -np.log(y_ + 1e-8) * y
    loss = np.sum(xent)
    return loss, grad
    
    
def train_sentence(W, U, sentence):
    for i in range(len(sentence)):
        # python allows indexing beyond len(list), but not before
        if i <= 2:
            context = np.concatenate([sentence[:i], sentence[i+1:i+3]])
        else:
            context = np.concatenate([sentence[i-2:i], sentence[i+1:i+3]])
        # positive sampling
        center_word = sentence[i]
        loss, (dW, dU) = backward(W, U, center_word, np.sum(context, axis=0))
        W -= lr * dW
        U -= lr * dU
        # create probabilities for negative samples
        word_counts = np.asarray(list(observations.values()))
        word_probs = np.power(word_counts, 0.75) / np.sum(np.power(word_counts, 0.75))
        # remove probability of context word being negatively sampled
        for context_word in context:
            word_probs[np.where(context_word.flatten())] = 0
        # negative sampling
        word_probs /= np.sum(word_probs)  # normalize to 1
        # sample indices of 20 words
        negative_idx = np.random.choice(np.arange(len(word_probs)), size=20, p=word_probs)
        # create a size-20 one-hot window
        neg_context = np.zeros([20, len(token_mapping), 1])
        for i, idx in enumerate(negative_idx):
            neg_context[i, idx, 0] = 0
        loss, (dW, dU) = backward(W, U, center_word, -np.sum(neg_context, axis=0))
        W -= lr * dW
        U -= lr * dU


def train_epoch(i, W, U, X, observations, token_mapping):
    for sentence in tqdm(X, desc='Training sentence', leave=False):
        W, U = train_sentence(W, U, sentence, observations, token_mapping)
    return W, U


def save_model(W, U, root_dir):
    os.makedirs(root_dir, exist_ok=True)
    np.save(join(root_dir, 'W'), W)
    np.save(join(root_dir, 'U'), U)
    print('word2vec model saved in:', root_dir)
    
    
def load_model(root_dir):
    W = np.load(join(root_dir, 'W_new.npy'))
    U = np.load(join(root_dir, 'U_new.npy'))
    return W, U
    
    
def train_word2vec(filename, n_epochs=1):
    # read input file
    observations, states = parse(filename)
    sentences, labels = read_file(filename)
    # preprocess
    token_mapping = get_token_mapping(observations)
    X = prepare_inputs(token_mapping, sentences)
    # init model
    vocab_size = len(token_mapping)
    W, U = init_vars(vocab_size, latent_size=300)
    # train
    for i in trange(n_epochs, desc='Training word2vec'):
        W, U = train_epoch(i, W, U, X, observations, token_mapping)
    return W, U
    

if __name__ == '__main__':
    dataset = 'FR'
    filename = f'../data/{dataset}/train'
    W, U = train_word2vec(filename)

    save_model(W, U, f'weights/word2vec/{dataset}')
