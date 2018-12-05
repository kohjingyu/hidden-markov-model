import argparse
import numpy as np
from tqdm import tqdm, trange

from word2vec import load_model as load_word2vec
from file_io import parse, read_file
from preprocess import get_token_mapping, tokenize
from evaluate import get_scores


def one_hot_encode(n, depth):
    a = np.zeros([depth])
    a[n] = 1
    return a


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def word2vec(w2v_W, w2v_U, sentence):
    """
    sentence: array of one-hot vectors [n_words, n_tokens]
    """
    weights = 0.5 * (w2v_W + w2v_U.T)  # [300, n_tokens]
    return sentence.dot(weights.T)  # [n_words, 300]


def prepare_inputs(token_mapping, w2v_W, w2v_U, sentences):
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
    
    one_hot_tokens = [np.asarray(ls) for ls in one_hot_tokens]
    vec_tokens = [word2vec(w2v_W, w2v_U, sentence) for sentence in tqdm(one_hot_tokens, desc='Vectorizing tokens')]
    return vec_tokens


def prepare_labels(state_mapping, sequences):
    """
    Convert each state from str to its corresponding int value.
    Convert the int to a one-hot vector.
    """
    encoded_labels = [[state_mapping[state] for state in label] for label in sequences]
    
    depth = len(state_mapping)
    one_hot_labels = [[one_hot_encode(label, depth) for label in sequence] for sequence in encoded_labels]
    one_hot_labels = [np.asarray(ls) for ls in one_hot_labels]
    return one_hot_labels


def init_vars(input_size, output_size, n_hidden, seed=0):
    np.random.seed(seed)
    # layer 1
    W = np.random.normal(0, 0.1, size=[input_size, n_hidden])
    b = np.ones(shape=[1, n_hidden]) * 0.1
    # layer 2
    V = np.random.normal(0, 0.1, size=[n_hidden, n_hidden])
    c = np.ones(shape=[1, n_hidden]) * 0.1
    # output layer
    U = np.random.normal(0, 0.1, size=[n_hidden, output_size])
    d = np.ones(shape=[1, output_size]) * 0.1
    # time for some unpacking magic
    model = (W, b, V, c, U, d)
    return model


def forward(W, b, V, c, U, d, x):
    """
    Compute the outputs at each time step.
    
    x - shape [n, 300]
    """
    a1 = x.dot(W) + b
    h1 = a1 * (a1 >= 0)
    
    a2 = h1.dot(V) + c
    h2 = a2 * (a2 >= 0)
    
    o = h2.dot(U) + d
    y_ = softmax(o)
    return y_


def backward(W, b, V, c, U, d, x, y):
    assert len(x) == len(y), [len(x), len(y)]

    # feedforward
    a1 = x.dot(W) + b
    h1 = a1 * (a1 >= 0)  # ReLU
    h1 *= np.random.rand(*h1.shape) >= dropout
    
    a2 = h1.dot(V) + c
    h2 = a2 * (a2 >= 0)  # ReLU
    h2 *= np.random.rand(*h2.shape) >= dropout
    
    o = h2.dot(U) + d
    y_ = softmax(o)

    # backprop
    do = y_ - y
    
    dd = np.mean(do, axis=0, keepdims=True)
    dU = h2.T.dot(do)
    
    dh2 = do.dot(U.T)
    da2 = dh2 * (a2 >= 0)
    
    dc = np.mean(da2, axis=0, keepdims=True)
    dV = h1.T.dot(da2)
    
    dh1 = da2.dot(V.T)
    da1 = dh1 * (a1 >= 0)
    
    db = np.mean(da1, axis=0, keepdims=True)
    dW = x.T.dot(da1)

    grad = (dW, db, dV, dc, dU, dd)
    # compute loss
    xent = -np.log(y_ + 1e-8) * y
    loss = np.mean(np.sum(xent, axis=1))
    return loss, grad


def predict(state_mapping, probs):
    states = list(state_mapping)
    indices = np.argmax(probs, axis=1)  # take argmax of the probs
    return [states[int(i)] for i in indices]  # map indices back to states


def train(model, X_train, y_train, X_val, state_mapping, val_sentences, n_epochs, batch_size):
    global lr  # changing lr for decay requires global write

    checkpoints = []  # store model weights after every epoch
    losses = []  # training loss
    scores = []  # f1-score
    cum_grad = [np.zeros_like(weight) for weight in model]  # cumulative gradient updates
    
    n_iters = len(X_train) // batch_size
    for i in trange(n_epochs, desc='Training MLP'):
        for j in trange(n_iters, desc='Epoch {}'.format(i+1), leave=False):
            idx = np.random.choice(np.arange(len(X_train)), size=batch_size)
            x, y = X_train[idx], y_train[idx]
            loss, grad = backward(*model, x=x, y=y)

            # gradient accumulation
            for cum_weight, weight_update in zip(cum_grad, grad):
                cum_weight += np.square(weight_update)
            # gradient update
            for weight, cum_weight, weight_update in zip(model, cum_grad, grad):
                weight -= (lr / (np.sqrt(cum_weight) + 1e-7)) * weight_update

            losses.append(loss)
            
        # validate
        results = []  # get and store predictions
        for i in trange(len(X_val), desc='Validation', leave=False):
            y_pred = predict(state_mapping, forward(*model, x=X_val[i]))
            results.append(y_pred)

        lr *= decay
        scores.append(get_scores(results, val_sentences, out_filename, val_filename))
        checkpoints.append(tuple(weights.copy() for weights in model))
    return checkpoints, losses, scores


def main():
    # read file
    observations, states = parse(train_filename)
    train_sentences, train_labels = read_file(train_filename)

    val_sentences, val_labels = read_file(val_filename)

    # preprocess
    token_mapping = get_token_mapping(observations)
    X_train = prepare_inputs(token_mapping, w2v_W, w2v_U, train_sentences)

    state_mapping = {state: i for i, state in enumerate(states)}
    y_train = prepare_labels(state_mapping, train_labels)

    X_val = prepare_inputs(token_mapping, w2v_W, w2v_U, val_sentences)
    y_val = prepare_labels(state_mapping, val_labels)

    # train model
    model = init_vars(input_size=300, output_size=len(state_mapping), n_hidden=n_hidden)

    X_flat, y_flat = [], []
    for sentence in X_train:
        X_flat.extend(sentence)
    X_flat = np.asarray(X_flat)
    for sentence in y_train:
        y_flat.extend(sentence)
    y_flat = np.asarray(y_flat)

    checkpoints, losses, scores = train(model, X_flat, y_flat, X_val, state_mapping, val_sentences, n_epochs, batch_size)
    
    f_entity = [tup[0][-1] for tup in scores]
    f_type = [tup[1][-1] for tup in scores]
    print('Entity:', (np.argmax(f_entity), np.max(f_entity)))
    print('Entity type:', (np.argmax(f_type), np.max(f_type)))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dank MLP trainer')

    parser.add_argument('dataset', help='{EN, FR}')
    parser.add_argument('--epochs', default=10, type=int, help='No. of epochs [10]')
    parser.add_argument('--batch', default=64, type=int, help='Batch size [64]')
    parser.add_argument('--lr', default=1e-2, type=float, help='Learning rate [1e-2]')
    parser.add_argument('--hidden', default=128, type=int, help='No. of hidden units [128]')
    parser.add_argument('--decay', default=1.0, type=float, help='Learning rate decay [1.0]')
    parser.add_argument('--dropout', default=0.0, type=float, help='Probability of dropping [0.0]')
    args = parser.parse_args()

    n_epochs = args.epochs
    batch_size = args.batch
    lr = args.lr
    n_hidden = args.hidden
    decay = args.decay
    dropout = args.dropout
    
    dataset = args.dataset
    
    train_filename = '../data/{}/train'.format(dataset)
    val_filename = '../data/{}/dev.out'.format(dataset)
    out_filename = '../data/{}/dev.p5.out'.format(dataset)
    
    word2vec_dir = 'weights/word2vec/{}'.format(dataset)
    w2v_W, w2v_U = load_word2vec(word2vec_dir)
    
    main()
