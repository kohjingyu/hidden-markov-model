
# coding: utf-8

# In[4]:


import numpy as np
from tqdm import tqdm

import torch
from torch import nn, optim
import torch.nn.functional as F

import time

import sys

# In[5]:


def prepare_sequence(seq, to_ix):
    idxs = []
    
    for w in seq:
        if w in to_ix:
            idxs.append(to_ix[w])
        else:
            idxs.append(len(to_ix))
    
    return idxs #torch.tensor(idxs, dtype=torch.long)

def produce_tags(seq, to_tag):
    tags = []
    
    for x in seq:
#         if x in to_tag:
        tags.append(to_tag[x])
#         else:
#             tags.append(to_tag[np.random.randint(len(to_tag))])
    return tags


# In[11]:


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python LSTM_Chunker.py SG")

    dataset = sys.argv[1] # {SG, CN, EN, FR}
    assert(dataset in ["SG", "CN", "EN", "FR"])
    train_filename = f"data/{dataset}/train"
    validation_filename = f"data/{dataset}/dev.out"

    EMBEDDING_DIM = 16
    HIDDEN_DIM = 128
    epochs = 20
    batch_size = 1
    gpu = True


    # In[12]:


    # Prepare training data
    with open(train_filename, "r") as f:
        lines = f.readlines()

    train_data = []
    sentence = []
    tags = []

    max_length = 0

    word_to_ix = {}
    tag_to_ix = {}

    for line in lines:
        line_split = line.strip().split(" ")
        
        if len(line_split) == 1:
            # New sentence
            if len(sentence) > max_length:
                max_length = len(sentence)

            train_data.append([sentence, tags])
            sentence = []
            tags = []
        elif len(line_split) == 2:
            word, tag = line_split[0], line_split[1]
            
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)

            sentence.append(word)
            tags.append(tag)
            
    idx_to_tags = {v: k for k, v in tag_to_ix.items()}


    # In[13]:


    print(max_length)


    # In[14]:


    class LSTMTagger(nn.Module):
        def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
            super(LSTMTagger, self).__init__()
            self.hidden_dim = hidden_dim

            # TODO: Transfer embedding weights from word2vec or something
            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)

            # The linear layer that maps from hidden state space to tag space
            self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
            self.hidden = self.init_hidden(batch_size)

        def init_hidden(self, size):
            # The axes semantics are (num_layers, minibatch_size, hidden_dim)
            zeros = torch.zeros(1, size, self.hidden_dim)
            
            if gpu:
                zeros = zeros.cuda()
            
            return (zeros, zeros)

        def forward(self, sentence):
            embeds = self.word_embeddings(sentence)
            lstm_out, self.hidden = self.lstm(
                embeds.view(len(sentence[0]), len(sentence), -1), self.hidden)
            tag_space = self.hidden2tag(lstm_out)
            tag_scores = F.log_softmax(tag_space, dim=2)
            return tag_scores


    # In[94]:


    vocab_size = len(word_to_ix)+1
    tagset_size = len(tag_to_ix)+1
    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, vocab_size, tagset_size) # +1 for unknown words
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    if gpu:
        model.cuda()

    for epoch in range(epochs):
        total_loss = 0
        num_batches = len(train_data) // batch_size
        batch_data = []
        label_data = []
        start = time.time()
        
        for sentence, tags in train_data:
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(tags, tag_to_ix)
            
            batch_data.append(sentence_in)
            label_data.append(targets)
            
            if len(batch_data) == batch_size:
                # Process batch
                max_length = max([len(x) for x in batch_data])
                
                model.zero_grad()
                model.hidden = model.init_hidden(len(batch_data))

                # Pad batch and labels
                for i in range(len(batch_data)):
                    data_tag = len(word_to_ix)
                    label_tag = len(tag_to_ix)
                    pad_amount = max_length - len(batch_data[i])
                    
                    if pad_amount > 0:
                        batch_data[i] += [data_tag] * pad_amount
                        label_data[i] += [label_tag] * pad_amount
                
                batch_data_tensor = torch.tensor(np.array(batch_data))
                batch_targets_tensor = torch.tensor(np.array(label_data), dtype=torch.long)
                
                if gpu:
                    batch_data_tensor = batch_data_tensor.cuda()
                    batch_targets_tensor = batch_targets_tensor.cuda()
                
                preds = model(batch_data_tensor)
                preds = preds.view(-1, tagset_size)
                batch_targets_tensor = batch_targets_tensor.view(-1)
                
                loss = loss_function(preds, batch_targets_tensor)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            
                batch_data = []
                label_data = []

        print("Epoch {0}, average loss: {1}, time taken: {2}s".format(epoch, total_loss / num_batches, time.time() - start), flush=True)


    # In[105]:


    # Prepare validation data
    with open(validation_filename, "r") as f:
        lines = f.readlines()

    val_data = []
    sentence = []
    tags = []

    for line in lines:
        line_split = line.strip().split(" ")
        
        if len(line_split) == 1:
            # New sentence
            val_data.append([sentence, tags])
            sentence = []
            tags = []
        else:
            word, tag = line_split[0], line_split[1]
            
            sentence.append(word)
            tags.append(tag)


    # In[106]:


    val_preds = []

    model.hidden = model.init_hidden(1)

    # See what the scores are after training
    with torch.no_grad():
        for i in tqdm(range(len(val_data))):
            sentence = val_data[i][0]
            inputs = prepare_sequence(sentence, word_to_ix)
            batch_data_tensor = torch.tensor(np.array(inputs)[np.newaxis,:])
            
            if gpu:
                batch_data_tensor = batch_data_tensor.cuda()
            
            preds = model(batch_data_tensor)
            tags = np.squeeze(preds.argmax(dim=2).cpu().data.numpy())
            predicted_tags = produce_tags(tags, idx_to_tags)
            
            val_preds.append(predicted_tags)


    # In[100]:


    with open(validation_filename.replace(".out", "_lstm.pred"), "w") as f:
        for i in range(len(val_data)):
            sentence = val_data[i][0]
            pred = val_preds[i]
            assert(len(sentence) == len(pred))
            
            for j in range(len(sentence)):
                f.write(sentence[j] + " " + pred[j] + "\n")
            
            f.write("\n")

