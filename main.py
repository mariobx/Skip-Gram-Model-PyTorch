from __future__ import print_function
from tqdm import tqdm
# from tqdm import tqdm_gui
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys, pdb, os, shutil, pickle
from pprint import pprint 

import torch
import torch.optim as optim
import torch.nn as nn

# it is a little tricky on run SummaryWriter by installing a suitable version of pytorch. so if you are able to import SummaryWriter from torch.utils.tensorboard, this script will record summaries. Otherwise it would not.
try:
    from torch.utils.tensorboard import SummaryWriter
    write_summary = True
except:
    write_summary = False

from skipgram import Word2Vec_neg_sampling, check_cuda
from utils_modified import count_parameters
from datasets import word2vec_dataset
from utils_modified import q, nearest_word, count_parameters

# for tensorboard to work properly on embeddings projections
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

def print_nearest_words(model, test_words, word_to_ix, ix_to_word, top = 5):
    
    model.eval()
    emb_matrix = model.embeddings_input.weight.data.cpu()
    
    nearest_words_dict = {}

    print('==============================================')
    for t_w in test_words:
        
        inp_emb = emb_matrix[word_to_ix[t_w], :]  

        emb_ranking_top, _ = nearest_word(inp_emb, emb_matrix, top = top+1)
        print(t_w.ljust(10), ' | ', ', '.join([ix_to_word[i] for i in emb_ranking_top[1:]]))

    return nearest_words_dict

EMBEDDING_DIM = 128
DEVICE = torch.device("cuda" if check_cuda() else "cpu")
BATCH_SIZE = 64
LEARNING_RATE = 0.001
CONTEXT_SIZE = 2
FRACTION_DATA = 1
SUBSAMPLING = 1e-5
SAMPLING_RATE = 1e-3
NUM_EPOCHS = 5
NEGATIVE_SAMPLES = 10
DISPLAY_EVERY_N_BATCH = 1000
DISPLAY_BATCH_LOSS = True
SAVE_EVERY_N_EPOCH = 2
LR = 0.001
MODEL_ID = 'fordham-data'
PREPROCESSED_DATA_DIR  = os.path.join(MODEL_ID, 'preprocessed_data')
PREPROCESSED_DATA_PATH = os.path.join(PREPROCESSED_DATA_DIR, 'preprocessed_data_' + MODEL_ID + '_' + str(FRACTION_DATA) + '.pickle')
SUMMARY_DIR = os.path.join(MODEL_ID, 'summary') 
MODEL_DIR = os.path.join(MODEL_ID, 'models')
TEST_WORDS = ['COMPUTERSCIENCEI', 'COMPUTERSCIENCEII', 'CALCULUSI', 'CALCULUSII', 'PHYSICSI', 'INTRODUCTIONTOITALIANI']
# remove MODEL_DIR if it exists
if os.path.exists(MODEL_DIR):
    shutil.rmtree(MODEL_DIR)
# create MODEL_DIR    
os.makedirs(MODEL_DIR)

# SUMMARY_DIR is the path of the directory where the tensorboard SummaryWriter files are written
if write_summary:
    if os.path.exists(SUMMARY_DIR):
        # the directory is removed, if it already exists
        shutil.rmtree(SUMMARY_DIR)

    writer = SummaryWriter(SUMMARY_DIR) # this command automatically creates the directory at SUMMARY_DIR
    summary_counter = 0

# make training data
if not os.path.exists(PREPROCESSED_DATA_PATH):
    train_dataset = word2vec_dataset('fordham-data', CONTEXT_SIZE, FRACTION_DATA, SUBSAMPLING, SAMPLING_RATE)

    if not os.path.exists(PREPROCESSED_DATA_DIR):
        os.makedirs(PREPROCESSED_DATA_DIR)

    # pickle dump
    print('\ndumping pickle...')
    outfile = open(PREPROCESSED_DATA_PATH,'wb')
    pickle.dump(train_dataset, outfile)
    outfile.close()
    print('pickle dumped\n')

else:
    # pickle load
    print('\nloading pickle...')
    infile = open(PREPROCESSED_DATA_PATH,'rb')
    train_dataset = pickle.load(infile)
    infile.close()
    print('pickle loaded\n')

vocab = train_dataset.vocab
word_to_ix = train_dataset.word_to_ix
ix_to_word = train_dataset.ix_to_word

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
print('len(train_dataset): ', len(train_dataset))
print('len(train_loader): ', len(train_loader))
print('len(vocab): ', len(vocab), '\n')

# make noise distribution to sample negative examples from
word_freqs = np.array(list(vocab.values()))
unigram_dist = word_freqs/sum(word_freqs)
noise_dist = torch.from_numpy(unigram_dist**(0.75)/np.sum(unigram_dist**(0.75)))

losses = []

model = Word2Vec_neg_sampling(EMBEDDING_DIM, len(vocab), DEVICE, noise_dist, NEGATIVE_SAMPLES).to(DEVICE)
print('\nWe have {} Million trainable parameters here in the model'.format(count_parameters(model)))

# optimizer = optim.SGD(model.parameters(), lr = 0.008, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr = LR)
# print(model, '\n')

for epoch in tqdm(range(NUM_EPOCHS)):
    print('\n===== EPOCH {}/{} ====='.format(epoch + 1, NUM_EPOCHS))    
    # print('\nTRAINING...')

    # model.train()
    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
        print('batch# ' + str(batch_idx+1).zfill(len(str(len(train_loader)))) + '/' + str(len(train_loader)), end = '\r')
        
        model.train()

        x_batch           = x_batch.to(DEVICE)
        y_batch           = y_batch.to(DEVICE)
        
        optimizer.zero_grad()
        loss = model(x_batch, y_batch)
        
        loss.backward()
        optimizer.step()    
        
        losses.append(loss.item())
        if write_summary:
            # write tensorboard summaries
            writer.add_scalar(f'batch_loss', loss.item(), summary_counter)
            summary_counter += 1

        if batch_idx%DISPLAY_EVERY_N_BATCH == 0 and DISPLAY_BATCH_LOSS:
            print(f'Batch: {batch_idx+1}/{len(train_loader)}, Loss: {loss.item()}')    
            # show 5 closest words to some test words
            print_nearest_words(model, TEST_WORDS, word_to_ix, ix_to_word, top = 5)        

    # write embeddings every SAVE_EVERY_N_EPOCH epoch
    if epoch%SAVE_EVERY_N_EPOCH == 0: 
        metadata = [ix_to_word[k] for k in range(len(ix_to_word))]
        metadata_path = os.path.join(MODEL_DIR, "metadata.tsv")
        with open(metadata_path, "w") as f:
            for meta in metadata:
                f.write(f"{meta}\n")
        writer.add_embedding(model.embeddings_input.weight.data,
        metadata=metadata_path, 
        global_step=epoch)

        torch.save({'model_state_dict': model.state_dict(), 
                    'losses': losses,
                    'word_to_ix': word_to_ix,
                    'ix_to_word': ix_to_word
                    },                  
                    '{}/model{}.pth'.format(MODEL_DIR, epoch))

plt.figure(figsize = (50, 50))
plt.xlabel("batches")
plt.ylabel("batch_loss")
plt.title("loss vs #batch")

plt.plot(losses)
plt.savefig('losses.png')
plt.show()

# '''
EMBEDDINGS = model.embeddings_input.weight.data
print('EMBEDDINGS.shape: ', EMBEDDINGS.shape)

from sklearn.manifold import TSNE

print('\n', 'running TSNE...')
tsne = TSNE(n_components = 2).fit_transform(EMBEDDINGS.cpu())
print('tsne.shape: ', tsne.shape) #(15, 2)

############ VISUALIZING ############
x, y = [], []
annotations = []
for idx, coord in enumerate(tsne):
    # print(coord)
    annotations.append(ix_to_word[idx])
    x.append(coord[0])
    y.append(coord[1])   

# test_words = ['king', 'queen', 'berlin', 'capital', 'germany', 'palace', 'stays']
# test_words = ['sun', 'moon', 'earth', 'while', 'open', 'run', 'distance', 'energy', 'coal', 'exploit']
# test_words = ['amazing', 'beautiful', 'work', 'breakfast', 'husband', 'hotel', 'quick', 'cockroach']

test_words = TEST_WORDS_VIZ
print('test_words: ', test_words)

plt.figure(figsize = (50, 50))
for i in range(len(test_words)):
    word = test_words[i]
    #print('word: ', word)
    vocab_idx = word_to_ix[word]
    # print('vocab_idx: ', vocab_idx)
    plt.scatter(x[vocab_idx], y[vocab_idx])
    plt.annotate(word, xy = (x[vocab_idx], y[vocab_idx]), \
        ha='right',va='bottom')

plt.savefig("w2v.png")
plt.show()
# '''