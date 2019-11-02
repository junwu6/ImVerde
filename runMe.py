from Model import ImVerde as model
import argparse
import numpy as np
import pickle as pkl
import sys
from sklearn.metrics import average_precision_score

DATASET = 'cora' # cora, citeseer or pubmed

parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', help='learning rate for supervised loss', type=float, default=0.1)
parser.add_argument('--embedding_size', help='embedding dimensions', type=int, default=50)
parser.add_argument('--window_size', help='window size in random walk sequences', type=int, default=3)
parser.add_argument('--path_size', help='length of random walk sequences', type=int, default=10)
parser.add_argument('--batch_size', help='batch size for supervised loss', type=int, default=200)
parser.add_argument('--g_batch_size', help='batch size for graph context loss', type=int, default=200)
parser.add_argument('--g_sample_size', help='batch size for label context loss', type=int, default=100)
parser.add_argument('--neg_samp', help='negative sampling rate; zero means using softmax', type=int, default=10)
parser.add_argument('--g_learning_rate', help='learning rate for unsupervised loss', type=float, default=1e-2)
parser.add_argument('--model_file', help='filename for saving models', type=str, default='trans.model')
parser.add_argument('--use_feature', help='whether use input features', type=bool, default=True)
parser.add_argument('--update_emb', help='whether update embedding when optimizing supervised loss', type=bool,
                    default=True)
parser.add_argument('--layer_loss', help='whether incur loss on hidden layers', type=bool, default=True)
args = parser.parse_args()


def comp_accu(tpy, ty):
    correct_number = np.count_nonzero(np.argmax(ty, axis = 1) == 1)
    predict_minority = [tpy[i][1] for i in range(len(tpy))]
    sorted_idx = np.argsort(predict_minority)
    predict_label = np.zeros(len(tpy))
    top_index = sorted_idx[-correct_number:]
    predict_label[top_index] = 1
    precision = ((np.argmax(ty, axis = 1) == 1) & (predict_label == np.argmax(ty, axis = 1))).sum() * 1.0 / correct_number
    return precision


print("load the data:", DATASET)
NAMES = ['x', 'y', 'tx', 'ty', 'graph']
objects = []
for i in range(len(NAMES)):
    with open("data/trans.{}.{}".format(DATASET, NAMES[i]), 'rb') as f:
        if sys.version_info > (3, 0):
            objects.append(pkl.load(f, encoding='latin1'))
        else:
            objects.append(pkl.load(f))
x, y, tx, ty, graph = tuple(objects)

for i in range(len(y)):
    if (y[i] == [0, 1, 0, 0, 0, 0, 0]).all():
        y[i] = [0, 1, 0, 0, 0, 0, 0]
    else:
        y[i] = [1, 0, 0, 0, 0, 0, 0]

for i in range(len(ty)):
    if (ty[i] == [0, 1, 0, 0, 0, 0, 0]).all():
        ty[i] = [0, 1, 0, 0, 0, 0, 0]
    else:
        ty[i] = [1, 0, 0, 0, 0, 0, 0]

weight = []
for i in range(len(graph)):
    iweight = []
    for j in graph[i]:
        iweight.append(1.0)
    weight.append(iweight)

print("Initial the model")
m = model(args)
m.add_data(x, y, graph, weight, 2, True)
m.build()
m.init_train(init_iter_graph=150)

print("Start training")
max_iters = 50000
max_accu, max_recall = 0, 0
y_score = None
for iter_cnt in range(max_iters):
    m.step_train(max_iter=1, iter_graph=0, iter_inst=1, iter_label=0)
    tpy = m.predict(tx)
    accu = comp_accu(tpy, ty)
    print(iter_cnt, accu, max_accu)
    if accu > max_accu:
        m.store_params()
        max_accu = max(max_accu, accu)
        y_score = tpy

APscore = average_precision_score(ty[:, 1], y_score[:, 1])
print("The average_precision_score is: {}".format(APscore))
