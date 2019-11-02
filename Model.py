import lasagne
from theano import sparse
import theano.tensor as T
import theano
import layers
import numpy as np
import random
from collections import defaultdict as dd
import copy

from base_model import base_model


class ImVerde(base_model):
    """Planetoid-T.
    """

    def add_data(self, x, y, graph, weight, r, is_vdrw):
        """add data to the model.
        x (scipy.sparse.csr_matrix): feature vectors for training data.
        y (numpy.ndarray): one-hot label encoding for training data.
        graph (dict): the format is {index: list_of_neighbor_index}. Only supports binary graph.
        Let L and U be the number of training and dev instances.
        The training instances must be indexed from 0 to L - 1 with the same order in x and y.
        By default, our implementation assumes that the dev instances are indexed from L to L + U - 1, unless otherwise
        specified in self.predict.
        """
        self.x, self.y, self.graph, self.weight, self.r, self.is_vdrw = x, y, graph, weight, r, is_vdrw

    def build(self):
        """build the model. This method should be called after self.add_data.
        """
        x_sym = sparse.csr_matrix('x', dtype='float32')
        y_sym = T.imatrix('y')
        g_sym = T.imatrix('g')
        gy_sym = T.vector('gy')
        ind_sym = T.ivector('ind')

        l_x_in = lasagne.layers.InputLayer(shape=(None, self.x.shape[1]), input_var=x_sym)
        l_g_in = lasagne.layers.InputLayer(shape=(None, 2), input_var=g_sym)
        l_ind_in = lasagne.layers.InputLayer(shape=(None,), input_var=ind_sym)
        l_gy_in = lasagne.layers.InputLayer(shape=(None,), input_var=gy_sym)

        num_ver = len(self.graph)
        l_emb_in = lasagne.layers.SliceLayer(l_g_in, indices=0, axis=1)  # the first column in l_g_in
        l_emb_in = lasagne.layers.EmbeddingLayer(l_emb_in, input_size=num_ver,
                                                 output_size=self.embedding_size)  # word embedding
        l_emb_out = lasagne.layers.SliceLayer(l_g_in, indices=1, axis=1)  # the second column in l_g_in
        if self.neg_samp > 0:
            l_emb_out = lasagne.layers.EmbeddingLayer(l_emb_out, input_size=num_ver, output_size=self.embedding_size)

        l_emd_f = lasagne.layers.EmbeddingLayer(l_ind_in, input_size=num_ver, output_size=self.embedding_size,
                                                W=l_emb_in.W)
        l_x_hid = layers.SparseLayer(l_x_in, self.y.shape[1], nonlinearity=lasagne.nonlinearities.softmax)

        if self.use_feature:
            l_emd_f = layers.DenseLayer(l_emd_f, self.y.shape[1], nonlinearity=lasagne.nonlinearities.softmax)
            l_y = lasagne.layers.ConcatLayer([l_x_hid, l_emd_f], axis=1)
            l_y = layers.DenseLayer(l_y, self.y.shape[1], nonlinearity=lasagne.nonlinearities.softmax)
        else:
            l_y = layers.DenseLayer(l_emd_f, self.y.shape[1], nonlinearity=lasagne.nonlinearities.softmax)

        py_sym = lasagne.layers.get_output(l_y)
        loss = lasagne.objectives.categorical_crossentropy(py_sym, y_sym).mean()
        if self.layer_loss and self.use_feature:
            hid_sym = lasagne.layers.get_output(l_x_hid)
            loss += lasagne.objectives.categorical_crossentropy(hid_sym, y_sym).mean()
            emd_sym = lasagne.layers.get_output(l_emd_f)
            loss += lasagne.objectives.categorical_crossentropy(emd_sym, y_sym).mean()

        if self.neg_samp == 0:
            l_gy = layers.DenseLayer(l_emb_in, num_ver, nonlinearity=lasagne.nonlinearities.softmax)
            pgy_sym = lasagne.layers.get_output(l_gy)
            g_loss = lasagne.objectives.categorical_crossentropy(pgy_sym, lasagne.layers.get_output(l_emb_out)).sum()
        else:
            l_gy = lasagne.layers.ElemwiseMergeLayer([l_emb_in, l_emb_out], T.mul)
            pgy_sym = lasagne.layers.get_output(l_gy)
            g_loss = - T.log(T.nnet.sigmoid(T.sum(pgy_sym, axis=1) * gy_sym)).sum()

        params = [l_emd_f.W, l_emd_f.b, l_x_hid.W, l_x_hid.b, l_y.W, l_y.b] if self.use_feature else [l_y.W, l_y.b]
        if self.update_emb:
            params = lasagne.layers.get_all_params(l_y)
        updates = lasagne.updates.sgd(loss, params, learning_rate=self.learning_rate)

        self.train_fn = theano.function([x_sym, y_sym, ind_sym], loss, updates=updates, on_unused_input='ignore')
        self.test_fn = theano.function([x_sym, ind_sym], py_sym, on_unused_input='ignore')
        self.l = [l_gy, l_y]

        g_params = lasagne.layers.get_all_params(l_gy, trainable=True)
        g_updates = lasagne.updates.sgd(g_loss, g_params, learning_rate=self.g_learning_rate)
        self.g_fn = theano.function([g_sym, gy_sym], g_loss, updates=g_updates, on_unused_input='ignore')

    def gen_train_inst(self):
        """generator for batches for classification loss.
        """
        while True:
            ind = np.array(np.random.permutation(self.x.shape[0]), dtype=np.int32)
            i = 0
            while i < ind.shape[0]:
                j = min(ind.shape[0], i + self.batch_size)
                yield self.x[ind[i: j]], self.y[ind[i: j]], ind[i: j]
                i = j

    def gen_graph(self):
        """generator for batches for graph context loss.
        """
        num_ver = len(self.graph)
        ### label nodes (binary classification)
        labels, label2inst, not_label = [], dd(list), dd(list)
        for i in range(self.x.shape[0]):
            flag = False
            for j in range(self.y.shape[1]):
                if self.y[i, j] == 1 and not flag:
                    labels.append(j)  # labels: i-th sample belongs to j-th class
                    label2inst[j].append(i)  # label2inst: j-th class contains i-th sample
                    flag = True
                elif self.y[i, j] == 0:
                    not_label[j].append(i)

        numOFlabeled = self.x.shape[0]
        numOFminority = len(label2inst[1])

        while True:
            ind = np.random.permutation(num_ver)  # change the array order
            i = 0
            print(i)
            while i < ind.shape[0]:
                g, gy = [], []
                j = min(ind.shape[0], i + self.g_batch_size)
                iprobability = copy.deepcopy(self.weight)
                # mini-batch sampling (0 for majority, 1 for minority)
                indx_min = label2inst[1]
                indx_maj = random.sample(label2inst[0], numOFminority)
                indx_unlab = np.random.permutation(num_ver - numOFlabeled) + numOFlabeled
                ind_samp = indx_min + indx_maj + list(indx_unlab[:(self.g_batch_size - numOFminority * 2)])

                for k in ind_samp:
                    if len(self.graph[k]) == 0:
                        continue
                    path = [k]  # initial vertex in random walk

                    # vertex-diminished random walk
                    for _ in range(self.path_size):
                        probability = iprobability[path[-1]]
                        if sum(probability) != 0:
                            norm = [float(p) / (sum(probability)) for p in probability]

                        # If labeled, find nodes with the same label.
                        flag0 = -1  # (-1 for unlabeled, 0 for majority, 1 for minority)
                        label_graph = []
                        for inode in range(len(label2inst)):
                            if path[-1] in label2inst[inode]:
                                flag0 = inode
                                label_graph = label2inst[inode]
                                break

                        if flag0 < 0 or random.randint(1, 10) > self.r:
                            vertexAdd = random.choice(np.random.choice(self.graph[path[-1]], 1000, p=norm))
                            path.append(vertexAdd)
                            for gv in self.graph[vertexAdd]:
                                if vertexAdd in self.graph[gv]:
                                    index = self.graph[gv].index(vertexAdd)
                                    if self.is_vdrw:
                                        iprobability[gv][index] = iprobability[gv][index] * 0.7
                                    else:
                                        iprobability[gv][index] = iprobability[gv][index] + self.weight[gv][index]
                        else:
                            vertexAdd = random.choice(label_graph)
                            path.append(vertexAdd)

                    for l in range(len(path)):
                        for m in range(l - self.window_size, l + self.window_size + 1):
                            if m < 0 or m >= len(path): continue
                            g.append([path[l], path[m]])  # add L-st and M-st nodes as context
                            gy.append(1.0)  # positive sample
                            for _ in range(self.neg_samp):
                                g.append([path[l], random.randint(0, num_ver - 1)])  # randomly select one node
                                gy.append(- 1.0)  # negative sample
                yield np.array(g, dtype=np.int32), np.array(gy, dtype=np.float32)
                i = j

    def init_train(self, init_iter_graph):
        """pre-training of graph embeddings.
        init_iter_label (int): # iterations for optimizing label context loss.
        init_iter_graph (int): # iterations for optimizing graph context loss.
        """
        for i in range(init_iter_graph):
            gx, gy = next(self.graph_generator)
            loss = self.g_fn(gx, gy)
            print('iter graph', i, loss)

    def step_train(self, max_iter, iter_graph, iter_inst, iter_label):
        """a training step. Iteratively sample batches for three loss functions.
        max_iter (int): # iterations for the current training step.
        iter_graph (int): # iterations for optimizing the graph context loss.
        iter_inst (int): # iterations for optimizing the classification loss.
        iter_label (int): # iterations for optimizing the label context loss.
        """
        for _ in range(max_iter):
            for _ in range(self.comp_iter(iter_graph)):
                gx, gy = next(self.graph_generator)
                self.g_fn(gx, gy)

            for _ in range(self.comp_iter(iter_inst)):
                x, y, index = next(self.inst_generator)
                x = x.astype(np.float32)
                y = y.astype(np.int32)
                self.train_fn(x, y, index)

            for _ in range(self.comp_iter(iter_label)):
                gx, gy = next(self.label_generator)
                self.g_fn(gx, gy)

    def predict(self, tx, index=None):
        """predict the dev or test instances.
        tx (scipy.sparse.csr_matrix): feature vectors for dev instances.
        index (numpy.ndarray): indices for dev instances in the graph. By default, we use the indices from L to L + U - 1.

        returns (numpy.ndarray, #instacnes * #classes): classification probabilities for dev instances.
        """
        tx = tx.astype(np.float32)
        if index is None:
            index = np.arange(self.x.shape[0], self.x.shape[0] + tx.shape[0], dtype=np.int32)
        return self.test_fn(tx, index)
