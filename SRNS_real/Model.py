import os
import pdb
import tensorflow as tf
import numpy as np
from time import time
import pickle
import sys


class BPR():
    def __init__(self, args, num_user, num_item):
        self.learning_rate = args.lr
        self.opt = args.optimizer
        self.regs = float(args.regs)
        self.batch_size=args.batch_size
        self.model_file = args.model_file
        self.num_user = num_user
        self.num_item = num_item
        self.embedding_size = args.embedding_size
        self.use_pretrain = args.use_pretrain
        self.model = args.model


    def _create_placeholders(self):
        self.user_input=tf.placeholder(tf.int32,shape=[None,1],name="user_input")
        self.item_input_pos=tf.placeholder(tf.int32,shape=[None,1],name="item_input_pos")
        self.item_input_neg=tf.placeholder(tf.int32,shape=[None,1],name="item_input_neg")
        self.example_weight=tf.placeholder(tf.float32,shape=[None,1],name="example_weight")


    def _create_variables(self, model):

        self.embeddingmap_user = tf.Variable(
                tf.truncated_normal(shape=[self.num_user, self.embedding_size], mean=0.0, stddev=0.01),
                                    name='embedding_user', dtype=tf.float32)
        self.embeddingmap_item = tf.Variable(
                tf.truncated_normal(shape=[self.num_item, self.embedding_size], mean=0.0, stddev=0.01),
                                    name='embedding_item', dtype=tf.float32)
        self.h = tf.Variable(tf.random_uniform([self.embedding_size, 1], minval=-tf.sqrt(6 / (self.embedding_size + 1)),
                                    maxval=tf.sqrt(6 / (self.embedding_size + 1))), name='h')

        
    def _create_loss(self):
        # score, score_neg, 
        embedding_user = tf.nn.embedding_lookup(self.embeddingmap_user,self.user_input)
        embedding_user = tf.reshape(embedding_user,[-1,self.embedding_size])

        embedding_item_pos = tf.nn.embedding_lookup(self.embeddingmap_item,self.item_input_pos)
        embedding_item_pos = tf.reshape(embedding_item_pos,[-1,self.embedding_size])

        embedding_item_neg = tf.nn.embedding_lookup(self.embeddingmap_item,self.item_input_neg)
        embedding_item_neg = tf.reshape(embedding_item_neg,[-1,self.embedding_size])

        self.score = tf.reshape(tf.matmul(embedding_user * embedding_item_pos, self.h, name='output'),[-1,1])
        self.score_neg = tf.reshape(tf.matmul(embedding_user * embedding_item_neg, self.h, name='output_neg'),[-1,1])

        self.regularizer = tf.contrib.layers.l2_regularizer(self.regs)
        if self.regs!=0.0:
            self.loss_reg = self.regularizer(embedding_user)+self.regularizer(embedding_item_pos)+self.regularizer(embedding_item_neg)
        else:
            self.loss_reg = 0

        self.loss_vanilla = tf.reduce_sum(self.example_weight * tf.log(1 + tf.exp(self.score_neg - self.score - 1e-8)))
        self.loss = self.loss_vanilla + self.loss_reg


    def _create_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate,name='adam_opt').minimize(self.loss)


    def build_graph(self):
        graph = tf.get_default_graph()
        with graph.as_default() as g:
            with g.name_scope("GMF"):
                self._create_placeholders()
                self._create_variables(self.model_file)
                self._create_loss()
                self._create_optimizer()
       

    def save_model_withpath(self,sess,id):
        param = []
        embedding=sess.run([self.embeddingmap_user, self.embeddingmap_item, self.h])
        for each in embedding:
            param.append(each)
        pickle.dump(param,open(id,'wb'))


class NCF():
    def __init__(self, args, num_user, num_item):
        self.learning_rate = args.lr
        self.opt = args.optimizer
        self.regs = float(args.regs)
        self.batch_size=args.batch_size
        self.model_file = args.model_file
        self.num_user = num_user
        self.num_item = num_item
        self.embedding_size = args.embedding_size // 2
        self.use_pretrain = args.use_pretrain
        self.model = args.model


    def _create_placeholders(self):
        self.user_input=tf.placeholder(tf.int32,shape=[None,1],name="user_input")
        self.item_input_pos=tf.placeholder(tf.int32,shape=[None,1],name="item_input_pos")
        self.item_input_neg=tf.placeholder(tf.int32,shape=[None,1],name="item_input_neg")
        self.example_weight=tf.placeholder(tf.float32,shape=[None,1],name="example_weight")


    def _create_variables(self, model):

        self.embed_user_GMF = tf.Variable(
                tf.truncated_normal(shape=[self.num_user, self.embedding_size], mean=0.0, stddev=0.01),
                                    name='embedding_user_GMF', dtype=tf.float32)
        self.embed_item_GMF = tf.Variable(
                tf.truncated_normal(shape=[self.num_item, self.embedding_size], mean=0.0, stddev=0.01),
                                    name='embedding_item_GMF', dtype=tf.float32)

        self.embed_user_MLP = tf.Variable(
                tf.truncated_normal(shape=[self.num_user, self.embedding_size], mean=0.0, stddev=0.01),
                                    name='embedding_user_MLP', dtype=tf.float32)
        self.embed_item_MLP = tf.Variable(
                tf.truncated_normal(shape=[self.num_item, self.embedding_size], mean=0.0, stddev=0.01),
                                    name='embedding_item_MLP', dtype=tf.float32)
        
        self.w0 = tf.Variable(
                tf.truncated_normal(shape=[self.embedding_size*2, self.embedding_size], mean=0.0, stddev=0.01),
                                    name='w0', dtype=tf.float32)
        self.b0 = tf.Variable(
                tf.truncated_normal(shape=[self.embedding_size], mean=0.0, stddev=0.01), name='b0', dtype=tf.float32)
        
        self.w1 = tf.Variable(
                tf.truncated_normal(shape=[self.embedding_size, self.embedding_size], mean=0.0, stddev=0.01),
                                    name='w1', dtype=tf.float32)
        self.b1 = tf.Variable(
                tf.truncated_normal(shape=[self.embedding_size], mean=0.0, stddev=0.01), name='b1', dtype=tf.float32)
        
        self.w2 = tf.Variable(
                tf.truncated_normal(shape=[self.embedding_size, self.embedding_size], mean=0.0, stddev=0.01),
                                    name='w2', dtype=tf.float32)
        self.b2 = tf.Variable(
                tf.truncated_normal(shape=[self.embedding_size], mean=0.0, stddev=0.01), name='b2', dtype=tf.float32)

        self.h = tf.Variable(tf.random_uniform([self.embedding_size*2, 1], minval=-tf.sqrt(6 / (self.embedding_size*2 + 1)),
                                    maxval=tf.sqrt(6 / (self.embedding_size*2 + 1))), name='h')

        
    def _create_loss(self):

        embedding_user_GMF = tf.nn.embedding_lookup(self.embed_user_GMF,self.user_input)
        embedding_user_GMF = tf.reshape(embedding_user_GMF,[-1,self.embedding_size])

        embedding_item_pos_GMF = tf.nn.embedding_lookup(self.embed_item_GMF,self.item_input_pos)
        embedding_item_pos_GMF = tf.reshape(embedding_item_pos_GMF,[-1,self.embedding_size])

        embedding_item_neg_GMF = tf.nn.embedding_lookup(self.embed_item_GMF,self.item_input_neg)
        embedding_item_neg_GMF = tf.reshape(embedding_item_neg_GMF,[-1,self.embedding_size])

        embedding_user_MLP = tf.nn.embedding_lookup(self.embed_user_MLP,self.user_input)
        embedding_user_MLP = tf.reshape(embedding_user_MLP,[-1,self.embedding_size])

        embedding_item_pos_MLP = tf.nn.embedding_lookup(self.embed_item_MLP,self.item_input_pos)
        embedding_item_pos_MLP = tf.reshape(embedding_item_pos_MLP,[-1,self.embedding_size])

        embedding_item_neg_MLP = tf.nn.embedding_lookup(self.embed_item_MLP,self.item_input_neg)
        embedding_item_neg_MLP = tf.reshape(embedding_item_neg_MLP,[-1,self.embedding_size])

        pos_output_GMF = embedding_user_GMF * embedding_item_pos_GMF
        neg_output_GMF = embedding_user_GMF * embedding_item_neg_GMF

        pos_output_MLP = tf.concat([embedding_user_MLP, embedding_item_pos_MLP], -1)
        neg_output_MLP = tf.concat([embedding_user_MLP, embedding_item_neg_MLP], -1)

        pos_output_MLP = tf.nn.relu(tf.add(tf.matmul(tf.nn.dropout(pos_output_MLP, 0.8), self.w0), self.b0))
        pos_output_MLP = tf.nn.relu(tf.add(tf.matmul(tf.nn.dropout(pos_output_MLP, 0.8), self.w1), self.b1))
        pos_output_MLP = tf.nn.relu(tf.add(tf.matmul(tf.nn.dropout(pos_output_MLP, 0.8), self.w2), self.b2))

        neg_output_MLP = tf.nn.relu(tf.add(tf.matmul(tf.nn.dropout(neg_output_MLP, 0.8), self.w0), self.b0))
        neg_output_MLP = tf.nn.relu(tf.add(tf.matmul(tf.nn.dropout(neg_output_MLP, 0.8), self.w1), self.b1))
        neg_output_MLP = tf.nn.relu(tf.add(tf.matmul(tf.nn.dropout(neg_output_MLP, 0.8), self.w2), self.b2))

        self.score = tf.reshape(tf.matmul(tf.concat([pos_output_GMF, pos_output_MLP], -1), self.h, name='output'),[-1,1])
        self.score_neg = tf.reshape(tf.matmul(tf.concat([neg_output_GMF, neg_output_MLP], -1), self.h, name='output_neg'),[-1,1])

        self.regularizer = tf.contrib.layers.l2_regularizer(self.regs)
        if self.regs!=0.0:
            self.loss_reg = self.regularizer(embedding_user_GMF)+self.regularizer(embedding_item_pos_GMF)+self.regularizer(embedding_item_neg_GMF)+\
                                self.regularizer(embedding_user_MLP)+self.regularizer(embedding_item_pos_MLP)+self.regularizer(embedding_item_neg_MLP)
        else:
            self.loss_reg = 0

        self.loss_vanilla = tf.reduce_sum(self.example_weight * tf.log(1 + tf.exp(self.score_neg - self.score - 1e-8)))
        self.loss = self.loss_vanilla + self.loss_reg


    def _create_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate,name='adam_opt').minimize(self.loss)


    def build_graph(self):
        graph = tf.get_default_graph()
        with graph.as_default() as g:
            with g.name_scope("GMF"):
                self._create_placeholders()
                self._create_variables(self.model_file)
                self._create_loss()
                self._create_optimizer()
       

    def save_model_withpath(self,sess,id):
        param = []
        embedding=sess.run([self.embeddingmap_user, self.embeddingmap_item, 
                            self.w0, self.b0, self.w1, self.b1, self.w2, self.b2, self.h])
        for each in embedding:
            param.append(each)
        pickle.dump(param,open(id,'wb'))


class CML():
    def __init__(self, args, num_user, num_item):
        self.learning_rate = args.lr
        self.opt = args.optimizer
        self.regs = float(args.regs)
        self.batch_size=args.batch_size
        self.model_file = args.model_file
        self.num_user = num_user
        self.num_item = num_item
        self.embedding_size = args.embedding_size
        self.use_pretrain = args.use_pretrain
        self.model = args.model


    def _create_placeholders(self):
        self.user_input=tf.placeholder(tf.int32,shape=[None,1],name="user_input")
        self.item_input_pos=tf.placeholder(tf.int32,shape=[None,1],name="item_input_pos")
        self.item_input_neg=tf.placeholder(tf.int32,shape=[None,1],name="item_input_neg")
        self.example_weight=tf.placeholder(tf.float32,shape=[None,1],name="example_weight")


    def _create_variables(self, model):

        self.embeddingmap_user = tf.Variable(
                tf.truncated_normal(shape=[self.num_user, self.embedding_size], mean=0.0, stddev=0.01),
                                    name='embedding_user', dtype=tf.float32)
        self.embeddingmap_item = tf.Variable(
                tf.truncated_normal(shape=[self.num_item, self.embedding_size], mean=0.0, stddev=0.01),
                                    name='embedding_item', dtype=tf.float32)

        
    def _create_loss(self):
        # score, score_neg, 
        embedding_user = tf.nn.embedding_lookup(self.embeddingmap_user,self.user_input)
        embedding_user = tf.reshape(embedding_user,[-1,self.embedding_size])

        embedding_item_pos = tf.nn.embedding_lookup(self.embeddingmap_item,self.item_input_pos)
        embedding_item_pos = tf.reshape(embedding_item_pos,[-1,self.embedding_size])

        embedding_item_neg = tf.nn.embedding_lookup(self.embeddingmap_item,self.item_input_neg)
        embedding_item_neg = tf.reshape(embedding_item_neg,[-1,self.embedding_size])

        self.score = tf.reshape(tf.math.reduce_sum(-tf.math.square(embedding_user - embedding_item_pos), -1),[-1,1])
        self.score_neg = tf.reshape(tf.math.reduce_sum(-tf.math.square(embedding_user - embedding_item_neg), -1),[-1,1])

        self.regularizer = tf.contrib.layers.l2_regularizer(self.regs)
        if self.regs!=0.0:
            self.loss_reg = self.regularizer(embedding_user)+self.regularizer(embedding_item_pos)+self.regularizer(embedding_item_neg)
        else:
            self.loss_reg = 0

        self.loss_vanilla = tf.reduce_sum(tf.nn.relu(self.score_neg - self.score + 1.0))
        self.loss = self.loss_vanilla + self.loss_reg


    def _create_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate,name='adam_opt').minimize(self.loss)


    def build_graph(self):
        graph = tf.get_default_graph()
        with graph.as_default() as g:
            with g.name_scope("GMF"):
                self._create_placeholders()
                self._create_variables(self.model_file)
                self._create_loss()
                self._create_optimizer()
       

    def save_model_withpath(self,sess,id):
        param = []
        embedding=sess.run([self.embeddingmap_user, self.embeddingmap_item, self.h])
        for each in embedding:
            param.append(each)
        pickle.dump(param,open(id,'wb'))


class LGN():
    def __init__(self, args, num_user, num_item, graph):
        self.learning_rate = args.lr
        self.opt = args.optimizer
        self.regs = float(args.regs)
        self.batch_size=args.batch_size
        self.model_file = args.model_file
        self.num_user = num_user
        self.num_item = num_item
        self.embedding_size = args.embedding_size
        self.use_pretrain = args.use_pretrain
        self.model = args.model
        self.graph = graph


    def _create_placeholders(self):
        self.user_input=tf.placeholder(tf.int32,shape=[None,1],name="user_input")
        self.item_input_pos=tf.placeholder(tf.int32,shape=[None,1],name="item_input_pos")
        self.item_input_neg=tf.placeholder(tf.int32,shape=[None,1],name="item_input_neg")
        self.example_weight=tf.placeholder(tf.float32,shape=[None,1],name="example_weight")


    def _create_variables(self, model):

        self.embeddingmap_user = tf.Variable(
                tf.truncated_normal(shape=[self.num_user, self.embedding_size], mean=0.0, stddev=0.01),
                                    name='embedding_user', dtype=tf.float32)
        self.embeddingmap_item = tf.Variable(
                tf.truncated_normal(shape=[self.num_item, self.embedding_size], mean=0.0, stddev=0.01),
                                    name='embedding_item', dtype=tf.float32)

        coo = self.graph.tocoo().astype(np.float32)
        index = tf.transpose(tf.stack([tf.constant(coo.row, dtype=tf.int64), tf.constant(coo.col, dtype=tf.int64)]))
        self.sparse_graph = tf.sparse.SparseTensor(index, tf.constant(coo.data, dtype=tf.float32), coo.shape)

        
    def _create_loss(self):

        all_emb = tf.concat([self.embeddingmap_user, self.embeddingmap_item], 0)
        embs = all_emb

        for layer in range(3):
            all_emb = tf.sparse.sparse_dense_matmul(self.sparse_graph, all_emb)
            embs += all_emb

        user_emb, item_emb = tf.split(embs, [self.num_user, self.num_item], axis=0)

        # score, score_neg, 
        embedding_user = tf.nn.embedding_lookup(user_emb,self.user_input)
        embedding_user = tf.reshape(embedding_user,[-1,self.embedding_size])

        embedding_item_pos = tf.nn.embedding_lookup(item_emb,self.item_input_pos)
        embedding_item_pos = tf.reshape(embedding_item_pos,[-1,self.embedding_size])

        embedding_item_neg = tf.nn.embedding_lookup(item_emb,self.item_input_neg)
        embedding_item_neg = tf.reshape(embedding_item_neg,[-1,self.embedding_size])

        self.score = tf.reshape(tf.reduce_sum(embedding_user * embedding_item_pos, axis=1),[-1,1])
        self.score_neg = tf.reshape(tf.reduce_sum(embedding_user * embedding_item_neg, axis=1),[-1,1])

        self.regularizer = tf.contrib.layers.l2_regularizer(self.regs)
        if self.regs!=0.0:
            self.loss_reg = self.regularizer(self.embeddingmap_user)+self.regularizer(self.embeddingmap_item)
        else:
            self.loss_reg = 0

        self.loss_vanilla = tf.reduce_sum(tf.log(1 + tf.exp(self.score_neg - self.score - 1e-8)))
        self.loss = self.loss_vanilla + self.loss_reg


    def _create_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate,name='adam_opt').minimize(self.loss)


    def build_graph(self):
        graph = tf.get_default_graph()
        with graph.as_default() as g:
            with g.name_scope("GMF"):
                self._create_placeholders()
                self._create_variables(self.model_file)
                self._create_loss()
                self._create_optimizer()
       

    def save_model_withpath(self,sess,id):
        param = []
        embedding=sess.run([self.embeddingmap_user, self.embeddingmap_item, self.h])
        for each in embedding:
            param.append(each)
        pickle.dump(param,open(id,'wb'))