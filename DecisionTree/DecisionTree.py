# -*- coding:utf-8 -*-
import pandas as pd
import math
import numpy as np
import time
import copy
import networkx as nx
import matplotlib.pyplot as plt
import pygraphviz as pgv
from Node import *



def split(data,col,mode = 'gini_index'):
    data = data.sort_values(by=col)
    gini_result = []

    for i in range(m/2,len(data)-m/2):
        leftdata = data.head(i)
        rightdata = data.tail(len(data) - i)
        l = (1 - (float(len(np.where(leftdata['quality'] == 0)[0]))/len(leftdata['quality']))**2 - (float(len(np.where(leftdata['quality'] == 1)[0]))/len(leftdata['quality']))**2)/2
        r=  (1 - (float(len(np.where(rightdata['quality'] == 0)[0]))/len(rightdata['quality']))**2 - (float(len(np.where(rightdata['quality'] == 1)[0]))/len(rightdata['quality']))**2)/2
        gini = l+r
        gini_result.append(gini)
    index = np.argmin(gini_result)
    leftdata = data.head(m/2 + index)
    rightdata = data.tail(len(data) - index - m/2)
    return leftdata,rightdata,m/2+index,gini_result[index]

def choose_col(data,left_cols,mode = 'gini'):## data 和 剩余的 cols
    leftdata,rightdata = [],[]
    index = []
    result = []
    for i in left_cols:
        l,r,ind,re = split(data,i,mode = mode)
        leftdata.append(l)
        rightdata.append(r)
        index.append(ind)
        result.append(re)
    col_ind = np.argmin(re)
    return left_cols[col_ind],leftdata[col_ind],rightdata[col_ind],index[col_ind],result[col_ind]


class DecisionTree(object):
    def __init__(self, data, leftcols, m=50):
        self.data = data
        self.node = []
        self.G = pgv.AGraph(directed=True,strict=True)
        self.root,_ = self.build_Tree(data, leftcols, m)


    def build_Tree(self, data, leftcols, m=50):

        purity = not (any(data['quality'])) or all(data['quality'])
        if len(data) > m and len(leftcols) > 0 and not purity:
            col, left, right, index, result = choose_col(data, leftcols)
            newleftcols = copy.deepcopy(leftcols)
            newleftcols.remove(col)
            root = Node(col, left[col].tail(1), result, len(self.node))
            self.G.add_node(len(self.node),label = str(col)+'\n'+'< '+str(root.split_node.iloc[0]) + '\n' + 'gini = ' + str(result) +'\n' +'data:' + str(len(data)))
            ID = root.id
            self.node.append(root)
            root.leftNode ,lID = self.build_Tree(left, newleftcols, m)
            self.G.add_edge(ID,lID)

            root.rightNode, rID = self.build_Tree(right, newleftcols, m)
            self.G.add_edge(ID,rID)

            print('node success')
            return root , ID
        else:
            label = 0 if len(np.where(data['quality'] == 0)[0]) > len(np.where(data['quality'] == 1)[0]) else 1
            ID = len(self.node)
            leaf = Leaf(label, len(self.node))
            print('leaf success')
            self.G.add_node(ID, label='leaf: \n '+str(label)+'\n' +'data:' + str(len(data)))
            self.node.append(leaf)
            return leaf , ID

    def paint_tree(self,index):
        self.G.graph_attr['epsilon'] = '0.01'

        self.G.write('fooOld.dot')
        self.G.layout('dot')  # layout with dot
        self.G.draw('Tree'+str(index)+'.png')  # write to file

    def predict(self,data):

        node = self.root
        while True:
            if isinstance(node,Leaf):
                break
            if data[node.col_name] <= node.split_node.iloc[0]:
                node = node.leftNode
            else:
                node = node.rightNode
        return node.label



def kfold_train(k,m,data):
    index = 1
    result = []
    postive = np.where(d['quality'] == 1)[0]
    negative = np.where(d['quality'] == 0)[0]

    np.random.shuffle(postive)
    np.random.shuffle(negative)

    dataset_index = []
    negative = negative[:len(postive)] ##这里均衡化

    dataset_index.extend(postive)
    dataset_index.extend(negative)



    np.random.shuffle(dataset_index)
    datalen = len(dataset_index)
    kfold = datalen/k



    for i in range(k):

        train = data.head(0)
        test = data.head(0)
        test_index = dataset_index[i*kfold:(i+1)*kfold]
        train_index = dataset_index[0:i*kfold] + dataset_index[(i+1)*kfold:]

        for j in train_index:
            train = train.append(d.iloc[j])

        for j in test_index:
            test = test.append(d.iloc[j])

        print('here the ' + str(index) + '\'s  train')
        now = time.time()
        tree = DecisionTree(train, cols, m)
        tree.paint_tree(i)
        print('tree build spend', time.time() - now, 's')
        ##train 部分
        this_test = []
        print('len test',len(test),'start :',i*kfold,'end',(i+1)*kfold)
        for j in range(len(test)):

            label = tree.predict(test.iloc[j])
            if  label == test.iloc[j]['quality']:
                this_test.append(1.0)
            else:
                this_test.append(0.0)
        accuracy = float(sum(this_test))/len(this_test) * 100
        result.append(accuracy)
        print('this tree \'s accruracy =>' + str(accuracy))
    print('avg accuracy =>' +str(sum(result)/k))

d = pd.read_csv("ex6Data.csv")
cols = list(d.columns)
m = 10
cols = cols[:-1]
k = 10 ##10折交叉验证
kfold_train(k,m,d)

