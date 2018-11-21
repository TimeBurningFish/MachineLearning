# -*- coding:utf-8 -*-
class Node(object):
    def __init__(self,col_name,split_node,gini_result,ID):
        self.col_name = col_name
        self.split_node = split_node
        self.gini_result = gini_result
        self.leftNode = None
        self.rightNode = None
        self.id = ID
class Leaf(object):
    def __init__(self,label,ID):
        self.label = label
        self.ID = ID