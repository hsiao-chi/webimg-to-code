import random
from general.node.nodeModel import Node, Attribute
from general.node.nodeEnum import RootKey, NodeKey, LeafKey, Color
from util import get_random_text
from generateRule import Operator, NodeLayer, getRule


class NodeTreeGenerator:
    def __init__(self, rule=1):
        self.rule = getRule(rule)
        self.max_depth = self.rule["max_depth"]
        self.attribute = self.rule["attributes"]
        self.use_children_group = self.rule["use_children_group"]
        self.max_each_layer_node_num = self.rule["max_each_layer_node_num"]

    def generateNodeTree(self, parent_node: Node, depth):
        node = None
        pool = []
        children_num = 0
        children_group_flag = False
        if self.rule[parent_node.key]["children_group"] > 0:
            if ((self.use_children_group == Operator.random) and (random.choice([False, True]))) or (self.use_children_group == Operator.true):
                pool = self.rule[parent_node.key]["children_group"]
                children_group_flag = True
        else: 
            pool = self.rule[parent_node.key]["children"]

        if children_group_flag:
            children_num = len(pool)
        elif len(self.rule[parent_node.key]["children_quantity"]) > 0:
            
            pass
        else: 
            children_num = random.randrange(1, self.max_each_layer_node_num)
        # node_num = if len(self.rule[parent_node.key]["children_quantity"]) > 0
        pass

    def generateNode(self, parent_node, depth, pool):
        pass
