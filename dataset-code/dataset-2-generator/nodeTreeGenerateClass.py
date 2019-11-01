import random
from general.node.nodeModel import Node, Attribute
from general.node.nodeEnum import RootKey, NodeKey, LeafKey, Color, AttributeSet
from util import get_random_text
from generateRule import Operator, NodeLayer, getRule


class NodeTreeGenerator:
    def __init__(self, rule=1):
        self.rule = getRule(rule)
        self.max_depth = self.rule["max_depth"]
        self.attribute = self.rule["attributes"]
        # self.use_children_group = self.rule["use_children_group"]
        self.max_each_layer_node_num = self.rule["max_each_layer_node_num"]

    def generateNodeTree(self, parent_node: Node, parent_depth):
        node = None
        pool = []
        children_num = 1
        children_group_flag = False
        reciprocal_layer_layer = self.max_depth - parent_depth - 1
        brothers_limit = self.rule[parent_node.key]["children_brothers"]
        children_brother_node = None

        if (self.rule[parent_node.key]["children_group"] and len(self.rule[parent_node.key]["children"] == 0)) or (reciprocal_layer_layer <= 0):
            return parent_node

        if self.rule[parent_node.key]["children_group"]:
            if ((self.rule[parent_node.key]["children_group"]["enable"] == Operator.random) and (random.choice([False, True]))) or (self.rule[parent_node.key]["children_group"]["enable"]) or (reciprocal_layer_layer <=1):
                pool = self.rule[parent_node.key]["children_group"]["nodes"]
                children_group_flag = True
        
        if not children_group_flag:
            pool = self.rule[parent_node.key]["children"]
            for node in pool:
                if children_group_flag in self.rule[node.value]["disabled_reciprocal_layer"]:
                    pool.remove(node)

        print("pool num: ", len(pool))

        if children_group_flag:
            children_num = len(pool)
        elif self.rule[parent_node.key]["children_quantity"]:
            if self.rule[parent_node.key]["children_quantity"]["operator"] == Operator.equal:
                children_num = self.rule[parent_node.key]["children_quantity"]["value"]
            elif self.rule[parent_node.key]["children_quantity"]["operator"] == Operator.more_then:
                children_num = random.randrange(
                    self.rule[parent_node.key]["children_quantity"]["value"], self.max_each_layer_node_num)
        else:
            children_num = random.randrange(1, self.max_each_layer_node_num)
        print("children num: ", children_num)

        for index in range(children_num):
            if children_group_flag:
                node = generateNode(
                    parent_node,  parent_depth+1, pool[index].value)
            elif brothers_limit == Operator.same:
                if index == 0:
                    node = generateNode(
                        parent_node,  parent_depth+1, random.choice(pool).value)
                    children_brother_node = node.key
                else:
                    node = generateNode(
                        parent_node,  parent_depth+1, children_brother_node)
            else:
                node = generateNode(
                    parent_node,  parent_depth+1, random.choice(pool).value)

            node = generateNodeTree(node, parent_depth+1)
            parent_node.add_child(node) 

        return parent_node

    def generateNode(self, parent_node, depth, assigned_key):
        attribute = Attribute(self.attribute, self.rule[assigned_key]["attributes"])
        # node = Node(parent_node, )
        for i,  enabled in enumerate(attribute.enabledAttributes):
            value = None
            if enabled:
                if attribute.activatedAttributes[i] == AttributeSet.content:
                    value = get_random_text(5 if parent_node.key == NodeKey.button.value else 10 )
                else:
                    value = random.choice(list(attribute.activatedAttributes[i].value)).value
                attribute.assign_value(i, value)

        return Node(assigned_key, parent_node, attribute, depth)
