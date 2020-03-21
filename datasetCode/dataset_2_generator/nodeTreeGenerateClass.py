import random
from general.node.nodeModel import Node, Attribute
from general.node.nodeEnum import RootKey, NodeKey, LeafKey, Font_color, Bg_color, AttributeSet
from datasetCode.dataset_2_generator.util import get_random_text
from datasetCode.dataset_2_generator.generateRule import Operator, NodeLayer, getRule


class NodeTreeGenerator:
    def __init__(self, rule=1):
        self.rule = getRule(rule)
        self.max_depth = self.rule["max_depth"]
        self.attribute = self.rule["attributes"]
        # self.use_children_group = self.rule["use_children_group"]
        self.max_each_layer_node_num = self.rule["max_each_layer_node_num"]
        # print("max depth: ", self.max_depth, "  attribute: ", self.attribute, "  max each layer: ", self.max_each_layer_node_num)

    def generateNodeTree(self, parent_node: Node, parent_depth):
        node = None
        pool = []
        children_num = 1
        children_group_flag = False
        reciprocal_layer_layer = self.max_depth - parent_depth - 1
        brothers_limit = self.rule[parent_node.key]["children_brothers"]
        children_brother_node = None
        # print("-----\n", parent_node.key)
        # print("children len: ", len(self.rule[parent_node.key]["children"]), " / group: ", self.rule[parent_node.key]["children_group"] , " / reciprocal: ", (reciprocal_layer_layer))
        if ((self.rule[parent_node.key]["children_group"] == None) and len(self.rule[parent_node.key]["children"]) == 0) or (reciprocal_layer_layer < 0):
            return parent_node

        if self.rule[parent_node.key]["children_group"]:
            
            if self.rule[parent_node.key]["children_group"]["enable"] == Operator.random:
                children_group_flag = random.choice([False, True])
                # print("random group: ", children_group_flag)
            elif self.rule[parent_node.key]["children_group"]["enable"] == Operator.true:
                children_group_flag = True
            if reciprocal_layer_layer <=1:
                children_group_flag = True

        if children_group_flag:
            if len(self.rule[parent_node.key]["children_group"]["nodes"])> 1:
                i = random.randrange(len(self.rule[parent_node.key]["children_group"]["nodes"]))
            else:
                i=0
            pool = self.rule[parent_node.key]["children_group"]["nodes"][i]
        else:
            pool = self.rule[parent_node.key]["children"]
            for node in pool:
                if reciprocal_layer_layer in self.rule[node.value]["disabled_reciprocal_layer"]:
                    pool.remove(node)


        if children_group_flag:
            children_num = len(pool)
        elif self.rule[parent_node.key]["children_quantity"]:

            if self.rule[parent_node.key]["children_quantity"]["operator"] == Operator.equal:
                children_num = self.rule[parent_node.key]["children_quantity"]["value"]
            elif self.rule[parent_node.key]["children_quantity"]["operator"] == Operator.equal_more_then:
                children_num = random.randrange(
                    self.rule[parent_node.key]["children_quantity"]["value"], self.max_each_layer_node_num)
            elif self.rule[parent_node.key]["children_quantity"]["operator"] == Operator.between:
                children_num = random.randrange(
                    self.rule[parent_node.key]["children_quantity"]["value"][0], self.rule[parent_node.key]["children_quantity"]["value"][1])
            
            if parent_node.key == NodeKey.row.value and parent_depth == 1:
                if self.rule[parent_node.key]["children_quantity"]["operator"] == Operator.equal_more_then:
                    children_num = random.randrange(
                    1, self.max_each_layer_node_num)
                elif self.rule[parent_node.key]["children_quantity"]["operator"] == Operator.between:
                    children_num = random.randrange(
                        1, self.rule[parent_node.key]["children_quantity"]["value"][1])
        


        else:
            children_num = random.randrange(1, self.max_each_layer_node_num)
        
        for index in range(children_num):
            if children_group_flag:
                node = self.generateNode(
                    parent_node,  parent_depth+1, pool[index].value)
            elif brothers_limit == Operator.same:
                if index == 0:
                    node = self.generateNode(
                        parent_node,  parent_depth+1, random.choice(pool).value)
                    children_brother_node = node.key
                else:
                    node = self.generateNode(
                        parent_node,  parent_depth+1, children_brother_node)
            else:
                node = self.generateNode(
                    parent_node,  parent_depth+1, random.choice(pool).value)

            node = self.generateNodeTree(node, parent_depth+1)
            parent_node.add_child(node) 

        return parent_node

    def generateNode(self, parent_node, depth, assigned_key):
        attribute = Attribute(self.attribute, self.rule[assigned_key]["attributes"])
        # node = Node(parent_node, )
        for i in range(len(self.rule[assigned_key]["attributes"])):
            attribute.assign_value(i, None)
        for i,  enabled in enumerate(attribute.enabledAttributes):
            value = None
            if enabled:
                if attribute.activatedAttributes[i] == AttributeSet.content:
                    if assigned_key == LeafKey.button.value:
                        value = "\"" + get_random_text(10, 1) + "\""
                    elif assigned_key == LeafKey.title.value:
                        value = "\"" + get_random_text(5, 0) + "\""
                    else:
                        value = "\"" + get_random_text(30, 5, False) + "\""
                    attribute.assign_value(i, value)
                else:
                    try:
                        group_value = random.choice(list(self.rule[assigned_key]["attributes_set"]["groups"]))
                        for j, v in enumerate(group_value):
                            attribute.assign_value(i+j, v.value)
                    except KeyError:
                        value = random.choice(list(self.rule[assigned_key]["attributes_set"][attribute.activatedAttributes[i].value])).value
                        attribute.assign_value(i, value)

        return Node(assigned_key, parent_node, attribute, depth)
