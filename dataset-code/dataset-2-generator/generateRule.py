from enum import Enum
from general.node.nodeEnum import RootKey, NodeKey, LeafKey, AttributeSet


class Operator(Enum):
    none = None
    same = "same"
    whole = "whole"  # any
    more_then = "more_then"
    equal = "equal"
    random = 'random'
    true = True
    false = False


class NodeLayer(Enum):
    first = 'first'
    last = 'last'
    last_1 = 'last_1'
    whole = 'whole'  # any


def getRule(rule=1):
    if rule == 1:
        return rule_1


rule_1 = {
    "attributes": [AttributeSet.font_color, AttributeSet.bg_color, AttributeSet.content],
    "use_children_group": Operator.random,
    "max_each_layer_node_num": 5,
    "max_depth": 6,
    RootKey.body.value: {
        "attributes": [],
        "children": [NodeKey.row],
        "parents": [],
        "brothers": Operator.none,
        "disabled_layer": [],
        "children_group": [],
        "children_quantity": []
    },
    NodeKey.row.value: {
        "attributes": [],
        "children": [NodeKey.col],
        "parents": [RootKey.body, NodeKey.col],
        "brothers": Operator.whole,
        "disabled_layer": [NodeLayer.last_1, NodeLayer.last],
        "children_group": [],
        "children_quantity": [
            {
                "child": NodeKey.col,
                "operator": Operator.more_then,
                "value": 1,
            }
        ]
    },
    NodeKey.col.value: {
        "attributes": [],
        "children": [NodeKey.row],
        "parents": [NodeKey.row],
        "brothers": Operator.whole,
        "disabled_layer": [NodeLayer.last, NodeLayer.last_1],
        "children_group": [LeafKey.title, LeafKey.text, NodeKey.button],
        "children_quantity": [
            {
                "child": NodeKey.row,
                "operator": Operator.equal,
                "value": 1,
            }
        ]
    },
    NodeKey.button.value: {
        "attributes": [False, True, False],
        "children": [LeafKey.text],
        "parents": [NodeKey.col],
        "brothers": Operator.whole,
        "disabled_layer": [NodeLayer.last],
        "children_group": [],
        "children_quantity": [
            {
                "child": LeafKey.text,
                "operator": Operator.equal,
                "value": 1,
            }
        ]
    },
    LeafKey.text.value: {
        "attributes": [True, False, True],
        "children": [],
        "parents": [NodeKey.col, NodeKey.button],
        "brothers": Operator.whole,
        "disabled_layer": [],
        "children_group": [],
        "children_quantity": []
    },
    LeafKey.title.value: {
        "attributes": [True, False, True],
        "children": [],
        "parents": [NodeKey.col],
        "brothers": Operator.whole,
        "disabled_layer": [],
        "children_group": [],
        "children_quantity": []
    },
}
