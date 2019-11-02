from enum import Enum
from general.node.nodeEnum import RootKey, NodeKey, LeafKey, AttributeSet


class Operator(Enum):
    none = None
    same = "same"
    whole = "whole"  # any
    more_then = "more_then"
    equal_more_then = "equal_more_then"

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
    # "use_children_group": Operator.random,
    "max_each_layer_node_num": 5,
    "max_depth": 6,
    RootKey.body.value: {
        "attributes": [False, False, False],
        "children": [NodeKey.row],
        "parents": [],
        "children_brothers": Operator.none,
        "disabled_reciprocal_layer": [],
        "children_group": None,
        "children_quantity": None
    },
    NodeKey.row.value: {
        "attributes": [False, False, False],
        "children": [NodeKey.col],
        "parents": [RootKey.body, NodeKey.col],
        "children_brothers": Operator.whole,
        "disabled_reciprocal_layer": [0, 1],
        "children_group": None,
        "children_quantity": {
            "operator": Operator.equal_more_then,
            "value": 2,
        }
    },
    NodeKey.col.value: {
        "attributes": [False, False, False],
        "children": [NodeKey.row],
        "parents": [NodeKey.row],
        "children_brothers": Operator.whole,
        "disabled_reciprocal_layer": [0, 1],
        "children_group": {
            "enable": Operator.random,
            "nodes": [LeafKey.title, LeafKey.text, NodeKey.button]
        },
        "children_quantity": {
            "operator": Operator.equal,
            "value": 1,
        }
    },
    NodeKey.button.value: {
        "attributes": [False, True, False],
        "children": [],
        "parents": [NodeKey.col],
        "children_brothers": Operator.whole,
        "disabled_reciprocal_layer": [0],
        "children_group": {
            "enable": Operator.true,
            "nodes": [LeafKey.text]
        },
        "children_quantity": {
            "operator": Operator.equal,
            "value": 1,
        }
    },
    LeafKey.text.value: {
        "attributes": [True, False, True],
        "children": [],
        "parents": [NodeKey.col, NodeKey.button],
        "children_brothers": Operator.whole,
        "disabled_reciprocal_layer": [],
        "children_group": None,
        "children_quantity": None
    },
    LeafKey.title.value: {
        "attributes": [True, False, True],
        "children": [],
        "parents": [NodeKey.col],
        "children_brothers": Operator.whole,
        "disabled_reciprocal_layer": [],
        "children_group": None,
        "children_quantity": None
    },
}
