from enum import Enum
from general.node.nodeEnum import RootKey, NodeKey, LeafKey


class Operator(Enum):
    none = None
    same = "same"
    whole = "whole"  # any
    more_then = "more_then"
    equal = "equal"


class NodeLayer(Enum):
    first = 'first'
    last = 'last'
    last_1 = 'last_1'
    whole = 'whole'  # any


def getRule(rule=0):
    if rule == 0:
        return type1_rule


type1_rule = {
    "ATTRIBUTES": ['font_color', 'bg_color', 'content'],
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
