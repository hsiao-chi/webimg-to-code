from enum import Enum


class Path(Enum):
    assest = 'E:\\projects\\NTUST\\webimg-to-code\\dataset-2-generator\\assests\\'


class RootKey(Enum):
    body = 'body'


class NodeKey(Enum):
    row = 'row'
    col = 'col'
    list = 'list'
    absolute = 'absolute'
    button = 'button'


class LeafKey(Enum):
    div = 'div'
    text = 'text'
    title = 'title'
    icon = 'icon'


class Color(Enum):
    primary = 'primary'
    success = 'success'
    warning = 'warning'
    danger = 'dander'
    info = 'info'
    dark = 'dark'
    light = 'light'
    secondary = 'secondary'
    white = 'white'


class Size(Enum):
    lg = 'lg'
    sm = 'sm'


class Tag(Enum):
    node_opening = '{'
    node_closing = '}'
    attr_opening = '['
    attr_closing = ']'


class Placeholder(Enum):
    color = '$color'
    bgcolor = '$bgcolor'
    size = '$size'
    context = '$context'


have_font_color_attr_node = [LeafKey.text, LeafKey.title]
have_bg_color_attr_node = [LeafKey.div, NodeKey.absolute, NodeKey.button]
have_size_attr_node = [NodeKey.button, LeafKey.title, NodeKey.button]
have_context_attr_node = [LeafKey.text, LeafKey.title]


GENERATE_RULE = {
    "attributes": {
        "have_font_color_node": [LeafKey.text, LeafKey.title],
        "have_bg_color_node": [LeafKey.div, NodeKey.button],
        "have_size_node": [NodeKey.button, LeafKey.title],
        "have_context_node": [LeafKey.text, LeafKey.title],
    },
    "enabled_children": {
        NodeKey.button.value: [LeafKey.text, LeafKey.icon],
        NodeKey.row.value: [NodeKey.col],
        NodeKey.col.value: [NodeKey.button, NodeKey.list] + list(LeafKey),
        NodeKey.list.value: list(LeafKey),
        NodeKey.absolute.value: [NodeKey.row, NodeKey.list],
    },
    "enabled_parent": {
        NodeKey.button.value: [NodeKey.col, NodeKey.list],
        NodeKey.row.value: [RootKey.body, NodeKey.absolute],
        NodeKey.col.value: [NodeKey.row],
        NodeKey.list.value: [NodeKey.col, NodeKey.absolute],
        NodeKey.absolute.value: [RootKey.body],
    }
}
