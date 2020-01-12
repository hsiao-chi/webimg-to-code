from enum import Enum

class RootKey(Enum):
    body = 'body'


class NodeKey(Enum):
    row = 'row'
    col = 'col'
    list = 'list'
    absolute = 'absolute'
    button = 'btn'


class LeafKey(Enum):
    div = 'div'
    text = 'text'
    title = 'title'
    icon = 'icon'
    button = 'btn'




class Font_color(Enum):
    primary = 'text-primary'
    success = 'text-success'
    warning = 'text-warning'
    danger = 'text-danger'
    info = 'text-info'
    dark = 'text-dark'
    light = 'text-light'
    secondary = 'text-secondary'
    white = 'text-white'

class Bg_color(Enum):
    primary = 'bg-primary'
    success = 'bg-success'
    warning = 'bg-warning'
    danger = 'bg-danger'
    info = 'bg-info'
    dark = 'bg-dark'
    light = 'bg-light'
    secondary = 'bg-secondary'
    white = 'bg-white'
    # activate = 'bg-activate'
    # inactivate = 'bg-inactivate'
    
class Size(Enum):
    lg = 'lg'
    sm = 'sm'


class AttributeSet(Enum):
    font_color = 'font_color'
    bg_color = 'bg_color'
    content = 'string'

class Tag(Enum):
    node_opening = '{'
    node_closing = '}'
    attr_opening = '['
    attr_closing = ']'


class Placeholder(Enum):
    node = '{}'
    data_title = '$data_title'
    color = '$color'
    bg_color = '$bg_color'
    size = '$size'
    content = '$content'
    leaf_col = '$leaf_col'
    col = "$col"