from enum import Enum

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




class Font_color(Enum):
    primary = 'primary'
    success = 'success'
    warning = 'warning'
    danger = 'danger'
    info = 'info'
    dark = 'dark'
    # light = 'light'
    secondary = 'secondary'
    # white = 'white'

class Bg_color(Enum):
    primary = 'primary'
    success = 'success'
    warning = 'warning'
    danger = 'danger'
    info = 'info'
    dark = 'dark'
    # light = 'light'
    secondary = 'secondary'
    # white = 'white'
    
class AttributeSet(Enum):
    font_color = Font_color
    bg_color = Bg_color
    content = 'string'

class Size(Enum):
    lg = 'lg'
    sm = 'sm'


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