from config import NodeKey, Tag, LeafKey


class Node:
    def __init__(self, key, parent_node, attributes, depth = 0):
        self.key = key
        self.parent = parent_node
        self.children = []
        self.attributes = attributes
        self.depth = depth
        # self.content_holder = content_holder

    def add_child(self, child):
        self.children.append(child)

    def show(self):
        if self.key in [key.value for key in list(LeafKey)]:
            print("{}{} {} ".format('\t'*self.depth, self.key, self.attributes.toString()))
        else:
            if self.attributes.isEmpty():
                print("{}{} {} ".format('\t'*self.depth, self.key, Tag.node_opening.value))
            else:
                print("{}{} {} {} ".format('\t'*self.depth, self.key, self.attributes.toString(),  Tag.node_opening.value))
            for child in self.children:
                child.show()
            print("{}{} ".format('\t'*self.depth, Tag.node_closing.value))

    def toDSL(self):
        place = ""
        if self.key in [key.value for key in list(LeafKey)]:
            place = place + "{}{} {} \n".format('\t'*self.depth, self.key, self.attributes.toString())
        else:
            if self.attributes.isEmpty():
                place = place + "{}{} {} \n".format('\t'*self.depth, self.key, Tag.node_opening.value)
            else:
                place = place + "{}{} {} {} \n".format('\t'*self.depth, self.key, self.attributes.toString(),  Tag.node_opening.value)
            for child in self.children:
                place = place + child.toDSL()
            place = place + "{}{} \n".format('\t'*self.depth, Tag.node_closing.value)

        return place

    def to_row_col_DSL(self):
        place=""
        have_child = True if len(self.children) > 0 else False
        have_child = False if self.key == NodeKey.button.value else have_child
        
        if have_child:
            if (self.depth %2 == 0):
                if (self.depth != 0):
                    place = place + "{}{} {}\n".format('\t'*(self.depth-1), NodeKey.col.value, Tag.node_opening.value)
            else:
                place = place + "{}{} {}\n".format('\t'*(self.depth-1), NodeKey.row.value, Tag.node_opening.value)
            for child in self.children:
                place = place + child.to_row_col_DSL()
            if (self.depth != 0):
                place = place + "{}{}\n".format('\t'*(self.depth-1), Tag.node_closing.value)
        
        else:
            if (self.depth %2 == 0):
                place = place + "{}{}\n".format('\t'*(self.depth-1), NodeKey.col.value)
            else:
                place = place + "{}{}\n".format('\t'*(self.depth-1), NodeKey.row.value)

        return place


class Attribute:
    def __init__(self, font_color, context, bg_color=None):
        self.font_color = font_color
        self.bg_color = bg_color
        self.context = context

    def set_font_color(self, font_color):
        self.font_color = font_color

    def set_bg_color(self, bg_color):
        self.bg_color = bg_color

    def set_context(self, context):
        self.context = context

    def isEmpty(self) -> bool:
        if self.font_color:
            return False
        elif self.bg_color:
            return False
        elif self.context:
            return False
        else:
            return True

    def toString(self):
        return " ".join([ Tag.attr_opening.value, str(self.font_color), str(self.bg_color), str(self.context), Tag.attr_closing.value])
