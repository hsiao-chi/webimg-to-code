from config import NodeKey, Tag


class Node:
    def __init__(self, key, parent_node, attributes):
        self.key = key
        self.parent = parent_node
        self.children = []
        self.attributes = attributes
        # self.content_holder = content_holder

    def add_child(self, child):
        self.children.append(child)

    def show(self):
        if self.attributes.isEmpty():
            print("{} {} ".format(self.key, Tag.node_opening.value))
        else:
            print("{} {} {} ".format(self.key, self.attributes.toString(),  Tag.node_opening.value))
        for child in self.children:
            child.show()
        print(" {} ".format(Tag.node_closing.value))

    def toDSL(self, place=""):
        if self.attributes.isEmpty():
            place = place + "{} {} \n".format(self.key, Tag.node_opening.value)
        else:
            place = place + "{} {} {} \n".format(self.key, self.attributes.toString(),  Tag.node_opening.value)
        for child in self.children:
            child.toDSL(place)
        place = place + "{} \n".format(Tag.node_closing.value)


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
