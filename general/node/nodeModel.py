from .nodeEnum import NodeKey, Tag, LeafKey, Placeholder

class Node:
    def __init__(self, key, parent_node, attributes, depth=0):
        self.key = key
        self.parent = parent_node
        self.children = []
        self.attributes = attributes
        self.depth = depth
        self.content_holder = Placeholder.node.value

    def add_child(self, child):
        self.children.append(child)

    def show(self):
        if self.key in [key.value for key in list(LeafKey)]:
            print("{}{} {} ".format('\t'*self.depth,
                                    self.key, self.attributes.toString()))
        else:
            if self.attributes.isEmpty():
                print("{}{} {} ".format('\t'*self.depth,
                                        self.key, Tag.node_opening.value))
            else:
                print("{}{} {} {} ".format('\t'*self.depth, self.key,
                                           self.attributes.toString(),  Tag.node_opening.value))
            for child in self.children:
                child.show()
            print("{}{} ".format('\t'*self.depth, Tag.node_closing.value))

    def toDSL(self):
        place = ""
        if self.key in [key.value for key in list(LeafKey)]:
            place = place + \
                "{}{} {} \n".format('\t'*(self.depth-1),
                                    self.key, self.attributes.toString())
        else:
            if (self.depth != 0):
                if self.attributes.isEmpty():
                    place = place + \
                        "{}{} {} \n".format(
                            '\t'*(self.depth-1), self.key, Tag.node_opening.value)
                else:
                    place = place + "{}{} {} {} \n".format('\t'*(
                        self.depth-1), self.key, self.attributes.toString(),  Tag.node_opening.value)

            for child in self.children:
                place = place + child.toDSL()

            if (self.depth != 0):
                place = place + \
                    "{}{} \n".format('\t'*(self.depth-1),
                                     Tag.node_closing.value)

        return place

    def to_row_col_DSL(self):
        place = ""
        have_child = True if len(self.children) > 0 else False
        have_child = False if self.key == NodeKey.button.value else have_child

        if have_child:
            if (self.depth % 2 == 0):
                if (self.depth != 0):
                    place = place + \
                        "{}{} {}\n".format(
                            '\t'*(self.depth-1), NodeKey.col.value, Tag.node_opening.value)
            else:
                place = place + \
                    "{}{} {}\n".format(
                        '\t'*(self.depth-1), NodeKey.row.value, Tag.node_opening.value)
            for child in self.children:
                place = place + child.to_row_col_DSL()
            if (self.depth != 0):
                place = place + \
                    "{}{}\n".format('\t'*(self.depth-1),
                                    Tag.node_closing.value)

        else:
            if (self.depth % 2 == 0):
                place = place + \
                    "{}{}\n".format('\t'*(self.depth-1), NodeKey.col.value)
            else:
                place = place + \
                    "{}{}\n".format('\t'*(self.depth-1), NodeKey.row.value)

        return place

    def toHTML(self, mapping, file_name="data"):
        content = ""
        for child in self.children:
            content += child.toHTML(mapping)
        template = mapping[self._get_template_type()]
        template = self.attributes.render_attribute(template)
        template = self._set_col_bg_color(template)
        template = self._set_data_title(template, file_name)

        if len(self.children) != 0:
            template = template.replace(self.content_holder, content, 1)
        return template

    def _get_template_type(self):
        if self.key == LeafKey.text.value:
            if self.parent.key == NodeKey.button.value:
                return 'span_text'
            else:
                return self.key
        return self.key

    def _set_col_bg_color(self, template):
        if self.key == NodeKey.col.value:
            if NodeKey.row.value in [node.key for node in self.children]:
                template = template.replace(Placeholder.leaf_col.value, "")
            else:
                template = template.replace(
                    Placeholder.leaf_col.value, "leaf-col")
        return template

    def _set_data_title(self, template, file_name):
        if template.find(Placeholder.data_title.value) != -1:
            template = template.replace(
                Placeholder.data_title.value, file_name)
        return template


class Attribute:
    def __init__(self, font_color=None, bg_color=None, content=None,):
        self.font_color = font_color
        self.bg_color = bg_color
        self.content = content

    def set_font_color(self, font_color):
        self.font_color = font_color

    def set_bg_color(self, bg_color):
        self.bg_color = bg_color

    def set_content(self, content):
        self.content = content

    def isEmpty(self) -> bool:
        if self.font_color:
            return False
        elif self.bg_color:
            return False
        elif self.content:
            return False
        else:
            return True

    def toString(self):
        return " ".join([Tag.attr_opening.value, str(self.font_color), str(self.bg_color), str(self.content), Tag.attr_closing.value])

    def list_to_attribut(self, attrs):
        self.font_color = attrs[0]
        self.bg_color = attrs[1]
        self.content = attrs[2]

    def render_attribute(self, template) -> str:
        print(template.find(Placeholder.content.value))
        if self.isEmpty():
            pass
        else:
            if template.find(Placeholder.color.value) != -1:
                # print("colorrrr")
                template = template.replace(
                    Placeholder.color.value, "" if self.font_color == None else ("text-"+self.font_color))
            if template.find(Placeholder.bg_color.value) != -1:
                template = template.replace(
                    Placeholder.bg_color.value, "" if self.bg_color == None else ("bg-"+self.bg_color))
            if template.find(Placeholder.content.value) != -1:
                
                template = template.replace(
                    Placeholder.content.value, "" if self.content == None else self.content)
        return template
