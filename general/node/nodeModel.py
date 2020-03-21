from .nodeEnum import NodeKey, Tag, LeafKey, Placeholder, AttributeSet, Font_color, Bg_color

class Attribute:
    def __init__(self, activatedAttributes: list = [], enabledAttributes: list = []):
        #  [AttributeEnum]
        self.activatedAttributes = activatedAttributes
        # [True, False...]
        self.enabledAttributes = enabledAttributes
        self.attributeValue = [None]*len(self.activatedAttributes)

    def assign_value(self, index, value):
        self.attributeValue[index] = value

    # def to_string(self):
    #     return " ".join([Tag.attr_opening.value] + list(filter(lambda x: x !=None, self.attributeValue)) + [Tag.attr_closing.value])

    def to_string(self, with_context=True):
        if with_context:
            return " ".join([Tag.attr_opening.value] + list(filter(lambda x: x !=None, self.attributeValue)) + [Tag.attr_closing.value])
        else:
            attrValue=[None]
            try:
                context_idx = self.activatedAttributes.index(AttributeSet.content)
                if self.enabledAttributes[context_idx]:
                    attrValue = self.attributeValue[:context_idx]+self.attributeValue[context_idx+1:]
                else:
                    attrValue = self.attributeValue
            except ValueError:
                attrValue = self.attributeValue
            finally:
                return " ".join([Tag.attr_opening.value] + list(filter(lambda x: x !=None, attrValue)) + [Tag.attr_closing.value])

    def is_empty(self): 
        return False if True in self.enabledAttributes else True

    def list_to_attribut(self, attrs):
        self.attributeValue = attrs

    def render_attribute(self, template) -> str:
        if self.is_empty():
            pass
        else:
            for value, activated in zip(self.attributeValue, self.activatedAttributes):
                print(value, activated)
                if value:
                    placeholder, data = self._placeholder_mapping(
                        activated, value)
                    if template.find(placeholder) != -1:
                        template = template.replace(
                            placeholder, "" if value == None else data)
        return template

    def _placeholder_mapping(self, activated, value):
        if activated == AttributeSet.font_color:
            return Placeholder.color.value, value
        elif activated == AttributeSet.bg_color:
            return Placeholder.bg_color.value, value
        elif activated == AttributeSet.content:
            return Placeholder.content.value, value[1:-1]

class Node:
    def __init__(self, key, parent_node, attributes: Attribute, depth=0):
        self.key = key
        self.parent = parent_node
        self.children = []
        self.attributes = attributes
        self.depth = depth
        self.content_holder = Placeholder.node.value

    def add_child(self, child):
        self.children.append(child)

    def set_attributes(self, attributes):
        self.attributes = attributes

    def show(self, with_context=True):
        if self.key in [key.value for key in list(LeafKey)]:
            print("{}{} {} ".format('\t'*self.depth,
                                    self.key, self.attributes.to_string(with_context)))
        else:
            if self.attributes.is_empty():
                print("{}{} {} ".format('\t'*self.depth,
                                        self.key, Tag.node_opening.value))
            else:
                print("{}{} {} {} ".format('\t'*self.depth, self.key,
                                           self.attributes.to_string(with_context),  Tag.node_opening.value))
            for child in self.children:
                child.show()
            print("{}{} ".format('\t'*self.depth, Tag.node_closing.value))

    def toDSL(self, with_context=True):
        place = ""
        if self.key in [key.value for key in list(LeafKey)]:
            place = place + \
                "{}{} {} \n".format('\t'*(self.depth-1),
                                    self.key, self.attributes.to_string(with_context))
        else:
            if (self.depth != 0):
                if self.attributes.is_empty():
                    place = place + \
                        "{}{} {} \n".format(
                            '\t'*(self.depth-1), self.key, Tag.node_opening.value)
                else:
                    place = place + "{}{} {} {} \n".format('\t'*(
                        self.depth-1), self.key, self.attributes.to_string(with_context),  Tag.node_opening.value)

            for child in self.children:
                place = place + child.toDSL(with_context=with_context)

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
        # template = self._set_col_class(template)
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
        if self.key == NodeKey.col.value:
            if NodeKey.row.value in [node.key for node in self.children]:
                return 'col'
            else:
                return 'leaf_col'
        return self.key

    def _set_col_class(self, template):
        if self.key == NodeKey.col.value:
            if NodeKey.row.value in [node.key for node in self.children]:
                template = template.replace(Placeholder.col.value, "col-auto")
            else:
                template = template.replace(
                    Placeholder.col.value, "col leaf-col")
        return template

    def _set_data_title(self, template, file_name):
        if template.find(Placeholder.data_title.value) != -1:
            template = template.replace(
                Placeholder.data_title.value, file_name)
        return template


class Attribute_old:
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
        # print(template.find(Placeholder.content.value))
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



