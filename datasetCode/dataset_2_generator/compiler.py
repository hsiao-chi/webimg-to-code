import json
from general.node.nodeModel import Node, Attribute
from general.node.nodeEnum import RootKey, NodeKey, LeafKey, Font_color, Bg_color, Tag
from datasetCode.dataset_2_generator.generateRule import getRule




class Compiler:
    def __init__(self, mapping_file_path, rule=1, node_tree=Node(RootKey.body.value, None, Attribute())):
        with open(mapping_file_path) as data_file:
            self.html_mapping = json.load(data_file)
        self.rule = getRule(rule)
        self.activatedAttributes = self.rule["attributes"]
        self.node_tree = node_tree
        self.node_opening_tag = Tag.node_opening.value
        self.node_closing_tag = Tag.node_closing.value
        self.attr_opening_tag = Tag.attr_opening.value
        self.attr_closing_tag = Tag.attr_closing.value

    def dsl_to_node_tree(self, dsl_file_path) -> Node:
        self.node_tree = Node(RootKey.body.value, None, Attribute(self.activatedAttributes, self.rule[RootKey.body.value]["attributes"]))
        depth = 1
        dsl = []
        with open(dsl_file_path, 'r') as dsl_file:
            dsl = dsl_file.read().split()
        current_parent_node = self.node_tree
        current_node = self.node_tree
        in_attr_flag = False
        attr = []
        for token in dsl:
            # print(token)
            if token == self.node_opening_tag:
                current_parent_node = current_node
                depth +=1
            elif token == self.node_closing_tag:
                depth -=1
                current_parent_node = current_parent_node.parent
            elif token == self.attr_opening_tag:
                in_attr_flag = True
                attr = []
            elif token == self.attr_closing_tag:
                in_attr_flag = False
                current_node.attributes.list_to_attribut(self._reconstruct_attr_block(attr))
            else:
                if in_attr_flag:
                    attr.append(token)
                else:
                    current_node = Node(token, current_parent_node, Attribute(self.activatedAttributes, self.rule[token]["attributes"]), depth)
                    current_parent_node.add_child(current_node)
            # print("now Deep: ", depth, "  current_parent: ", current_parent_node.key, "  current: ", current_node, "now_attr: ", attr)
        return self.node_tree
            
    def _reconstruct_attr_block(self, attr) -> list:
        temp=[]
        temp_text=""
        isText = False
        for token in attr:
            if (not isText) and ('\"' in token):
                isText = True
                temp_text = token
            elif isText and ('\"' in token):
                isText = False
                temp_text+= " "+token
                temp.append(temp_text[1:-1])
            elif isText:
                temp_text+= " "+token
            
            elif token == "None":
                temp.append(None)
            else:
                temp.append(token)

        return temp

           

    def node_tree_to_dsl(self, output_file_path, row_col_only=False) -> str:
        dsl = self.node_tree.to_row_col_DSL() if row_col_only else self.node_tree.toDSL()
        with open(output_file_path, 'w+') as dsl_file:
            dsl_file.write(dsl)
        return dsl

    def node_tree_to_html(self, output_file_path, file_name):
        html = self.node_tree.toHTML(self.html_mapping, file_name)
        with open(output_file_path, 'w+') as html_file:
            html_file.write(html)
        return html

   