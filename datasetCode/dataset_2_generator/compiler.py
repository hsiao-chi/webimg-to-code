import json
from general.node.nodeModel import Node, Attribute
from general.node.nodeEnum import RootKey, NodeKey, LeafKey, Font_color, Bg_color, Tag, AttributeSet
from datasetCode.dataset_2_generator.generateRule import getRule
from datasetCode.dataset_2_generator.util import get_random_text




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
                current_node.attributes.list_to_attribut(self._reconstruct_attr_block(current_node.key, attr))
            else:
                if in_attr_flag:
                    attr.append(token)
                else:
                    current_node = Node(token, current_parent_node, Attribute(self.activatedAttributes, self.rule[token]["attributes"]), depth)
                    current_parent_node.add_child(current_node)
        return self.node_tree
            
    def _reconstruct_attr_block(self, node, attr) -> list:
        temp=[None]*len(self.activatedAttributes)
        try:
            context_idx = self.activatedAttributes.index(AttributeSet.content)
        except :
            context_idx = -1
        temp_text=""
        isText = False
        for token in attr:
            if (not isText) and ('\"' in token):
                isText = True
                temp_text = token
            elif isText and ('\"' in token):
                isText = False
                temp_text+= " "+token
                temp[context_idx] = temp_text[1:-1]
            elif isText:
                temp_text+= " "+token
            # elif token == "None":
            #     temp.append(None)
            else:
                for attrIdx, activatedAttr in enumerate(self.activatedAttributes):
                    try:
                        attr_set = self.rule[node]["attributes_set"][activatedAttr.value]
                        if token in [attr.value for attr in attr_set]:
                            temp[attrIdx] = token
                    except KeyError:
                        pass
        if len(temp_text) > 0 and isText:
            isText=False
            temp[context_idx] = temp_text[1:-1]

        if  context_idx >= 0 and self.rule[node]["attributes"][context_idx] and temp[context_idx] == None:
            if node == LeafKey.button.value:
                temp[context_idx] =  get_random_text(10, 1) 
            elif node == LeafKey.title.value:
                temp[context_idx] =  get_random_text(5, 0) 
            else:
                temp[context_idx] =  get_random_text(30, 5, False) 
        return temp

           

    def node_tree_to_dsl(self, output_file_path, row_col_only=False, with_context=True) -> str:
        dsl = self.node_tree.to_row_col_DSL() if row_col_only else self.node_tree.toDSL(with_context=with_context)
        with open(output_file_path, 'w+') as dsl_file:
            dsl_file.write(dsl)
        return dsl

    def node_tree_to_html(self, output_file_path, file_name):
        html = self.node_tree.toHTML(self.html_mapping, file_name)
        with open(output_file_path, 'w+') as html_file:
            html_file.write(html)
        return html

    

   