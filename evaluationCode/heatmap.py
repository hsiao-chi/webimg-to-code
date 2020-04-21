import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set()
from general.util import read_file
import matplotlib.pyplot as plt

def show_heatmap(data, x_axis_labels, y_axis_labels, save_path=None, ratio=False):
    if ratio:
        sum_of_each_line = np.sum(data, axis=0)
        data = data / sum_of_each_line[:, None]
    print(data)
    ax = sns.heatmap(data, xticklabels=x_axis_labels, yticklabels=y_axis_labels, cmap="YlGnBu")
    plt.show()
    if save_path:
        plt.savefig(save_path+'-heatmap.png')

def compare_attr_class(gt_label_file, predit_label_file, y_axis_labels: list, x_axis_labels: list):
    # label 記得加"EOS"
    target_array = np.zeros((len(y_axis_labels), len(x_axis_labels)))
    gt_labels = read_file(gt_label_file, 'splitlines')
    predit_labels = read_file(predit_label_file, 'splitlines')
    gt_labels_array = [gt_label.split() for gt_label in gt_labels]
    for predit_label in predit_labels:
        predit_label = predit_label.split()
        for gt_label in gt_labels_array:
            if gt_label[0] == predit_label[0]:
                len_gt = len(gt_label[1:])
                len_predit = len(gt_label[1:])
                if len_gt > len_predit:
                    for i, gt in enumerate(gt_label[1:]):
                        try:
                            pred = predit_label[1+i]
                            target_array[y_axis_labels.index(gt), x_axis_labels.index(pred)] += 1
                        except ValueError or IndexError:
                            target_array[y_axis_labels.index(gt), x_axis_labels.index('EOS')] += 1
                elif len_gt < len_predit:
                    for i, pred in enumerate(predit_label[1:]):
                        try:
                            gt = gt_label[1+i]
                            target_array[y_axis_labels.index(gt), x_axis_labels.index(pred)] += 1
                        except ValueError or IndexError:
                            target_array[y_axis_labels.index('EOS'), x_axis_labels.index(pred)] += 1
                    pass
                else:
                    if len_gt == 0:
                        target_array[y_axis_labels.index('EOS'), x_axis_labels.index('EOS')] += 1
                        pass
                    for gt, pred in zip(gt_label[1:], predit_label[1:]):
                        target_array[y_axis_labels.index(gt), x_axis_labels.index(pred)] += 1
    return target_array

    
