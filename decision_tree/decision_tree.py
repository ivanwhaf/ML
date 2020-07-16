import copy
import operator
from math import log

class DecisionTree():
    def __init__(self):
        pass
    
def calc_shannon_ent(dataset: list):
    """
    calculate the shannon entropy of a dataset
    H=-∑(n,i=1) p(xi)*log2p(xi)
    """
    num = len(dataset)

    label_counts = {}  # every class label's count
    for data in dataset:
        label = data[-1]
        if label not in label_counts.keys():
            label_counts[label] = 0
        label_counts[label] += 1

    shannon_ent = 0.0

    for label in label_counts:
        prob = float(label_counts[label])/num
        # l(xi)=-log2p(xi)
        shannon_ent -= prob*log(prob, 2)

    return shannon_ent


def split_dataset(dataset: list, axis, value):
    ret_dataset = []
    for data in dataset:
        if data[axis] == value:
            new_data = data[:axis]
            new_data.extend(data[axis+1:])
            ret_dataset.append(new_data)
    return ret_dataset


def choose_best_feature_to_split(dataset: list) -> int:
    """
    return best feature's index
    """
    num_features = len(dataset[0])-1

    base_entropy = calc_shannon_ent(dataset)
    best_info_gain, best_feature = 0.0, -1

    for i in range(num_features):
        feature_lst = [data[i] for data in dataset]
        unique_values = set(feature_lst)

        new_entropy = 0.0

        for value in unique_values:
            sub_dataset = split_dataset(dataset, i, value)
            prob = len(sub_dataset)/float(len(dataset))
            new_entropy += prob*calc_shannon_ent(sub_dataset)

        # info gain is entropy's decrease or decrease of data disorder
        info_gain = base_entropy-new_entropy

        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i

    return best_feature


def majority_count(class_lst):
    class_count = {}
    for _class in class_lst:
        if _class not in class_count.keys():
            class_lst[_class] = 0
        class_count[_class] += 1

    # sort the class_count dict by it's value->every class's count
    sorted_class_count = sorted(
        class_count.iteritems(), key=operator.itemgetter(1), reverse=True)

    return sorted_class_count[0][0]


class TreeNode:
    # can not init node like this:'child=[]',fuck!
    def __init__(self, child=None, feature_index=None, feature_label=None, class_label=None):
        # only leaf nodes have
        self.class_label = class_label

        # only middle nodes have
        self.child = child
        self.feature_index = feature_index
        self.feature_label = feature_label


def create_decision_tree(dataset: list, feature_labels: list, feature_value=None):
    class_lst = [data[-1] for data in dataset]

    # all the classes are same
    if class_lst.count(class_lst[0]) == len(class_lst):
        #print('leaf node')
        return TreeNode(class_label=class_lst[0])

    # have traversed all features
    if len(dataset[0]) == 1:
        return TreeNode(class_label=majority_count(class_lst))

    best_feature_index = choose_best_feature_to_split(dataset)
    best_feature_label = feature_labels[best_feature_index]
    #print('best feature label:'+best_feature_label)

    tree = TreeNode(child={}, feature_index=best_feature_index,
                    feature_label=best_feature_label)

    del(feature_labels[best_feature_index])

    feature_lst = [data[best_feature_index] for data in dataset]
    unique_values = set(feature_lst)

    for value in unique_values:
        sub_feature_labels = feature_labels[:]
        child = create_decision_tree(split_dataset(
            dataset, best_feature_index, value), sub_feature_labels, feature_value=value)

        tree.child[value] = child

    return tree


def classify(tree: TreeNode, feature_labels, pre_data):
    if not tree.class_label:
        feature = pre_data[tree.feature_index]
        child_tree = tree.child[feature]

        # print(tree.feature_index)
        del(feature_labels[tree.feature_index])

        reduced_pre_data = pre_data[:tree.feature_index]
        reduced_pre_data.extend(pre_data[tree.feature_index+1:])
        pre_data = reduced_pre_data
        # print(pre_data)

        ret = classify(child_tree, feature_labels, pre_data)
    else:
        ret = tree.class_label

    return ret


def main():
    dataset = [[1, 1, 'yes'], [1, 1, 'yes'], [
        1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    feature_labels = ['no surfacing?', 'flippers?']

    tree = create_decision_tree(dataset, copy.deepcopy(
        feature_labels), feature_value=None)

    pre_data = [1, 1, 'no']
    print(classify(tree, feature_labels, pre_data))


if __name__ == "__main__":
    main()
