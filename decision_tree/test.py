
class TreeNode:
    # can not init node like this:'child=[]',fuck!
    def __init__(self, child):
        self.child = child


def main():
    node1=TreeNode(child=[])
    node1.child.append(TreeNode(child=[]))

    node2=TreeNode(child=[])
    node2.child.append(TreeNode(child=[]))

    print(id(node1.child))
    print(id(node2.child))

    # 两个地址一样

    print(node1.child)
    print(node2.child)


if __name__ == "__main__":
    main()