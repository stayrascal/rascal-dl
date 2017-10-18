#! /usr/bin/env python
# coding:utf-8

class Node:
    def __init__(self, data):
        self.left = None
        self.right = None
        self.data = data

    def insert(self, data):
        if data < self.data:
            if self.left is None:
                self.left = Node(data)
            else:
                self.left.insert(data)
        elif data > self.data:
            if self.right is None:
                self.right = Node(data)
            else:
                self.right.insert(data)

    def lookup(self, data, parent=None):
        if data < self.data:
            if self.left is None:
                return None, None
            return self.left.lookup(data, self)
        elif data > self.data:
            if self.right is None:
                return None, None
            return self.right.lookup(data, self)
        else:
            return self, parent

    def delete(self, data):
        node, parent = self.lookup(data)
        if node is not None:
            children_count = node.children_count()
            if children_count == 0:
                # delete this node if this node don't have children
                if parent.left is Node:
                    parent.left = None
                else:
                    parent.right = None
                del Node
            if children_count == 1:
                # pop up child node and replace this node
                n = node.left if node.left else node.right
                if parent:
                    if parent.left is Node:
                        parent.left = n
                    else:
                        parent.right = n
                del node
            else:
                parent = node
                successor = node.right
                while successor.left:
                    parent = successor
                    successor = successor.left
                node.data = successor.data
                if parent.left == successor:
                    parent.left = successor.right
                else:
                    parent.right = successor.right

    def compare_trees(self, node):
        if node is None:
            return False
        if self.data != node.data:
            return False
        res = True
        if self.left is None and node.left:
            return False
        else:
            res = self.left.compare_trees(node.left)
        if res is False:
            return False
        if self.right is None and node.right:
            return False
        else:
            res = self.right.compare_trees(node.right)
        return res

    def print_tree(self):
        if self.left:
            self.left.print_tree()
        print(self.data)
        if self.right:
            self.right.print_tree()

    def tree_data(self):
        stack = []
        node = self
        while stack or node:
            if node:
                stack.append(node)
                node = node.left
            else:
                node = stack.pop()
                yield node.data
                node = node.right

    def children_count(self):
        cnt = 0
        if self.left:
            cnt += 1
        if self.right:
            cnt += 1
        return cnt


class CNode:
    left, right, data = None, None, 0

    def __init__(self, data):
        self.left = None
        self.right = None
        self.data = data


class CBOrdTree:
    def __init__(self):
        self.root = None

    def addNode(self, data):
        return CNode(data)

    def insert(self, root, data):
        if root == None:
            return self.addNode(data)
        else:
            if data <= root.data:
                root.left = self.insert(root.left, data)
            else:
                root.right = self.insert(root.right, data)
            return root

    def lookup(self, root, target):
        if root == None:
            return 0
        else:
            if target == root.data:
                return 1
            else:
                if target < root.data:
                    return self.lookup(root.left, target)
                else:
                    return self.lookup(root.right, target)

    def minValue(self, root):
        while (root.left != None):
            root = root.left
        return root.data

    def maxDepth(self, root):
        if root == None:
            return 0
        else:
            ldepth = self.maxDepth(self.left)
            rdepth = self.maxDepth(self.right)
            return max(ldepth, rdepth) + 1

    def size(self, root):
        if root == None:
            return 0
        else:
            return self.size(root.left) + self.size(root.right) + 1

    def printTree(self, root):
        if root == None:
            pass
        else:
            self.printTree(root.left)
            print(root.data)
            self.printTree(root.right)

    def printRevTree(self, root):
        if root == None:
            pass
        else:
            self.printRevTree(root.right)
            print(root.data)
            self.printRevTree(root.left)


if __name__ == '__main__':
    BTree = CBOrdTree()
    root = BTree.addNode(0)
    for i in range(0, 5):
        data = int(input('insert the node value nr {}:'.format(i)))
        BTree.insert(root, data)

    print()
    BTree.printTree(root)
    print()
    BTree.printRevTree(root)
    print()

    data = int(input('insert a value to find: '))
    if BTree.lookup(root, data):
        print('found')
    else:
        print('not found')

    print(BTree.minValue(root))
    print(BTree.maxDepth(root))
    print(BTree.size(root))
