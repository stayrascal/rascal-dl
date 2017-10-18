#! /usr/bin/env python
# coding:utf-8


from collections import namedtuple
from sys import stdout

Node = namedtuple('Node', 'data, left, right')


def preorder(node):
    '''
    pre-order, NLR
    :param node:
    :return:
    '''
    if node is not None:
        print(node.data)
        preorder(node.left)
        preorder(node.right)


def inorder(node):
    '''
    in-order LNR
    :param node:
    :return:
    '''
    if node is not None:
        inorder(node.left)
        print(node.data)
        inorder(node.right)


def postorder(node):
    '''
    post-order LRN
    :param node:
    :return:
    '''
    if node is not None:
        postorder(node.left)
        postorder(node.right)
        print(node.data)


def levelorder(node, more=None):
    if node is not None:
        if more is None:
            more = []
        more.extend([node.left, node.right])
        print(node.data)
    if more:
        levelorder(more[0], more[1:])


if __name__ == '__main__':
    tree = Node(1,
                Node(2,
                     Node(4,
                          Node(7, None, None),
                          None),
                     Node(5, None, None)),
                Node(3,
                     Node(6,
                          Node(8, None, None),
                          Node(9, None, None)),
                     None))

    print('levelorder:')
    levelorder(tree)
