'''Binary Sum Tree structure

Created by Minhui Li on March 22, 2020
'''


from typing import Tuple


class BinarySumTree(object):
    '''Binary Sum Tree

    A binary tree. The parent node contains the sum of its two children.
    The leaf nodes contain data concerned.
    '''

    def __init__(
        self,
        node_num: int = 1
    ) -> None:

        if node_num < 1 or node_num % 2 == 0:
            raise ValueError('The number of nodes should be odd and at least 1.')

        self.tree = [0.] * node_num
        self.size = node_num

    def getParentInd(
        self,
        index: int
    ) -> int:
        '''Get the parent node index.

        Args:
        index
            The node index

        Returns:
            The index of its parent node 
        '''

        return (index - 1) // 2

    def getChildrenInd(
        self,
        index: int
    ) -> Tuple[int, int]:
        '''Get the indices of two children.

        Args:
        index
            The node index

        Returns:
            Indices of two children 
        '''

        left = 2 * index + 1
        right = left + 1
        return left, right

    def isLeaf(self, index:int) -> bool:
        '''Judge if the node is a leaf node.

        Args:
        index:
            The node index

        Returns:
            If the node is a leaf node
        '''

        if self.size // 2 <= index < self.size:
            return True
        else:
            return False

    def forward(
        self,
        number: float,
        index: int = 0
    ) -> int:
        '''Find a leaf node

        Args:
        number:
            A number provided
        index:
            The node index

        Returns:
            The leaf node index
        '''

        if number < 0 or number > self.root:
            raise ValueError('The number provided is invalid. Shoud be in range of [0, {}]'.format(self.root))
        if index < 0 or index >= self.size:
            raise ValueError('The index provided is invalid. Shoud be in range of [0, {})'.format(self.size))

        if self.isLeaf(index=index):
            return index
        left, right = self.getChildrenInd(index=index)
        if number <= self.tree[left]:
            return self.forward(number=number, index=left)
        else:
            return self.forward(number=number - self.tree[left], index=right)

    def backward(
        self,
        index: int,
        delta:float,
    ) -> None:
        '''Propagate changes in a node backwards.

        Args:
        index:
            The node index
        delta:
            Change amount
        '''

        self.tree[index] += delta

        if index > 0:
            parent = self.getParentInd(index=index)
            self.backward(index = parent, delta=delta)

    @property
    def root(self) -> float:
        return self.tree[0]

    def __len__(self) -> float:
        return self.size