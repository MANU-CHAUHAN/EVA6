"""
Python based main auto-grad engine for the entire system. Implements backprop on DAG.
"""
import numpy as np


class Tensor:
    def __init__(self, data, previous_op=None, parent_nodes=[]):
        if not isinstance(data, (int, float)):
            raise TypeError("Only `int` and `float` type of data accepted")

        self.data = data                    # the actual value
        # the operation responsible for creation of current node
        self.previous_op = previous_op
        self.parent_nodes = parent_nodes    # list of parent nodes creating this node
        self.grad = 0                       # derivative of output wrt self
        self.grad_wrt = {}                  # derivative wrt to parent nodes

    def __add__(self, other):
        '''for self + other'''
        other = other if isinstance(other, Tensor) else Tensor(other)
        output = Tensor(self.data + other.data, previous_op='+',
                        parent_nodes=[self, other])

        # derivative of output wrt to self
        # z = x + y, dz/dx = 1 + 0
        output.grad_wrt[self] = 1

        # derivative of output wrt other
        # z = x + y, dz/dy = 0+ 1
        output.grad_wrt[other] = 1
        return output

    def __radd__(self, other):
        ''' reverse of __add__, for other + self scenario'''
        return self.__add__(other)

    def __sub__(self, other):
        '''for self - other'''
        other = other if isinstance(other, Tensor) else Tensor(other)
        output = Tensor(data=(self.data - other.data),
                        previous_op='-', parent_nodes=[self, other])
        # derivative dz/dx
        output.grad_wrt[self] = 1
        # derivative dz/dy
        output.grad_wrt[other] = -1
        return output

    def __rsub__(self, other):
        '''for reverse sub, other - self scenario'''
        other = other if isinstance(other, Tensor) else Tensor(other)
        output = Tensor((other.data - self.data), '-', [self, other])
        # derivative of output wrt `self`, dz/dx = dy/dx - dx/dx
        output.grad_wrt[self] = -1
        # derivative of output wrt 'other', dz/dy = 1
        output.grad_wrt[other] = 1
        return output

    def __mul__(self, other):
        '''for self * other '''
        other = other if isinstance(other, Tensor) else Tensor(other)
        output = Tensor(self.data * other.data, '*', [self, other])
        # derivative dz/dx for x * y
        output.grad_wrt[self] = other.data
        # derivative dz/dy
        output.grad_wrt[other] = self.data
        return output

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, power):
        ''' for self ^ other '''
        assert isinstance(power, (int, float)
                          ), 'power must be either float or int'
        output = Tensor(self.data ** power, '^', [self])
        # derivative of output wrt self using the rule power * x ^ (power - 1)
        output.grad_wrt[self] = power * self.data ** (power - 1)
        return output

    def __truediv__(self, other):
        ''' for self/other scenario '''
        other = other if isinstance(other, Tensor) else Tensor(other)
        output = Tensor(self.data / other.data, '/', [self, other])
        # derivative if output wrt self
        output.grad_wrt[self] = 1 / other.data
        # derivative of output wrt other, see it as self * other ^ -1 => self * -1 * other ^ -2
        output.grad_wrt[other] = -1 * self.data * (other.data) ** -2

    def __rtruediv__(self, other):
        ''' for other/self scenario '''
        other = other if isinstance(
            other, Tensor) else Tensor(other)

        out = Tensor(other.data / self.data, '/', [self, other])

        # derivative of output wrt `self`, z = y/x, (here x -> self), dz/dx = y*(d[x^-1]/dx) = y*-1*x^-2
        out.grad_wrt[self] = -other.data * (self.data ** -2)

        # derivative of output wrt `other`, z = y/x, (here y -> other), dz/dy = (1/x)*(dy/dy) = 1/x
        out.grad_wrt[other] = 1 / self.data
        return out

    def __neg__(self):
        ''' for -self scenario '''
        return self.__mul__(-1)

    def relu(self):
        ''' ReLU (Rectified Linear Unit) is used as activation fucntion and allows only positive values to pass through it '''

        only_positives = max(0, self.data)  # only use values > 0

        out = Tensor(only_positives, 'ReLU', [self])

        # derivative of relu function will be 0 or 1
        out.grad_wrt[self] = int(self.data > 0)
        return out

    def sigmoid(self):
        ''' for using signoid activation function on node '''
        f = 1 / (1 + np.exp(-self.data))
        out = Tensor(f, 'sigmoid', [self])
        out.grad_wrt[self] = f * (1 - f)
        return out

    def __repr__(self):
        return f'Tensor(data={self.data:.2f}, grad={self.grad:.2f}), prev_op={self.prev_op})'

    def backward(self):
        ''' To calculate derivate of `outout` wrt to all nodes in DAG.

            The optimized approach for calculating derivatives with mapping of type R^n --> R^m , where n >> m, is reverse automatic differentiation, which utilizes the chain-rule (https://en.wikipedia.org/wiki/Automatic_differentiation#The_chain_rule,_forward_and_reverse_accumulation).
            Reverse-mode AD splits the task into 2 parts, namely, forward and reverse pass.

            This is exactly what is implemented in PyTorch, Tensorflow and other NN libraries.

            We move in reverse order from output towards input.

        '''
        
