# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 20:10:31 2021

@author: Tigran Boynagryan
"""

class Matrix:
    def __init__(self, *args, **kwargs):
        """
        Takes 2 keyword arguments: filename or list. If filename is given
        read the matrix from file. Else, read it directly from list.
        """
        if 'filename' in kwargs:
            self.read_from_file(kwargs['filename'])
        elif 'list' in kwargs:
            self.read_as_list(kwargs['list'])

    def read_as_list(self, matrix_list):
        if len(matrix_list) == 0:
            self._matrix = []
            self._columns = 0
            self._rows = 0
            return

        columns_count_0 = len(matrix_list[0])
        if not all(len(row) == columns_count_0 for row in matrix_list):
            raise ValueError('Got incorrect matrix')

        self._matrix = matrix_list
        self._rows = len(self._matrix)
        self._columns = len(self._matrix[0])

    def read_from_file(self, filename):
        with open(filename, 'r') as f:
            matrix_list = f.readlines()
        matrix_list = list(map(lambda s: list(map(float, s[:-1].split(' '))), matrix_list))
        self.read_as_list(matrix_list)

    def __str__(self):
        s = '---------MATRIX---------\n'
        s += '\n'.join(str(row) for row in self._matrix)
        s += '\n'
        s += f'colums = {self.shape[0]}\nrows = {self.shape[1]}'
        s += '\n------------------------\n'
        return s

    def write_to_file(self, filename):
        """
        Write the matrix to the given filename.
        TODO: implement
        """
        with open(filename, 'w') as f:
            f.write('\n'.join(' '.join(list(map(str,i))) + '\n' for i in self._matrix))
    
    def transpose(self):
        """
        Transpose the matrix.
        TODO: implement
        """
        return [[self._matrix[i][j] for i in range(self._rows)] for j in range(self._columns)]

    @property
    def shape(self):
        return self._columns, self._rows

    def __len__(self):
        return len(self._matrix)

    def __getitem__(self, item):
        return self._matrix[item]

    def __setitem__(self, item, value):
        self._matrix[item] = value

    def __add__(self, other):
        """
        The + operator. Sum two matrices.
        TODO: implement
        """
        if isinstance(other, Matrix):
            if self._rows != other._rows or self._columns != other._columns:
                raise ValueError('Invalid dims')
            return Matrix(list=[[self._matrix[i][j]+other._matrix[i][j] for j in range(self._columns)] for i in range(self._rows)])
        raise NotImplementedError()
                        

    def __mul__(self, other):
        """
        The * operator. Element-wise matrix multiplication.
        Columns and rows sizes of two matrices should be the same.
        If other is not a matrix (int, float) multiply all elements of the matrix to other.
        TODO: implement
        """

        if type(other) in (float, int):
            return [[self._matrix[i][j]*other for j in range(self._columns)] for i in range(self._rows)]
        elif isinstance(other, Matrix):
            if self._columns != other._rows:
                raise ValueError('Invalid dims')
            return [[sum([self._matrix[i][k]*other._matrix[k][j] for k in range(self._columns)]) for j in range(other._columns)] for i in range(self._rows)]
        raise NotImplementedError()

    def __matmul__(self, other):
        """
        The @ operator. Mathematical matrix multiplication.
        The number of columns in the first matrix must be equal to the number of rows in the second matrix.
        TODO: implement
        """
        if isinstance(other, Matrix):
            if self._columns != other._rows:
                raise ValueError('Invalid dims')
            return [[self._matrix[i][j]*other._matrix[j][i] for j in range(self._columns)] for i in range(self._rows)]
        raise NotImplementedError()
        

    @property
    def trace(self):
        """
        Find the trace of the matrix.
        TODO: implement
        """
        if self._columns != self._rows:
            raise ValueError('Not square matrix')
        return sum([self._matrix[i][i] for i in range(self._rows)])

    @property
    def determinant(self):
        """
        Check if the matrix is square, find the determinant.
        TODO: implement
        """
        if self._columns != self._rows:
            raise ValueError('Not square matrix')
        if self._rows == 2:
            return self._matrix[0][0]*self._matrix[1][1] - self._matrix[1][0]*self._matrix[0][1]
        sum = 0
        for j in range(self._rows):
            tmp = Matrix(list=[[self._matrix[x][y] for y in range(self._rows) if y != j] for x in range(1, self._rows)])
            print(tmp)
            sum += (-1)**(j)*Matrix(list=tmp).determinant
        return sum
                

test_mat = Matrix(list=[[1, 2, 3], [4, 5, 6]])
test_mat2 = Matrix(list=[[1, 1], [2, 2], [1, 1]])
test_mat3 = Matrix(list=[[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(test_mat.transpose())
print(test_mat * test_mat2)
print(test_mat @ test_mat2)
print(test_mat3.trace, Matrix(list=test_mat2*test_mat).trace)
print(test_mat3.determinant)
