'''Ion: Data structure

Created by Minhui Li on November 18, 2020
'''


from copy import deepcopy
from numbers import Number
from typing import Dict, Iterator, Iterable, List, Sequence
from typing import Any, Union, Optional

import numpy as np
import torch


def _is_number(
    value: Any
) -> bool:

    return isinstance(value, (Number, np.number))


class MetaIon(type):
    '''Metaclass of Ion to store meta-data of Ion
    '''

    def __init__(self, name, bases, namespace, **kwargs):

        self.reserved_keys = (
            'items',
            'is_empty',
            'to',
            'shape',
            'get',
            'update',
        )


class Ion(metaclass=MetaIon):
    '''The internal data structure in embryo.

    Ion represents biological signal, is intrinsically a (recursive) dictionary
    of object that can be either numpy array, torch tensor, or ion themself.
    '''

    def __init__(
        self,
        data_dict: Optional[
            Union[
                Dict[str, Any],
                'Ion',
                Sequence[Union[Dict[str, Any], 'Ion']],
                np.ndarray,
            ]
        ] = None,
        copy: bool = False,
        **kwargs: Any
    ) -> None:
        '''Initialization method

        Args:
            data_dict: dictionary to convert
            copy: whether return a new copy of the data dict

        Raises:
            TypeError, when a key is not a string
            AttributeError, when a key is a defined attribute or method
        '''

        if copy:
            data_dict = deepcopy(data_dict)
        if data_dict is not None:
            if isinstance(data_dict, (dict, Ion)):
                for k, v in data_dict.items():
                    if not isinstance(k, str):
                        raise TypeError(
                            'Keys should all be string, ',
                            'but got {}'.format(type(k))
                        )
                    # Avoid key collision
                    # with defined attributes or methods.
                    # Reserved keys are defined in the metaclass.
                    if k in Ion.reserved_keys:
                        raise AttributeError(
                            'Please rename your key \'{}\''.format(k),
                            ' as it is reserved.'
                        )
                    setattr(self, k, v)
            # elif _is_batch_set(batch_dict):
            #     self.stack_(batch_dict)
        if len(kwargs) > 0:
            self.__init__(kwargs, copy=copy)

    def __contains__(
        self,
        key: str
    ) -> bool:
        '''Support \'in\' operator in conditional statement.
        '''

        if isinstance(key, str):
            return key in self.__dict__
        else:
            raise TypeError(
                'Support only string membership check ',
                'but got type {}.'.format(type(key))
            )

    def __getitem__(
        self,
        index: Union[str, slice, int, np.integer, np.ndarray, List[int]]
    ) -> Any:
        '''Support self[index].
        '''

        # As dict indexing
        if isinstance(index, str):
            return getattr(self, index)
        
        # As sequence indexing
        data_items = self.items
        if len(data_items) > 0:
            # Initialize a new Ion ready for return.
            new_ion = Ion()
            for k, v in data_items:
                if isinstance(v, Ion) and v.is_empty():
                    setattr(new_ion, k, Ion())
                else:
                    setattr(new_ion, k, v[index])
            return new_ion
        else:
            raise IndexError('Access an empty object.')

    def __setitem__(
        self,
        index: Union[str, slice, int, np.integer, np.ndarray, List[int]],
        value: Any
    ) -> None:
        '''Support value assignment to self[index]
        '''

        # Dict assignment
        if isinstance(index, str):
            setattr(self, index, value)
            return

        if isinstance(value, Ion):
            for k in self.__dict__.keys():
                if k in value:
                    self.__dict__[k][index] = value[k]
        else:
            raise ValueError(
                'Only support value assignment of Ion object, '
                'but got type {}.'.format(type(value))
            )

    def __iter__(self) -> Iterator:
        '''Support iterating operations

        Iterate over attributes in the object.
        '''

        for k in self.__dict__.keys():
            yield k

    def __iadd__(
        self,
        other: Union['Ion', Dict[str, Any], Number, np.number]
    ) -> 'Ion':
        '''Algebraic addition with another number or Ion instance in-place
        '''

        if _is_number(other):
            for k, v in self.items:
                if not (isinstance(v, Ion) and v.is_empty()):
                    self.__dict__[k] += other
        elif isinstance(other, Ion):
            for k, v in self.items:
                # Corresponding value in <other>, None if missing
                other_v = other.__dict__.get(k, None)
                if other_v and not (isinstance(v, Ion) and v.is_empty()):
                    self.__dict__[k] += other_v
        else:
            raise TypeError(
                'Only addition by Ion object or number is supported, '
                'but got type {}.'.format(type(other))
            )
        
        return self

    def __add__(
        self,
        other: Union['Ion', Dict[str, Any], Number, np.number]
    ) -> 'Ion':
        '''Algebraic addition with another number or Ion instance out-of-place
        '''

        return deepcopy(self).__iadd__(other)

    def __imul__(
        self,
        other: Union[Number, np.number]
    ) -> 'Ion':
        '''Algebraic multiplication with a scalar value in-place
        '''

        if not _is_number(other):
            raise TypeError(
                'Only multiplication by a scalar number is supported, '
                'but got type {}.'.format(type(other))
            )

        for k, v in self.items:
            if not (isinstance(v, Ion) and v.is_empty()):
                self.__dict__[k] *= other

        return self

    def __mul__(
        self,
        other: Union[Number, np.number]
    ) -> 'Ion':
        '''Algebraic multiplication with a scalar value out-of-place
        '''

        return deepcopy(self).__imul__(other)

    def __itruediv__(
        self,
        other: Union[Number, np.number]
    ) -> 'Ion':
        '''Algebraic multiplication with a scalar value in-place
        '''

        if not _is_number(other):
            raise TypeError(
                'Only division by a scalar number is supported, '
                'but got type {}.'.format(type(other))
            )

        for k, v in self.items:
            if not (isinstance(v, Ion) and v.is_empty()):
                self.__dict__[k] /= other

        return self

    def __truediv__(
        self,
        other: Union[Number, np.number]
    ) -> 'Ion':
        '''Algebraic division with a scalar value out-of-place
        '''

        return deepcopy(self).__itruediv__(other)

    @property
    def items(self) -> 'dict_items':

        return self.__dict__.items()

    def is_empty(
        self,
    ) -> bool:
        '''Check if is empty

        Returns:
            ...
        '''

        return len(self.__dict__) == 0

    def to(
        self,
        ctype: str,
        dtype: Optional[Union[torch.dtype, np.dtype]] = None,
        device: Optional[Union[str, int, torch.device]] = None,
    ) -> None:
        '''Convert internal data to specific type

        Args:
            ctype: class type to be converted, 'torch' or 'numpy'
            dtype: numpy data type or torch data type
            device: torch device, ignored if to numpy

        Raises:
            ValueError, when unsupported ctype is provided.
        '''

        for k, v in self.items:
            if isinstance(v, Ion):
                v.to(ctype=ctype, dtype=dtype, device=device)
            else:
                if ctype == 'torch':
                    # If v is a numpy array, convert it to torch tensor first.
                    if isinstance(v, np.ndarray):
                        v = torch.from_numpy(v)
                    # Convert device and dtype attribute.
                    # Ignore values other than torch tensor.
                    if isinstance(v, torch.Tensor):
                        v = v.to(device=device, dtype=dtype)
                elif ctype == 'numpy':
                    # If v is a torch tensor, convert it to numpy array first.
                    if isinstance(v, torch.Tensor):
                        v = v.detach().to(device='cpu').numpy()
                    # Convert dtype, ignore values other than numpy array.
                    if isinstance(v, np.ndarray):
                        setattr(self, k, v.astype(dtype))
                else:
                    raise ValueError(
                        'Only class type torch and numpy are supported, ',
                        'but got {}.'.format(ctype)
                    )
        
    @property
    def shape(self) -> Dict[str, Any]:

        shape_dict = {}
        for k, v in self.items:
            if hasattr(v, 'shape'):
                shape_dict[k] = v.shape
        return shape_dict

    def get(
        self,
        key: str,
        default: Optional[Any] = None
    ) -> Any:
        '''Get value from key with default safe guard.
        '''

        return self.__dict__.get(key, default=default)

    def update(
        self,
        other: Optional[Union[Dict[str, Any], 'Ion']] = None,
        **kwargs
    ) -> None:
        '''Update ion in-place either with another dict/ion or with attributes.

        Args:
            other: update data source
        
        Raises:
            TypeError, for unsupported update data type.
        '''
        
        if other:
            if isinstance(other, Ion):
                update_items = other.items
            elif isinstance(other, dict):
                update_items = other.items()
            else:
                raise TypeError(
                    'Unsupported update data type.'
                )
            for k, v in update_items:
                setattr(self, k, v)
        if kwargs:
            self.update(kwargs)


def extend_space(
    value: Any,
    extend_size: int = 1,
) -> Any:
    '''Extend value-like space.

    Args:
        value: A
        extend_size:
    '''

    if isinstance(value, (Number, np.number)):
        return np.zeros(shape=extend_size, dtype=type(value))
    elif isinstance(value, np.ndarray):
        shape = (extend_size, *value.shape)
        return np.zeros(shape=shape, dtype=value.dtype)
    elif isinstance(value, torch.Tensor):
        shape = (extend_size, *value.size())
        return torch.zeros(size=shape, device=value.device, dtype=value.dtype)
    elif isinstance(value, Ion):
        new_ion = Ion()
        for k, v in value.items:
            new_ion[k] = extend_space(value=v)
    else:
        return np.full(shape=extend_size, fill_value=None, dtype=np.object)


if __name__ == '__main__':
    i2 = Ion()
    i1 = Ion(a=np.zeros((2,2)))
    print(i1.__dict__)
    print('Should be {\'a\':...(all zeors)}')
    i1 += 1
    i1 *= 2
    print(i1.__dict__)
    print('Should be {\'a\':...(all 2s)}')
    print(i1.is_empty(), 'Should be False')
    print(i2.is_empty(), 'Should be True')
    print('a' in i1, 'Should be True')