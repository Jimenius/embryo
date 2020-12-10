'''Registry

Derived from fvcore project https://github.com/facebookresearch/fvcore
'''


from typing import Any, Dict, Optional


class Registry:
    '''The registry that provides name -> object mapping
    
    Support users' custom modules.
    '''

    def __init__(
        self,
        name: str
    ) -> None:
        self._name: str = name
        self._obj_map: Dict[str, object] = {}

    def _do_register(
        self,
        name: str,
        obj: Any,
    ) -> None:
        assert (
            name not in self._obj_map
        ), 'An object named \'{}\' was already registered in \'{}\' registry!'.format(
            name, self._name
        )
        self._obj_map[name] = obj

    def register(
        self,
        obj: Optional[object] = None
    ) -> Optional[object]:
        '''Register the given object under the the name obj.__name__.

        Can be used as either a decorator or not.
        '''
        
        if obj is None:
            # used as a decorator
            def deco(func_or_class: object) -> object:
                name = func_or_class.__name__  # pyre-ignore
                self._do_register(name, func_or_class)
                return func_or_class

            return deco

        # used as a function call
        name = obj.__name__
        self._do_register(name, obj)

    def get(
        self,
        name: str
    ) -> object:
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(
                'No object named \'{}\' found in \'{}\' registry!'.format(name, self._name)
            )
        return ret

    def __contains__(self, name: str) -> bool:
        return name in self._obj_map
