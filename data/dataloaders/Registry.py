
class Registry(object):
    def __init__(self):
        self._module_dict = dict()

    def get(self, key, kwargs=None):
        return self._module_dict.get(key, None)(**kwargs)

    def register_module(self, module):
        if module.__name__ in self._module_dict:
            raise Exception

        self._module_dict[module.__name__] = module
        return module


REGISTRY_TYPE = Registry()
