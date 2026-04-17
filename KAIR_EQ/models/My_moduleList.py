from torch.nn import Module
from models.My_module import MyModule
from typing import List

class MyModuleList(MyModule):
    def __init__(self, modules: List[Module]):
        super(MyModuleList, self).__init__()
        
        self.modules = modules if modules is not None else []
        self.rot_num = 5


        for idx, module in enumerate(modules):
            self.add_module(f"module_{idx}", module)

    def forward(self, *input):

        for module in self.modules:
            input = module(*input)
        return input

    def rot_num_in(self, rot_num = 0):

        self.rot_num = rot_num
        for module in self.children():
            # print(module)
            if hasattr(module, 'rot_num'):
                # print(module)
                module.rot_num_in(rot_num)
        return self
    
    def append(self, module: Module):
        r"""Appends a given module to the end of the list.

        Args:
            module (nn.Module): module to append
        """
        self.add_module(str(len(self)), module)
        return self
    
    def __len__(self):
        return len(self._modules)
    
    def __iter__(self):
        return iter(self._modules.values())

    


