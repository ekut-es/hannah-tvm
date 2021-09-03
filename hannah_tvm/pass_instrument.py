from typing import Union, Sequence
import tvm

@tvm.instrument.pass_instrument
class PrintIR:
    """Print the name of the pass, the IR, only before passes execute."""
    def __init__(self, names : Union[str, Sequence[str]] = []):
        self.names = names

    def run_after_pass(self, mod, info):
        if self.names == "all" or str(info.name) in self.names or str(info.name) == self.names: 
            print("Mod after pass:", info.name)
            print(mod)