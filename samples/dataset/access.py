from pprint import pprint

from hannah_tvm.dataset import DatasetFull

dataset = DatasetFull()

for network_result in dataset.network_results():
    print("=========================")
    print("Model:", network_result.model)
    print("Board:", network_result.board)
    print("Target:", network_result.target)
    print("Tuner:", network_result.tuner)
    print()
    print("Relay Graph:")
    relay = network_result.relay
    print(relay)
    print()
    print("TIR Primfuncs:")
    tir_funcs = network_result.tir
    if tir_funcs is not None:
        for func in tir_funcs:
            print(func)
    print()
    print("Performance Measurements:")
    measurement = network_result.measurement
    pprint(measurement)
    print()
