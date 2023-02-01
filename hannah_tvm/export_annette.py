import json
import argparse
from pprint import pprint
from collections import OrderedDict
from tvm.relay import ExprVisitor
from tvm.ir import Op
from hannah_tvm.dataset import DatasetFull

def main():
    args = parse_args()

    dataset = DatasetFull()

    network_model = args.model
    tuner = args.tuner
    board = args.board

    print(args)

    for network_result in dataset.network_results():
        if network_model != 'all' and network_result.model != network_model:
            continue
        if tuner != 'all' and tuner != network_result.tuner:
            continue
        if board != 'all' and board != network_result.board:
            continue
        print("=========================")
        print("Model:", network_result.model)
        print("Board:", network_result.board)
        print("Target:", network_result.target)
        print("Tuner:", network_result.tuner)
        print()

        relay = network_result.relay
        measurements = network_result.measurement
        print("Relay Graph:")
        print(relay)
        print()
        pprint('Performance Measurements:')
        pprint(measurements)

        result_gen = AnnetteResultGenerator(relay)
        res = result_gen.generate_layer_dir()
        res = result_gen.add_durations(measurements)        

        print()
        print('Result:')
        pprint(res)
        print()

        if args.export_path:
            result_gen.json_export_to(args.export_path)
            print(f'Exported hannah-tvm-tune result JSON to: {args.export_path}')
            print()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('tuner')
    parser.add_argument('board')
    parser.add_argument('--export_path')

    return parser.parse_args()


class AnnetteResultGenerator(ExprVisitor):
    '''
    Creates a result data structure based on a TVM measurement run, that can 
    be used by Annette.
    Parses a TVM relay graph, extracts all single operations (layers) as well as
    where they have been fused and creates a result dict, that also contains the
    duration of each fused function.
    '''
    def __init__(self, relay_graph=None):
        super(AnnetteResultGenerator, self).__init__()
        self.relay_graph = relay_graph
        self.layers = OrderedDict()
        self.latest_fn_hash = 'None'

    def visit(self, expr):
        if expr in self.memo_map and isinstance(expr, Op):
            del self.memo_map[expr] # visit all occurrences of Op
        super(ExprVisitor, self).visit(expr)

    def visit_function(self, fn):
        curr_hash = fn.attrs['hash'] if 'hash' in fn.attrs else 'None'
        # Check for hash collisions and update the ones colliding:
        if curr_hash in [h.split('-')[0] for h in self.layers.keys()]:
            curr_hash = self.update_colliding_hashes(curr_hash)

        self.latest_fn_hash = curr_hash

        self.visit(fn.body)
        for x in fn.params:
            self.visit(x)

    def visit_op(self, op):
        # print(f'op: {op}')
        # print(dir(op))
        # # print(op.has_attr('kernel_size'))
        # curr_fn_hash = self.latest_fn_hash
        # if not curr_fn_hash in self.layers.keys():
        #     self.layers[curr_fn_hash] = {'ops': [], 'duration (us)': 0.}
        #     # self.layers.move_to_end(curr_fn_hash, last=True)
        # # self.layers[curr_fn_hash]['ops'].insert(0, op.name)
        # self.layers[curr_fn_hash]['ops'].append(op.name)
        pass

    def visit_call(self, call):
        for a in call.args:
            self.visit(a)

        op = call.op
        if isinstance(op, Op):
            curr_fn_hash = self.latest_fn_hash
            if not curr_fn_hash in self.layers.keys():
                self.layers[curr_fn_hash] = {'ops': [], 'duration (us)': 0.}
                # self.layers.move_to_end(curr_fn_hash, last=True)

            out_op_descr = {'name': op.name}
            if hasattr(call.attrs, 'kernel_size'):
                out_op_descr['kernel_size'] = [int(i) for i in call.attrs.kernel_size]
            if hasattr(call.attrs, 'pool_size'):
                out_op_descr['pool_size'] = [int(i) for i in call.attrs.pool_size]
            if hasattr(call.attrs, 'strides'):
                out_op_descr['strides'] = [int(i) for i in call.attrs.strides]
            self.layers[curr_fn_hash]['ops'].append(out_op_descr)

        self.visit(call.op)

    def update_colliding_hashes(self, hash):
        '''
        Updates hashes aka keys of self.layers, s.t. those that appear multiples times
        are postfixed with "-n", for n = 0, 1, 2, ...
        '''
        new_layer_dict = OrderedDict()
        # num_collisions = len([k for k in self.layers.keys() if k.split('-')[0] == hash])
        num_collisions = 0
        for old_key, val in self.layers.items():
            old_key_hash = old_key.split('-')[0]
            if old_key_hash == hash:
                new_key = f'{old_key_hash}-{num_collisions}'
                num_collisions += 1
            else:
                new_key = old_key
            new_layer_dict[new_key] = val

        self.layers = new_layer_dict
        return f'{hash}-{num_collisions}'

    def generate_layer_dir(self):
        self.visit(self.relay_graph)
        return self.layers

    def add_durations(self, measurement):
        # Insert duration into executed layer data structure:
        for i, (k, v) in enumerate(self.layers.items()):
            orig_hash = k.split('-')[0]
            assert orig_hash == measurement['calls'][i]['Hash']['string']
            v['duration (us)'] = measurement['calls'][i]['Duration (us)']['microseconds']

        # Insert total network duration:
        board_type = list(measurement['device_metrics'].keys())[0]
        total_duration = measurement['device_metrics'][board_type]['Duration (us)']['microseconds']
        self.layers['total'] = {'duration (us)': total_duration, 'ops': None}

        return self.layers

    def json_export_to(self, filepath):
        with open(filepath, 'w') as json_file:
            json.dump(self.layers, json_file)


if __name__ == '__main__':
    main()
