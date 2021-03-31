import tvm
import tvm.relay as relay
import tvm.relay.testing as testing 
import tvm.auto_scheduler as auto_scheduler

@tvm.tir.transform.prim_func_pass(opt_level=0)
def print_tir(f, mod, ctx):
    print("Analyzing:", f)
    
    
    return f 

def test_analysis():
    mod, params = testing.resnet.get_workload(1, 10, num_layers=18)

    image_shape=(3, 224, 224)

    target = "llvm"
    target_host = "llvm"
    ctx = tvm.cpu(0)
    with tvm.transform.PassContext(opt_level=3, config={"tir.add_lower_pass": [(1, print_tir)]}):
        lib = relay.build(mod, target=target, target_host=target_host, params=params)
        print(lib)

    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)

    for idx, task in enumerate(tasks):
        print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
        print(task.compute_dag)

    log_file = "tuning.csv"
    def run_tuning():
        print("Begin tuning...")
        tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=2000,  # change this to 20000 to achieve the best performance
            runner=auto_scheduler.LocalRunner(repeat=10, enable_cpu_cache_flush=True),
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        )

        tuner.tune(tune_option)

    run_tuning()

if __name__ == "__main__":
    test_analysis()