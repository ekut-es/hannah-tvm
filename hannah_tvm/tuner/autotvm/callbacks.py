def progress_callback(step_progress, progress):
    def _callback(tuner, inputs, results):

        with progress.get_lock():
            progress.value += step_progress * len(results)

    return _callback
