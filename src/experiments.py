from timeit import timeit
import time

# TODO
def time_exec_stupid(func, *args, **kwargs):

    # warmup
    func(*args, **kwargs)

    start_time = time.time()
    while True:
        if time.time() - start_time > 45:
            break
        func(*args, **kwargs)

    return timeit(lambda: func(*args, **kwargs), number=1)


def time_exec(func, *args, **kwargs):
    return timeit(lambda: func(*args, **kwargs), number=1)
