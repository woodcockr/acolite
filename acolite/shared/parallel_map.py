## def parallel_map
## Unified parallel execution interface supporting concurrent.futures (threading)
## and dask as execution backends, controlled by the acolite-mp_scheduler setting.
## written for ACOLITE-MP
## 2026-04-10

def parallel_map(func, args_list, scheduler='threading', max_workers=None):
    """
    Execute func over args_list in parallel using the specified scheduler.

    Parameters
    ----------
    func : callable
        Function to call. Each element of args_list is passed as a single argument.
    args_list : iterable
        Iterable of argument tuples/values to map over.
    scheduler : str
        Execution backend: 'threading' (concurrent.futures.ThreadPoolExecutor, default)
        or 'dask' (dask.delayed with threaded scheduler).
    max_workers : int or None
        Maximum number of parallel workers. None uses the backend default.

    Returns
    -------
    list
        List of results in the same order as args_list.
    """
    args_list = list(args_list)

    if scheduler == 'dask':
        return _parallel_map_dask(func, args_list, max_workers=max_workers)
    else:
        return _parallel_map_threading(func, args_list, max_workers=max_workers)


def _parallel_map_threading(func, args_list, max_workers=None):
    """Execute func over args_list using concurrent.futures.ThreadPoolExecutor."""
    import concurrent.futures
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(func, args) for args in args_list]
        for future in futures:
            results.append(future.result())
    return results


def _parallel_map_dask(func, args_list, max_workers=None):
    """Execute func over args_list using dask.delayed with the threaded scheduler."""
    import dask
    tasks = [dask.delayed(func)(args) for args in args_list]
    num_workers = max_workers if max_workers is not None else None
    results = dask.compute(*tasks, scheduler='threads', num_workers=num_workers)
    return list(results)
