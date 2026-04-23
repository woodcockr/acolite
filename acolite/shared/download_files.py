## def download_files
## parallel download of a list of (url, local_path) jobs using a thread pool
## written by GitHub Copilot, 2026-04-23
## thin wrapper around acolite.shared.download_file -- callers continue to get
## per-file backoff/retry semantics; this layer only adds bounded concurrency.

def download_files(jobs, max_workers = 4, verbosity = 0,
                   auth = None, verify_ssl = True, retry = 4,
                   backoff_base = 1.0, backoff_cap = 30.0):
    """Download many files in parallel.

    Arguments
    ---------
    jobs : iterable of (url, local_path)
        Pairs to download. Entries whose ``local_path`` already exists are
        skipped without contacting the network.
    max_workers : int
        Default thread-pool size. Overridden by ``ac.config['lut_download_workers']``
        when that key is present and parses as a positive integer.
    verbosity, auth, verify_ssl, retry, backoff_base, backoff_cap
        Forwarded to :func:`acolite.shared.download_file`.

    Returns
    -------
    dict
        Mapping ``local_path -> True`` on success or ``local_path -> Exception``
        on failure.
    """
    import os
    import concurrent.futures
    import acolite as ac

    ## allow workspace-wide override of pool size via config
    cfg_workers = ac.config.get('lut_download_workers') if hasattr(ac, 'config') else None
    if cfg_workers is not None:
        try:
            cfg_workers = int(cfg_workers)
            if cfg_workers > 0:
                max_workers = cfg_workers
        except (TypeError, ValueError):
            pass

    ## de-dupe and pre-filter jobs that are already on disk
    seen = set()
    pending = []
    results = {}
    for entry in jobs:
        url, local_path = entry[0], entry[1]
        local_abs = os.path.abspath(local_path)
        if local_abs in seen:
            continue
        seen.add(local_abs)
        if os.path.exists(local_abs):
            results[local_abs] = True
            continue
        pending.append((url, local_abs))

    if not pending:
        if verbosity > 0:
            print('download_files: nothing to do (all {} target(s) already present)'.format(len(results)))
        return results

    workers = max(1, min(max_workers, len(pending)))
    total = len(pending)
    if verbosity > 0:
        print('download_files: {} target(s) to fetch with {} worker(s)'.format(total, workers))

    def _job(url, path):
        ac.shared.download_file(
            url, path, auth = auth, verify_ssl = verify_ssl,
            retry = retry, verbosity = verbosity,
            backoff_base = backoff_base, backoff_cap = backoff_cap,
        )

    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers = workers) as ex:
        future_to_path = {ex.submit(_job, u, p): p for u, p in pending}
        for fut in concurrent.futures.as_completed(future_to_path):
            path = future_to_path[fut]
            completed += 1
            try:
                fut.result()
                results[path] = True
                if verbosity > 0:
                    print('[{}/{}] downloaded {}'.format(completed, total, path))
            except Exception as e:
                results[path] = e
                print('[{}/{}] failed {}: {}'.format(completed, total, path, e))

    if verbosity > 0:
        ok = sum(1 for v in results.values() if v is True)
        fail = sum(1 for v in results.values() if v is not True)
        print('download_files: {} ok, {} failed'.format(ok, fail))

    return results
