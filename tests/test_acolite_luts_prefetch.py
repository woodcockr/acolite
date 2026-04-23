"""Tests for the parallel LUT prefetch path in ``ac.acolite.acolite_luts``.

Three of the four tests are fully mocked and run offline. The fourth test
(``test_retrieve_luts_cli_real_download_l8_oli``) actually downloads a small
set of L8_OLI LUTs and is skipped unless ``ACOLITE_TEST_NETWORK=1`` is set.
"""
import os
import sys
import time

import pytest

import acolite as ac
from acolite.aerlut.import_lut import _remote_paths_lut
from acolite.aerlut.import_rsky_lut import _remote_paths_rsky
from acolite.aerlut.reverse_lut import _remote_paths_reverse


_BASE_LUTS = ['ACOLITE-LUT-202110-MOD1', 'ACOLITE-LUT-202110-MOD2']
_PRESSURES = [500, 750, 1013, 1100]
_RSKY_LUT = 'ACOLITE-RSKY-202102-82W'
_PARS = ['romix', 'romix+rsky_t']


def _expected_sensor_lut_jobs(sensor):
    jobs = []
    for base_lut in _BASE_LUTS:
        for pr in _PRESSURES:
            lutid = '{}-{}mb'.format(base_lut, '{}'.format(pr).zfill(4))
            lutdir = '{}/{}'.format(ac.config['lut_dir'], '-'.join(lutid.split('-')[0:3]))
            jobs.append(_remote_paths_lut(lutid, lutdir, sensor=sensor))
    return jobs


def _expected_rsky_jobs(sensor):
    return [_remote_paths_rsky(model, lutbase=_RSKY_LUT, sensor=sensor)
            for model in (1, 2)]


# ---------------------------------------------------------------------------
# 1. Mocked end-to-end: prefetch builder produces the expected job list
# ---------------------------------------------------------------------------
def test_acolite_luts_prefetch_builds_expected_jobs(monkeypatch):
    captured = []

    def fake_download_files(jobs, **kw):
        captured.extend(jobs)
        return {p: True for _, p in jobs}

    monkeypatch.setattr(ac.shared, 'download_files', fake_download_files)
    ## stop after prefetch so we don't try to read non-existent NetCDF files
    monkeypatch.setattr(ac.aerlut, 'import_luts',
                        lambda *a, **k: (_ for _ in ()).throw(SystemExit('STOP')))

    sensors = ['L8_OLI', 'S2A_MSI']

    ## --- pass 1: compute_reverse=False ---
    captured.clear()
    try:
        ac.acolite.acolite_luts(sensor=','.join(sensors), compute_reverse=False)
    except SystemExit:
        pass

    expected = []
    for s in sensors:
        expected.extend(_expected_sensor_lut_jobs(s))
        expected.extend(_expected_rsky_jobs(s))

    assert len(captured) == len(expected) == 2 * (8 + 2)
    assert set(captured) == set(expected)
    ## paths must be absolute and unique
    paths = [p for _, p in captured]
    assert len(paths) == len(set(paths))
    for url, path in captured:
        assert url.startswith('http')
        assert os.path.isabs(path)

    ## --- pass 2: compute_reverse=True ---
    captured.clear()
    try:
        ac.acolite.acolite_luts(sensor=','.join(sensors), compute_reverse=True)
    except SystemExit:
        pass

    rev_paths = [p for _, p in captured if 'Reverse' in p]
    assert rev_paths, 'expected at least one reverse-LUT job'

    ## reverse jobs must only target sensors in ac.config['reverse_lut_sensors']
    for s in sensors:
        if s in ac.config['reverse_lut_sensors']:
            rsr_file = ac.config['data_dir'] + '/RSR/{}.txt'.format(s)
            _, rsr_bands = ac.shared.rsr_read(rsr_file)
            for base_lut in _BASE_LUTS:
                for par in _PARS:
                    for b in rsr_bands:
                        url, path = _remote_paths_reverse(s, base_lut, par, b)
                        assert (url, path) in captured, \
                            'missing reverse job for {}/{}/{}/{}'.format(s, base_lut, par, b)
        else:
            assert not any('/{}/'.format(s) in p for p in rev_paths if 'Reverse' in p), \
                'unexpected reverse job for sensor {} (not in reverse_lut_sensors)'.format(s)


# ---------------------------------------------------------------------------
# 1b. Per-sensor load loop: parallel dispatch, fault isolation, short-circuit
# ---------------------------------------------------------------------------
def _stub_acolite_luts_internals(monkeypatch, import_luts_side_effect=None):
    """Common stubs for the per-sensor load loop tests: skip prefetch and
    no-op the gas/reverse calls so we can drive the loop without I/O."""
    monkeypatch.setattr(ac.shared, 'download_files', lambda jobs, **kw: {})
    monkeypatch.setattr(ac.ac, 'gas_transmittance', lambda *a, **kw: {})
    monkeypatch.setattr(ac.aerlut, 'reverse_lut', lambda *a, **kw: None)
    if import_luts_side_effect is not None:
        monkeypatch.setattr(ac.aerlut, 'import_luts', import_luts_side_effect)


def test_per_sensor_loop_dispatches_concurrently(monkeypatch):
    """With multiple sensors, the load loop should dispatch on a thread pool."""
    import threading

    threads_seen = set()
    sensors_seen = []
    barrier = threading.Barrier(4, timeout=5)

    def fake_import_luts(sensor=None, **kw):
        threads_seen.add(threading.current_thread().name)
        sensors_seen.append(sensor)
        ## block until all 4 workers reach this point - proves parallelism
        barrier.wait()
        return {}

    _stub_acolite_luts_internals(monkeypatch, import_luts_side_effect=fake_import_luts)

    ac.acolite.acolite_luts(sensor='L8_OLI,S2A_MSI,L9_OLI,S2B_MSI',
                            compute_reverse=False, get_remote=False)

    ## all four sensors processed and at least 2 distinct worker threads used
    assert len(sensors_seen) == 4
    assert len(threads_seen) >= 2, 'expected concurrent dispatch, got {}'.format(threads_seen)


def test_per_sensor_loop_isolates_failures(monkeypatch):
    """A failure in one sensor must not prevent other sensors from completing,
    and must surface as a final exception identifying the offending sensor."""
    completed = []

    def fake_import_luts(sensor=None, **kw):
        if sensor == 'S2A_MSI':
            raise RuntimeError('boom')
        completed.append(sensor)
        return {}

    _stub_acolite_luts_internals(monkeypatch, import_luts_side_effect=fake_import_luts)

    with pytest.raises(Exception, match='S2A_MSI'):
        ac.acolite.acolite_luts(sensor='L8_OLI,S2A_MSI,L9_OLI,S2B_MSI',
                                compute_reverse=False, get_remote=False)

    ## the three healthy sensors all finished despite the failing one
    assert set(completed) == {'L8_OLI', 'L9_OLI', 'S2B_MSI'}


def test_per_sensor_loop_single_sensor_stays_on_main_thread(monkeypatch):
    """Single-sensor case must short-circuit the pool to avoid overhead."""
    import threading

    seen_thread = []

    def fake_import_luts(sensor=None, **kw):
        seen_thread.append(threading.current_thread().name)
        return {}

    _stub_acolite_luts_internals(monkeypatch, import_luts_side_effect=fake_import_luts)

    ac.acolite.acolite_luts(sensor='L8_OLI', compute_reverse=False, get_remote=False)

    assert seen_thread == [threading.main_thread().name]


# ---------------------------------------------------------------------------
# 2. download_files: idempotency, parallelism, config override
# ---------------------------------------------------------------------------
def test_download_files_parallel_and_idempotent(monkeypatch, tmp_path):
    ## one pre-existing target, three to "download"
    existing = tmp_path / 'existing.bin'
    existing.write_bytes(b'already here')

    targets = [tmp_path / 'a.bin', tmp_path / 'b.bin', tmp_path / 'c.bin']
    jobs = [('http://example.invalid/{}'.format(t.name), str(t))
            for t in targets]
    jobs.insert(0, ('http://example.invalid/existing', str(existing)))

    call_log = []

    def fake_download(url, file, **kw):
        call_log.append((url, file, time.monotonic()))
        time.sleep(0.1)  # simulate I/O
        with open(file, 'wb') as f:
            f.write(b'stub')

    monkeypatch.setattr(ac.shared, 'download_file', fake_download)

    ## parallel run with 4 workers
    t0 = time.monotonic()
    results = ac.shared.download_files(jobs, max_workers=4, verbosity=0)
    elapsed_par = time.monotonic() - t0

    assert results[os.path.abspath(str(existing))] is True
    for t in targets:
        assert results[os.path.abspath(str(t))] is True
        assert t.exists()
    ## fake_download was NOT called for the pre-existing file
    assert not any(c[1] == os.path.abspath(str(existing)) for c in call_log)
    assert len(call_log) == 3
    ## three 0.1s tasks should comfortably finish in <0.25s with 4 workers
    assert elapsed_par < 0.25, 'parallel dispatch too slow: {:.3f}s'.format(elapsed_par)

    ## clean up to re-test serially via config override
    for t in targets:
        t.unlink()
    call_log.clear()

    monkeypatch.setitem(ac.config, 'lut_download_workers', 1)
    t0 = time.monotonic()
    ac.shared.download_files(jobs, max_workers=4, verbosity=0)
    elapsed_seq = time.monotonic() - t0

    ## three 0.1s tasks dispatched serially → at least 0.3s
    assert elapsed_seq >= 0.28, 'serial override did not take effect: {:.3f}s'.format(elapsed_seq)
    assert len(call_log) == 3


# ---------------------------------------------------------------------------
# 3. download_file: backoff, Retry-After, fast-fail on 4xx
# ---------------------------------------------------------------------------
class _FakeResponse:
    def __init__(self, status, headers=None, body=b''):
        self.status_code = status
        self.ok = 200 <= status < 300
        self.headers = headers or {}
        self.url = 'http://example.invalid/file'
        self.text = body.decode('utf-8', 'replace') if body else ''
        self._body = body

    def iter_content(self, chunk_size=1024 * 1024):
        if self._body:
            yield self._body


class _FakeSession:
    """Pre-scripted requests.Session replacement; ``scripted`` is shared across
    instances so re-instantiation between retry attempts preserves the queue."""
    def __init__(self, scripted):
        self._scripted = scripted

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def request(self, method, url, **kw):
        ## mirror the redirect probe: peek without consuming
        return self._scripted[0]

    def get(self, url, **kw):
        return self._scripted.pop(0)


def test_download_file_backoff_and_retry_after(monkeypatch, tmp_path):
    monkeypatch.setitem(ac.config, 'scratch_dir', str(tmp_path / 'scratch'))

    sleeps = []
    monkeypatch.setattr('time.sleep', lambda s: sleeps.append(s))

    scripted = [
        _FakeResponse(429, headers={'Retry-After': '0'}),
        _FakeResponse(503),
        _FakeResponse(200, body=b'payload'),
    ]
    monkeypatch.setattr('requests.Session', lambda: _FakeSession(scripted))

    dest = tmp_path / 'ok.bin'
    ac.shared.download_file('http://example.invalid/file', str(dest),
                            retry=4, backoff_base=0.5, backoff_cap=2.0,
                            verbosity=0)

    assert dest.exists() and dest.read_bytes() == b'payload'
    ## Two retry sleeps recorded: first honours Retry-After (~0s), second is backoff
    assert len(sleeps) == 2
    assert sleeps[0] <= 2.0  # capped
    assert 0 < sleeps[1] <= 2.0  # exponential backoff with jitter, capped

    ## --- 404 must be fatal, no retries ---
    sleeps.clear()
    scripted_404 = [_FakeResponse(404)]
    monkeypatch.setattr('requests.Session', lambda: _FakeSession(scripted_404))

    bad = tmp_path / 'missing.bin'
    with pytest.raises(Exception):
        ac.shared.download_file('http://example.invalid/missing', str(bad),
                                retry=4, backoff_base=0.5, backoff_cap=2.0,
                                verbosity=0)
    assert not bad.exists()
    assert sleeps == []  # no retry sleep for non-retryable status


# ---------------------------------------------------------------------------
# 4. Real network download via the CLI entry point (opt-in)
# ---------------------------------------------------------------------------
@pytest.mark.network
@pytest.mark.skipif(os.environ.get('ACOLITE_TEST_NETWORK') != '1',
                    reason='network test; set ACOLITE_TEST_NETWORK=1 to enable')
def test_retrieve_luts_cli_real_download_l8_oli(monkeypatch, tmp_path):
    """Download L8_OLI LUTs via the actual CLI entry point.

    Exercises the user-facing path: ``python launch_acolite.py --retrieve_luts
    --sensor L8_OLI``. Redirects ``ac.config['lut_dir']`` and ``scratch_dir``
    into ``tmp_path`` so the repo's data cache is untouched and the run is
    fully reproducible.
    """
    lut_root = tmp_path / 'LUT'
    scratch = tmp_path / 'scratch'
    monkeypatch.setitem(ac.config, 'lut_dir', str(lut_root))
    monkeypatch.setitem(ac.config, 'scratch_dir', str(scratch))

    ## import the launch script and call its entry function in-process
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    import launch_acolite as launcher

    monkeypatch.setattr(sys, 'argv',
                        ['launch_acolite.py', '--retrieve_luts', '--sensor', 'L8_OLI'])
    launcher.launch_acolite()

    ## compute expected files via the same helpers used by the prefetch
    expected = []
    expected.extend(_expected_sensor_lut_jobs('L8_OLI'))
    expected.extend(_expected_rsky_jobs('L8_OLI'))

    for url, path in expected:
        assert os.path.exists(path), 'missing file after retrieve_luts: {}'.format(path)
        assert os.path.getsize(path) > 1024, 'suspiciously small file: {}'.format(path)

    ## warm-cache idempotency: second run must not call download_file at all
    download_calls = []
    real_download = ac.shared.download_file

    def recording_download(*a, **kw):
        download_calls.append(a[:2])
        return real_download(*a, **kw)

    monkeypatch.setattr(ac.shared, 'download_file', recording_download)
    launcher.launch_acolite()
    assert download_calls == [], \
        'expected no downloads on warm cache, got {}'.format(download_calls)
