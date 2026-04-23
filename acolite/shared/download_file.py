## def download_file
## download_file with authorisation option
## written by Quinten Vanhellemont, RBINS for the PONDER project
## 2018-03-14
## modifications: 2018-11-19 (QV) added verbosity option, removed the parallel download
##                2020-04-24 (QV) earthdata login
##                2021-05-31 (QV) added local directory creation
##                2022-04-08 (QV) download to scratch directory first
##                2022-07-07 (QV) added SRTM1 DEM
##                2022-08-04 (QV) added GED and retry option
##                2022-08-17 (QV) added .netrc auth, simplified url checks for earthdata
##                2024-05-01 (QV) added earthdatacloud.nasa.gov check for earthdata
##                2024-05-22 (QV) use EARTHDATA_urls from config
##                2026-04-23 (Copilot) replaced recursive retry with bounded loop,
##                                     exponential backoff with full jitter,
##                                     Retry-After honouring, fast-fail on non-retryable 4xx,
##                                     dropped unconditional time.sleep(1), thread-safe
##                                     scratch handling

## HTTP statuses that should trigger a retry rather than a permanent failure
_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


def _parse_retry_after(value):
    """Parse a Retry-After header value (seconds or HTTP-date) into a float number of
    seconds. Returns None if it cannot be parsed."""
    if value is None: return None
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    try:
        from email.utils import parsedate_to_datetime
        import datetime
        dt = parsedate_to_datetime(value)
        if dt is None: return None
        if dt.tzinfo is None:
            now = datetime.datetime.utcnow()
        else:
            now = datetime.datetime.now(dt.tzinfo)
        delta = (dt - now).total_seconds()
        if delta < 0: return 0.0
        return delta
    except Exception:
        return None


def download_file(url, file, auth = None, session = None,
                    parallel = False, verbosity = 0, verify_ssl = True, retry = 4,
                    backoff_base = 1.0, backoff_cap = 30.0):

    import requests, time, os, shutil, netrc, random, uuid
    import acolite as ac

    file_path = os.path.abspath(file)
    file_dir = os.path.dirname(file_path)
    if file_dir and not os.path.exists(file_dir):
        os.makedirs(file_dir, exist_ok = True)

    ## skip if destination already exists
    if os.path.exists(file_path):
        if verbosity > 1:
            print("Skipping {}, already present at {}".format(url, file_path))
        return

    ## first download to a unique temp location to avoid clashes between threads
    scratch_dir = ac.config['scratch_dir']
    if not os.path.exists(scratch_dir):
        os.makedirs(scratch_dir, exist_ok = True)
    bn = os.path.basename(file_path)
    temp_file = '{}/{}.{}.{}.part'.format(
        scratch_dir, bn, os.getpid(), uuid.uuid4().hex,
    )
    if os.path.exists(temp_file):
        try: os.remove(temp_file)
        except OSError: pass

    start = time.time()

    if any([u in url for u in ac.config['EARTHDATA_urls']]):
        ## try to get auth from netrc
        if (auth is None):
            try:
                nr = netrc.netrc()
                ret = nr.authenticators('earthdata')
                if ret is not None:
                    login, account, password = ret
                    login = login.strip('"')
                    password = password.strip('"')
                    auth = (login, password)
            except:
                pass

        if (auth is None) & ('EARTHDATA_u' in os.environ) & ('EARTHDATA_p' in os.environ):
            username = os.environ['EARTHDATA_u']
            password = os.environ['EARTHDATA_p']
            auth = (username, password)
        if (auth is None):
            print('EARTHDATA user name and password required for download of {}'.format(url))
            return()

    attempts = max(1, int(retry) + 1)
    last_error = None

    for attempt in range(1, attempts + 1):
        try:
            with requests.Session() as s:
                r1 = s.request('get', url, verify=verify_ssl)
                r = s.get(r1.url, auth=auth, verify=verify_ssl)

                if r.ok:
                    with open(temp_file, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=1024*1024):
                            if chunk: # filter out keep-alive new chunks
                                f.write(chunk)
                    last_error = None
                    break

                ## non-OK response
                status = r.status_code
                if verbosity > 2: print(r.text)

                if status not in _RETRYABLE_STATUS:
                    last_error = Exception(
                        "Download of {} failed with HTTP {}".format(url, status))
                    break

                last_error = Exception(
                    "Download of {} failed with HTTP {} (attempt {}/{})".format(
                        url, status, attempt, attempts))

                if attempt >= attempts: break

                retry_after = _parse_retry_after(r.headers.get('Retry-After'))
                if retry_after is not None:
                    sleep_for = min(backoff_cap, max(0.0, retry_after))
                else:
                    sleep_for = min(backoff_cap, backoff_base * (2 ** (attempt - 1)))
                    sleep_for += random.uniform(0.0, backoff_base)
                if verbosity > 0:
                    print('HTTP {} for {}, retrying in {:.1f}s ({}/{})'.format(
                        status, url, sleep_for, attempt, attempts - 1))
                time.sleep(sleep_for)

        except (requests.ConnectionError, requests.Timeout) as e:
            last_error = e
            if attempt >= attempts: break
            sleep_for = min(backoff_cap, backoff_base * (2 ** (attempt - 1)))
            sleep_for += random.uniform(0.0, backoff_base)
            if verbosity > 0:
                print('Network error for {} ({}), retrying in {:.1f}s ({}/{})'.format(
                    url, type(e).__name__, sleep_for, attempt, attempts - 1))
            time.sleep(sleep_for)

    ## copy temp file
    if os.path.exists(temp_file) and last_error is None:
        try:
            shutil.move(temp_file, file_path)
        except Exception:
            shutil.copyfile(temp_file, file_path)
            try: os.remove(temp_file)
            except OSError: pass
    else:
        if os.path.exists(temp_file):
            try: os.remove(temp_file)
            except OSError: pass

    if last_error is not None and not os.path.exists(file_path):
        raise Exception("File download failed: {}".format(last_error))

    if verbosity > 1:
        print("Downloaded {}, elapsed Time: {:.1f}s".format(url, time.time() - start))
