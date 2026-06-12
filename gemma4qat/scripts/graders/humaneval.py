# WARNING: this executes model-generated code. Run only in an isolated environment
# (throwaway VM, container, or unprivileged user with no network/secrets). Never on
# the same machine that holds credentials you care about.
import multiprocessing
import re

FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)


def extract_code(text):
    m = FENCE.search(text)
    return m.group(1) if m else text


def _worker(program, q):
    try:
        env = {"__name__": "__main__"}
        exec(program, env)
        q.put(True)
    except Exception:
        q.put(False)


def check(completion_code, test_code, entry_point, timeout=10):
    program = f"{completion_code}\n{test_code}\ncheck({entry_point})\n"
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_worker, args=(program, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return False
    return (not q.empty()) and q.get()
