import pytest


def test_status(test_client):
    """
    Test the /status endpoint.
    """
    response = test_client.get('/status')
    assert response.status_code == 200
    assert response.json() == {
        'queue': 0,
        'num_workers': 4
    }


@pytest.mark.parametrize("type", ["judge", "run"])
def test_cpp(test_client, type):
    data = {
        "type": "cpp",
        "solution": """#include <cstdio>
#include <unistd.h>
int main(){sleep(3);printf("a");return 0;}
""",
        "expected_output": "a"
    }
    response = test_client.post(f'/{type}', json=data)
    print(response.json())
    assert response.status_code == 200
    assert response.json()['run_success']
    assert response.json()['success']


@pytest.mark.parametrize("type", ["judge", "run"])
def test_cpp_timeout(test_client, type):
    data = {
        "type": "cpp",
        "solution": """#include <cstdio>
#include <unistd.h>
int main(){sleep(10);printf("a");return 0;}
""",
        "expected_output": "a"
    }
    response = test_client.post(f'/{type}', json=data)
    print(response.json())
    assert response.status_code == 200
    assert not response.json()['run_success']
    assert not response.json()['success']
    assert response.json()['reason'] == 'worker_timeout'
    if type == 'run':
        assert response.json()['stdout'].strip() == 'Suicide from timeout.'


@pytest.mark.parametrize("type", ["judge", "run"])
def test_cpp_fail(test_client, type):
    data = {
        "type": "cpp",
        "solution": """#include <cstdio>
int main(){printf("a");return 0;}
""",
        "expected_output": "b"
    }
    response = test_client.post(f'{type}', json=data)
    print(response.json())
    assert response.status_code == 200
    assert response.json()['run_success']
    assert not response.json()['success']


@pytest.mark.parametrize("type", ["judge", "run"])
def test_cpp_compile_error(test_client, type):
    # compile error
    data = {
        "type": "cpp",
        "solution": """#include <cstdio>
int main(){printf("a")xx;return 0;}
""",
        "expected_output": "b"
    }
    response = test_client.post(f'{type}', json=data)
    print(response.json())
    assert response.status_code == 200
    assert not response.json()['run_success']
    assert not response.json()['success']


@pytest.mark.parametrize("type", ["judge", "run"])
def test_python(test_client, type):
    data = {
        "type": "python",
        "solution": "print(input())",
        "input": "a",
        "expected_output": "a"
    }
    response = test_client.post(f'{type}', json=data)
    print(response.json())
    assert response.status_code == 200
    assert response.json()['success']
    assert response.json()['run_success']


@pytest.mark.parametrize("type", ["judge", "run"])
def test_python_timeout(test_client, type):
    data = {
        "type": "python",
        "solution": "from time import sleep\nsleep(10)",
        "input": "a",
        "expected_output": "a"
    }
    response = test_client.post(f'{type}', json=data)
    print(response.json())
    assert response.status_code == 200
    assert not response.json()['success']
    assert not response.json()['run_success']
    assert response.json()['reason'] == 'worker_timeout'
    if type == 'run':
        assert response.json()['stdout'].strip() == 'Suicide from timeout.'


@pytest.mark.parametrize("type", ["judge", "run"])
def test_python_fail(test_client, type):
    data = {
        "type": "python",
        "solution": "print(input())",
        "input": "a",
        "expected_output": "b"
    }
    response = test_client.post(f'{type}', json=data)
    print(response.json())
    assert response.status_code == 200
    assert not response.json()['success']
    assert response.json()['run_success']


@pytest.mark.parametrize("type", ["judge", "run"])
def test_python_missing_input(test_client, type):
    data = {
        "type": "python",
        "solution": "print(input())",
        "input": "",
        "expected_output": "a"
    }
    response = test_client.post(f'{type}', json=data)
    print(response.json())
    assert response.status_code == 200
    assert not response.json()['success']
    assert not response.json()['run_success']


@pytest.mark.parametrize("type", ["judge", "run"])
@pytest.mark.parametrize("batch_type", ["batch", "long-batch"])
def test_batch(test_client, type, batch_type):
    data = {
        'type': 'batch',
        "submissions": [{
        "type": "cpp",
        "solution": """#include <cstdio>
int main(){printf("a");return 0;}
""",
        "expected_output": "b"
        }, {
        "type": "python",
        "solution": "print(input())",
        "input": "a",
        "expected_output": "b"
        }, {
            "type": "python",
            "solution": "print(input())",
            "input": "a",
            "expected_output": "a"
        }, {
            "type": "cpp",
            "solution": """#include <cstdio>
#include <unistd.h>
int main(){sleep(3);printf("a");return 0;}
""",
            "expected_output": "a"
        }]
    }
    response = test_client.post(f'{type}/{batch_type}', json=data)
    print(response.json())
    assert response.status_code == 200
    results = response.json()['results']

    assert len(results) == 4
    assert not results[0]['success']
    assert results[0]['run_success']
    assert not results[1]['success']
    assert results[1]['run_success']
    assert results[2]['success']
    assert results[2]['run_success']
    assert results[3]['success']
    assert results[3]['run_success']


@pytest.mark.parametrize("type", ["judge", "run"])
@pytest.mark.parametrize("batch_type", ["batch", "long-batch"])
def test_batch_fail(test_client, type, batch_type):
    data = {
        'type': 'batch',
        "submissions": [{
        "type": "python",
        "solution": "print(input())",
        "input": "",
        "expected_output": "b"
        }, {
            "type": "python",
            "solution": "print(input())",
            "input": "a",
            "expected_output": "a"
        }, {
        "type": "python",
        "solution": "print(input())",
        "input": "",
        "expected_output": "b"
        }, {
            "type": "python",
            "solution": "print(input())",
            "input": "a",
            "expected_output": "a"
        },{
        "type": "python",
        "solution": "print(input())",
        "input": "",
        "expected_output": "b"
        }, {
            "type": "python",
            "solution": "print(input())",
            "input": "a",
            "expected_output": "a"
        }]
    }
    response = test_client.post(f'{type}/{batch_type}', json=data)
    print(response.json())
    assert response.status_code == 200
    results = response.json()['results']

    assert len(results) == 6
    assert not results[0]['success']
    assert not results[0]['run_success']
    assert results[1]['success']
    assert results[1]['run_success']
    assert not results[2]['success']
    assert not results[2]['run_success']
    assert results[3]['success']
    assert results[3]['run_success']
    assert not results[4]['success']
    assert not results[4]['run_success']
    assert results[5]['success']
    assert results[5]['run_success']


@pytest.mark.parametrize("type", ["judge", "run"])
def test_multi_process(test_client, type):
    code = """
from time import sleep
import os
from multiprocessing import Process


def worker():
    print('Worker process started')
    sleep(100)
    print('Worker process finished')


print('PP: Starting worker process')
# Start a worker process
p = Process(target=worker)
p.start()
print('worker:', os.getpid())
sleep(1)
print('PP: Worker process Finished, but leaving its child process running')
"""
    data = {
        "type": "python",
        "solution": code,
        "input": "",
        "expected_output": "a"
    }
    response = test_client.post(f'{type}', json=data)
    print(response.json())
    assert response.status_code == 200
    assert not response.json()['success']
    assert not response.json()['run_success']


@pytest.mark.parametrize("type", ["judge", "run"])
def test_io_valid(test_client, type):
    code = """
with open('test.txt', 'w') as f:
    f.write('Hello, world!')
"""
    data = {
        "type": "python",
        "solution": code,
        "input": "",
        "expected_output": ""
    }
    response = test_client.post(f'{type}', json=data)
    print(response.json())
    assert response.status_code == 200
    assert response.json()['success']
    assert response.json()['run_success']

@pytest.mark.parametrize("type", ["judge", "run"])
def test_io_invalid(test_client, type):
    from pathlib import Path
    test_file = Path('/tmp/test.txt')
    test_file.unlink(missing_ok=True)
    code = """
import pathlib
print(pathlib.Path('.').resolve())
with open('../test.txt', 'w') as f:
    f.write('Hello, world!')
"""
    data = {
        "type": "python",
        "solution": code,
        "input": "",
        "expected_output": ""
    }
    response = test_client.post(f'{type}', json=data)
    print(response.json())
    assert response.status_code == 200

    print(f"Jailbreak Status (Is sandbox enabled): {not Path('/tmp/test.txt').exists()}")
