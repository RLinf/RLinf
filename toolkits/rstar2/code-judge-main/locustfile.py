# locust --headless --users 20 --spawn-rate 1 -H http://0.0.0.0:8000

from locust import HttpUser, task, between


class StreeUser(HttpUser):
    wait_time = between(0.01, 0.02)

    @task(50)
    def test_cpp(self):
        self.client.post("/judge", json={
            "type": "cpp",
            "solution": """#include <cstdio>
#include <unistd.h>
int main(){sleep(3);printf("a");return 0;}
""",
            "expected_output": "a"
        })

    @task(50)
    def test_python(self):
        self.client.post("/judge", json={
            "type": "python",
            "solution": "print(input())",
            "input": "a",
            "expected_output": "a"
        })
