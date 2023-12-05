# from locust import HttpUser, task, between

# class User(HttpUser):
#     wait_time = between(1, 5)  # Wait time between tasks (1 to 5 seconds)


#     @task
#     def query_serving_system(self):
#         # Replace "/query" with the endpoint you want to test
#         # Add necessary data for POST request if needed
#         self.client.post("/", json={"key": "value"}) 


import csv
import string
from locust import HttpUser, task, constant_throughput, LoadTestShape
from collections import namedtuple
import random

LOGS_PATH = "logs"
import os

class User(HttpUser):
    wait_time = constant_throughput(5)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_file = open(
            f"{LOGS_PATH}/user_{''.join(random.choices(string.ascii_lowercase, k=10))}.csv",
            "w",
        )
        self.csv_writer = csv.DictWriter(self.log_file, fieldnames=["id","lat", "e2e_lat"])
        self.csv_writer.writeheader()
        self.cnt = 0

    @task
    def query_request(self):
        resp = self.client.post(
            "/",
            json={
                "uri": f"queries/{0}.jpg"
            },
            timeout = 10,
        )
        self.cnt += 1
        res = resp.json()
        self.csv_writer.writerow({
            "id": self.cnt,
            "lat": res['lat'],
            "e2e_lat": resp.elapsed.total_seconds(),
        })

    def on_stop(self):
        self.log_file.close()
        return super().on_stop()


# https://github.com/locustio/locust/blob/master/examples/custom_shape/wait_user_count.py
Step = namedtuple("Step", ["users", "dwell"])


class StepLoadShape(LoadTestShape):
    """
    A step load shape that waits until the target user count has
    been reached before waiting on a per-step timer.
    """

    spawn_rate = 5
    targets_with_times = (
        Step(5, 10),
        Step(10, 10),
        Step(15, 10),
        Step(20, 10),
        Step(30, 20),
        Step(50, 60),
        Step(20, 10),
        Step(10, 10),
        Step(5, 10),
    )

    def __init__(self, *args, **kwargs):
        self.step = 0
        self.time_active = False
        super().__init__(*args, **kwargs)

    def tick(self):
        if self.step >= len(self.targets_with_times):
            return None

        target = self.targets_with_times[self.step]
        users = self.get_current_user_count()

        if target.users == users:
            if not self.time_active:
                self.reset_time()
                self.time_active = True
            run_time = self.get_run_time()
            if run_time > target.dwell:
                self.step += 1
                self.time_active = False

        return (target.users, 0.5)
