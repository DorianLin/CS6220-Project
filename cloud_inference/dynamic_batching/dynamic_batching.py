from time import sleep
import time

from ray import serve
from starlette.requests import Request

import logging
import sys

logger = logging.getLogger("ray.serve")

from collections import deque
from ray.serve.handle import DeploymentHandle, DeploymentResponse
import ray
import bisect
import asyncio
from torchvision import transforms
import torch
import torchvision
from PIL import Image

@serve.deployment(
    ray_actor_options={"num_cpus": 1,"num_gpus": 0.01},
    max_concurrent_queries=2,
    autoscaling_config={
        "target_num_ongoing_requests_per_replica": 1,
        "min_replicas": 1,
        "initial_replicas": 1,
        "max_replicas": 10,
    }
)
class ModelWorker:
    def __init__(self):
        self.preprocessor = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t[:3, ...]),  # remove alpha channel
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = torchvision.models.get_model("resnet50").to(self.device)

    async def __call__(self, requests):
        start_time = time.time()
        batch_size = len(requests)
        # run inference
        image_uri = requests[0]["uri"]
        image = Image.open(image_uri)
        image_tensor = self.preprocessor(image).unsqueeze(0).to(self.device)
        input_tensors = torch.cat([image_tensor] * batch_size)

        output_tensor = self.model(input_tensors)

        output = [int(torch.argmax(output_tensor[i])) for i in range(batch_size)]
        end_time = time.time()
        return {
            "batch_size": batch_size,
            "outputs": output,
            "lat": end_time - start_time,
        }


@serve.deployment(
    ray_actor_options={"num_cpus": 1},
    max_concurrent_queries=40,
    autoscaling_config={
        "target_num_ongoing_requests_per_replica": 32,
        "min_replicas": 1,
        "initial_replicas": 1,
        "max_replicas": 3,
    }
)
class Scheduler:
    def __init__(self, model_worker, max_batch_size=32, batch_wait_time = 0.02):
        self.queue = deque()  # Tuple of (request_data, future)
        self.lock = asyncio.Lock()
        self.worker = model_worker.options(use_new_handle_api=True)
        self.BATCH_SIZES = [1, 2, 4, 8, 16, 32]
        self.BATCH_MAX_SIZE = max_batch_size
        self.BATCH_WAIT_TIME = batch_wait_time  # 20 milliseconds

        asyncio.create_task(self.monitor_queue())

    async def monitor_queue(self):
        while True:
            await asyncio.sleep(self.BATCH_WAIT_TIME)
            async with self.lock:
                if len(self.queue) > 0:
                    batch_size = min(self.BATCH_MAX_SIZE, len(self.queue))
                    batch = [self.queue.popleft() for _ in range(batch_size)]
                    asyncio.create_task(self.process_batch(batch))

    async def __call__(self, request):
        future = asyncio.Future()
        async with self.lock:
            self.queue.append((request, future))

        return await future

    async def process_batch(self, batch):
        batch_to_process, futures = zip(*batch)
        batch_size = len(batch_to_process)
        batch_to_process, futures = list(batch_to_process), list(futures)
        # pad batch
        pos = bisect.bisect_left(self.BATCH_SIZES, batch_size)
        target_batch_size = self.BATCH_SIZES[min(pos, len(self.BATCH_SIZES) - 1)]

        print("Batch size:", batch_size, "target batch size:", target_batch_size)
        if batch_size < target_batch_size:
            padding_size = target_batch_size - batch_size
            batch_to_process.extend([batch_to_process[0]] * padding_size)

        results = await self.worker.remote(batch_to_process)
        outputs = results["outputs"]

        for i in range(min(len(futures), len(outputs))):
            futures[i].set_result({"class_index": outputs[i], "lat": results["lat"], "batch_size": target_batch_size})



@serve.deployment(
    ray_actor_options={"num_cpus": 1},
    autoscaling_config={
        "target_num_ongoing_requests_per_replica": 200,
        "min_replicas": 1,
        "initial_replicas": 1,
        "max_replicas": 2,
    }
)
class Ingress:
    def __init__(self, scheduler):
        self.scheduler: DeploymentHandle = scheduler.options(
            use_new_handle_api=True,
        )

    async def __call__(self, request: Request):
        try:
            request_data = await request.json()
        except Exception as e:
            return

        return await self.scheduler.remote(request_data)


def app_builder(args):
    ray.init(num_gpus=1)
    return  (
        Ingress.bind(
            Scheduler.bind(
                ModelWorker.bind()
            )
        )
    )