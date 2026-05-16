import ray
import time
from tip_assembly_env import Assembly

if __name__ == "__main__":
    a_remote = ray.remote(Assembly)
    a_remote = a_remote.options(num_gpus=1)
    worker = a_remote.remote()

    for _ in range(100):
        time.sleep(0.1)