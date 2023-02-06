import time
import torch_xla.core.xla_model as xm
import torch_xla.debug.profiler as xp
import torch_xla.distributed.xla_multiprocessing as xmp
from torch_xla.experimental import pjrt

def fn(index):
  device = xm.xla_device()
  if pjrt.using_pjrt():
    print(pjrt.device_attributes(str(device)), f"rank {pjrt.global_ordinal()}")

  timing = []
  for i in range(10):
    t_start = time.perf_counter()
    xm.rendezvous(f"test_{i}")
    t_end = time.perf_counter()
    timing.append(t_end - t_start)

  print(f"rank {xm.get_ordinal()}", timing)


if __name__ == '__main__':

  xmp.spawn(fn, args=(), nprocs=4)
