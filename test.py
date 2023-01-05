import argparse
import numpy as np
import socket
import time
import torch
import torch_xla.core.xla_model as xm
import torch_xla.debug.profiler as xp
import torch_xla.distributed.xla_multiprocessing as xmp


def parse_args():
  parser = argparse.ArgumentParser(
      description="Test collective communications."
  )

  parser.add_argument(
      "--name",
      dest="name",
      type=str,
      required=True,
      choices=["all_reduce", "all_gather"],
      help="Name of the op to be tested."
  )

  parser.add_argument(
      "--size",
      dest="size",
      type=int,
      default=4096,
      help="Size of the tensor to be used in MB."
  )

  parser.add_argument(
      "--group_size",
      dest="group_size",
      type=int,
      default=0,
      help="Number of ranks in a group."
  )

  parser.add_argument(
      "--repeat",
      dest="repeat",
      type=int,
      default=100,
      help="Number of times to repeat the test."
  )

  parser.add_argument(
      "--warmup",
      dest="warmup",
      type=int,
      default=50,
      help="Number of times to warmup the test."
  )

  return parser.parse_args()


def create_groups(world_size, group_size, interleaved=False):
  if group_size == 0:
    return None

  assert group_size > 1 and world_size % group_size == 0, f"world_size: {world_size}, group_size: {group_size}."

  groups = np.array(range(world_size))
  if interleaved:
    groups = groups.reshape(group_size, world_size // group_size)
    groups = groups.T.tolist()
  else:
    groups = groups.reshape(world_size // group_size, group_size)
    groups = groups.tolist()

  print(f"groups: {groups}")
  return groups


def test_all_reduce(size, repeat, warmup, group_size):
  device = xm.xla_device()
  tensor_size = size * 1024 * 256
  world_size = xm.xrt_world_size()
  print(f"world_size: {world_size}")
  groups = create_groups(world_size, group_size, interleaved=False)
  scale = 1 / world_size
  tensor = torch.rand(tensor_size, dtype=torch.float32, device=device)

  times = []
  for _ in range(warmup + repeat):
    tensor = xm.all_reduce(xm.REDUCE_SUM, tensor, scale=scale, groups=groups)
    t0 = time.time_ns() / (10 ** 9)
    xm.mark_step()
    xm.wait_device_ops()
    t1 = time.time_ns() / (10 ** 9)
    times.append(t1 - t0)

  avg_time = sum(times[warmup:]) / repeat
  throughput = size / avg_time * 2 * (world_size - 1) / world_size

  return avg_time, throughput


def test_all_gather(size, repeat, warmup, group_size):
  device = xm.xla_device()
  tensor_size = size * 1024 * 256
  world_size = xm.xrt_world_size()
  print(f"world_size: {world_size}")
  groups = create_groups(world_size, group_size, interleaved=False)
  chunk_size = tensor_size // world_size
  tensor = torch.rand(chunk_size, dtype=torch.float32, device=device)

  times = []
  for _ in range(warmup + repeat):
    gathered_tensor = xm.all_gather(tensor, groups=groups)
    t0 = time.time_ns() / (10 ** 9)
    xm.mark_step()
    xm.wait_device_ops()
    t1 = time.time_ns() / (10 ** 9)
    times.append(t1 - t0)

  avg_time = sum(times[warmup:]) / repeat
  throughput = size / avg_time * (world_size - 1) / world_size

  return avg_time, throughput


def fn(index, name, size, repeat, warmup, group_size):
  test_fn_map = {"all_reduce": test_all_reduce, "all_gather": test_all_gather}
  test_fn = test_fn_map[name]

  server = xp.start_server(9012, only_on_master=False)

  print(f"Testing collective communication op {name} for {repeat} times...")
  avg_time, throughput = test_fn(size, repeat, warmup, group_size)
  print(f"Average time per iteration: {avg_time} seconds.")
  print(f"Through put per process: {throughput} MB/s, or {throughput * 8 / 1024} Gbit/s.")

  tmp_tensor = torch.tensor(1.).to(xm.xla_device())
  xm.all_reduce(xm.REDUCE_SUM, [tmp_tensor])
  xm.mark_step()
  print(tmp_tensor)
  
  xm.rendezvous("end")
  
  if pjrt.using_pjrt():
    device = xm.xla_device()
    print(pjrt.device_attributes(str(device)), f"local_ordinal: {pjrt.local_ordinal()}", f"global_ordinal: {pjrt.global_ordinal()}")

  xm.rendezvous("end end")

if __name__ == '__main__':
  args = parse_args()

  xmp.spawn(fn, args=(args.name, args.size, args.repeat, args.warmup, args.group_size), nprocs=4)
