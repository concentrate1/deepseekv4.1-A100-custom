"""Pure validation of dense-stage and routed-expert device placement."""


def expert_shard_layout(dense_devices, expert_devices, counts, total_experts):
    devices = list(dense_devices if expert_devices is None else expert_devices)
    if not devices or len(set(devices)) != len(devices):
        raise ValueError('expert devices must be nonempty and unique')
    if not set(dense_devices).issubset(devices):
        raise ValueError('every dense owner must participate in expert parallelism')
    if counts is None:
        bounds = [total_experts * i // len(devices) for i in range(len(devices)+1)]
        counts = [b-a for a,b in zip(bounds,bounds[1:])]
    if len(counts) != len(devices) or any(n <= 0 for n in counts) or sum(counts) != total_experts:
        raise ValueError('positive expert shard counts must cover all experts exactly')
    result=[]
    start=0
    for device,count in zip(devices,counts):
        result.append((device,start,count))
        start+=count
    return result
