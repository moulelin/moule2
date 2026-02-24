"""Quick GPU health check."""
import torch

n = torch.cuda.device_count()
print(f"Visible GPUs: {n}")

for i in range(n):
    try:
        t = torch.randn(1000, 1000, device=f"cuda:{i}")
        del t
        torch.cuda.empty_cache()
        print(f"  GPU {i}: OK ({torch.cuda.get_device_name(i)}, "
              f"{torch.cuda.get_device_properties(i).total_mem / 1e9:.1f} GB)")
    except Exception as e:
        print(f"  GPU {i}: FAILED - {e}")

# Test NCCL communication if >1 GPU
if n >= 2:
    print(f"\nNCCL test (GPU 0 <-> GPU 1):")
    try:
        torch.distributed.init_process_group(
            backend="nccl", init_method="tcp://127.0.0.1:29500",
            rank=0, world_size=1)
        a = torch.randn(100, device="cuda:0")
        b = a.to("cuda:1")
        assert torch.allclose(a.cpu(), b.cpu())
        print("  P2P transfer: OK")
        torch.distributed.destroy_process_group()
    except Exception as e:
        print(f"  P2P transfer: FAILED - {e}")
