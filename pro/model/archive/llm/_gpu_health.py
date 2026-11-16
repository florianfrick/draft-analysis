import torch

print("available:", torch.cuda.is_available())
free, total = torch.cuda.mem_get_info()
print("free %.2f GB / total %.2f GB" % (free / 1e9, total / 1e9))
a = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
a.fill_(1)
torch.cuda.synchronize()
print("allocated 256 MB ok, sum", int(a.sum()))
free, total = torch.cuda.mem_get_info()
print("free %.2f GB" % (free / 1e9))
big = torch.empty(1024 * 1024 * 1024, dtype=torch.uint8, device="cuda")
big.fill_(2)
torch.cuda.synchronize()
print("allocated +1 GB ok")
del a, big
torch.cuda.empty_cache()
free, total = torch.cuda.mem_get_info()
print("after free: free %.2f GB / total %.2f GB" % (free / 1e9, total / 1e9))
