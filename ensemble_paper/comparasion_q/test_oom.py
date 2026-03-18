import tracemalloc
import gc
import psutil

print(f"Initial Mem: {psutil.virtual_memory().percent}%")
