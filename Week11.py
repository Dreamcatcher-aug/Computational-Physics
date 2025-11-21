from mpi4py import MPI
import numpy as np

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    N = 11111111

    N = comm.bcast(N, root=0)
    chunk_size = N // size
    start = rank * chunk_size + 1
    end = N if rank == size - 1 else (start + chunk_size - 1)
    local_sum = 0
    for num in range(start, end + 1):
        local_sum += num
    all_local_sums = comm.gather(local_sum, root=0)
    if rank == 0:
        total_sum = sum(all_local_sums)
        print(f"分布式计算结果：1到{N}的累加和 = {total_sum}")

if __name__ == "__main__":
    main()