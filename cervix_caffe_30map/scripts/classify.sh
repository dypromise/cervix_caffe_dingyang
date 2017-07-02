srun -p Test --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 \
    python classify_batch.py
