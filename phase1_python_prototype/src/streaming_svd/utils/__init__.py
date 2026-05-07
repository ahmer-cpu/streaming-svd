"""Common utilities module.

Timing and statistics tracking are handled inline within each algorithm
(rsvd, warm_rsvd) via the `stats` return dictionary, which contains:
    - stats['timings']       -- per-step wall-clock times (seconds)
    - stats['matmul_counts'] -- number of A@X and A.T@X multiplications
    - stats['params']        -- configuration used for the run
"""
