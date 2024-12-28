import os
import pandas as pd
from math import log
import numpy as np

# Function to calculate log-likelihood for a given set of observations
def log_likelihood(n, k):
    p = k / n
    if p == 0 or p == 1:
        return 0
    return k * np.log(p) + (n - k) * np.log(1 - p)

# Function to bucket FICO scores into categories based on the probability of default
def bucket_fico_scores(fico_scores, defaults, bins):
    num_bins = bins
    x = defaults
    y = fico_scores
    n = len(x)

    # Initialize counts for defaults and total occurrences per FICO score
    default_counts = [0 for _ in range(851)]
    total_counts = [0 for _ in range(851)]

    # Count the number of defaults and total occurrences for each FICO score
    for i in range(n):
        y[i] = int(y[i])  # Ensure fico_scores are integers
        default_counts[y[i] - 300] += x[i]
        total_counts[y[i] - 300] += 1

    # Accumulate totals and defaults for each FICO score
    for i in range(1, 551):  # Start from 1 to avoid out-of-bounds error
        default_counts[i] += default_counts[i - 1]
        total_counts[i] += total_counts[i - 1]

    # Initialize for log-likelihood optimization
    dp = [[[-float('inf'), 0] for _ in range(551)] for _ in range(num_bins + 1)]

    # calculate optimal bucket boundaries
    for i in range(num_bins + 1):
        for j in range(551):
            if i == 0:
                dp[i][j][0] = 0
            else:
                for k in range(j):
                    if total_counts[j] == total_counts[k]:
                        continue
                    if i == 1:
                        dp[i][j][0] = log_likelihood(total_counts[j], default_counts[j])
                    else:
                        new_likelihood = log_likelihood(total_counts[j] - total_counts[k], default_counts[j] - default_counts[k]) + dp[i - 1][k][0]
                        if dp[i][j][0] < new_likelihood:
                            dp[i][j][0] = new_likelihood
                            dp[i][j][1] = k

    # Reconstruct the optimal bucket boundaries from the dynamic programming table
    boundaries = []
    k = 550
    while num_bins > 0:  # Ensure num_bins > 0 to prevent infinite loop
        boundaries.append(k + 300)
        k = dp[num_bins][k][1]
        num_bins -= 1

    print(boundaries)
