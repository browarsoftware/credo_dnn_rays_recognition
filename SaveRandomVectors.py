import random
import numpy as np
from numpy import savetxt
sample_size = 235
my_seeds = [0, 101, 542, 1011, 3333, 4321, 6000, 7777, 10111, 15151]
vectors = np.zeros([sample_size, len(my_seeds)])
vectors = vectors.astype(int)
for a in range(len(my_seeds)):
    my_seed = my_seeds[a]
    random.seed(my_seed)
    my_random_sample = random.sample(range(2350), sample_size)
    vectors[:, a] = my_random_sample

print(vectors)
savetxt('RandomVectors.csv', vectors, delimiter=',')