import numpy as np

np.random.seed(0)

test_track = []

for row in range(100):
    test_track.append([])
    for column in range(50):
        if column > 10 and column < 45:
            if row >= 1:
                value = np.random.choice([2, 3], p=[0.9, 0.1])
                test_track[row].append(value)
            else:
                test_track[row].append(4)
        else:
            test_track[row].append(1)
