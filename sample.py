import numpy as np
 
matrix = np.array([[[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]],
                   [[11,12, 13],
                   [14, 15, 16],
                   [17, 18, 19]]])
 
# Slicing a subarray
# Get a 2x2 subarray
sub_matrix = matrix[:, 1:2, 1:2 ]  
print(sub_matrix)
arr2 = np.mean(sub_matrix, dtype = np.float64)
print(arr2)