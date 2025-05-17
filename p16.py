## feature scaling using standardization and min max normalization
 
# standardisation is also known as z-score normalization, where xbar= x- mu/sigma( where xbar is the new featuure after scaling
# x is the feature that is to be scaled, mu is the mean of the feature and sigma is the standard deviation of the feature)


import numpy as np
def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
	stdcol=np.std(data, axis=0)
	meancol=np.mean(data, axis=0)
	
	standardized_data=(data-meancol)/stdcol
	normalized_data=(data-np.min(data, axis=0))/(np.max(data, axis=0)-np.min(data, axis=0))
	return standardized_data, normalized_data

data = np.array([[1, 2], [3, 4], [5, 6]])
standardized_data, normalized_data = feature_scaling(data)
print("Standardized Data:\n", standardized_data)    
print("Normalized Data:\n", normalized_data)