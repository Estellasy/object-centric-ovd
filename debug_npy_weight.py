import numpy as np
weight_path = 'datasets/zeroshot_weights/neudet_clip_a+defect+cname.npy'
weights = np.load(weight_path)
print(f"Weight shape: {weights.shape}") # （6，1024）这里包含背景类