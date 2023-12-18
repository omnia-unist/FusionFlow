import numpy as np

# Load the original NPZ file
data = np.load('policy_DeepAA_imagenet_1.npz')

# Extract the arrays you want to modify
policy_probs = data['policy_probs']
l_ops = data['l_ops']
l_mags = data['l_mags']
ops = data['ops']
mags = data['mags']
op_names = data['op_names']

# Modify the arrays to the new shapes
policy_probs_new_shape = policy_probs[:, :136]  # Reduce to shape (5, 136)
l_ops_new_shape = 15  # Set to the new shape 15
ops_new_shape = ops[:136, :]  # Reduce to shape (136, 1)
op_names_new_shape = op_names[:15]  # Reduce to shape (15,)

# Create a new NPZ file with the modified arrays
np.savez('policy_DeepAA.npz',
         policy_probs=policy_probs_new_shape,
         l_ops=l_ops_new_shape,
         l_mags=l_mags,
         ops=ops_new_shape,
         mags=mags,
         op_names=op_names_new_shape)

# Close the original NPZ file
data.close()