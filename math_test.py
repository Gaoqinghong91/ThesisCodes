import numpy as np
depth_labels=128
t = []
alpha_=1e-3
print(alpha_)
beta_=1
# for K in range(depth_labels):
#     K_ = K
#     t.append((np.exp(np.log(alpha_) + np.log(beta_ / alpha_) * K_ /(depth_labels))))
for K in range(depth_labels):
    K_ = K
    t.append((np.log(np.exp(alpha_) + (np.exp(beta_ )- np.exp(alpha_)) * K_ /(depth_labels))))
print(t)
for i in range(depth_labels):
    if i==0:
        continue

    print(t[i]-t[i-1])