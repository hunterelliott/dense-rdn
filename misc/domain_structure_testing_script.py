import numpy as np
from matplotlib import pyplot as plt
import domains

import tensorflow as tf

x_shape = (32, 32, 3)
x_buffer = 8
yard = 0  # x_shape[0] + 2 * x_buffer
rd = domains.ReplicationDomain(x_shape, buffer=x_buffer, yard=yard)


fig = plt.figure()
ax = plt.gca()
ax.set_aspect('equal')

domains.draw_domain(rd.full_domain)

domains.draw_domain(rd.parent_domain)
domains.draw_domain(rd.daughter_domains[0])
domains.draw_domain(rd.daughter_domains[1])

print("Parent padding:")
print(domains.get_domain_padding(rd.full_domain, rd.parent_domain))
print("Parent with buffer padding:")
print(domains.get_domain_padding(rd.full_domain, rd.parent_domain, buffer=x_buffer))
print("Child 1 padding:")
print(domains.get_domain_padding(rd.full_domain, rd.daughter_domains[0]))
print("Child 2 padding:")
print(domains.get_domain_padding(rd.full_domain, rd.daughter_domains[1]))

test_coord = ((0,1), (0,3))
test_domain_from_coord = domains.Domain(coords=test_coord)

assert np.all(np.array(test_coord) == np.array(test_domain_from_coord.coords))

test_domain = domains.Domain(center=test_domain_from_coord.center,
                             height=test_domain_from_coord.height,
                             width=test_domain_from_coord.width)

assert np.all(np.array(test_domain_from_coord.coords) == np.array(test_domain.coords))
assert np.all(test_domain_from_coord.polygon == test_domain.polygon)


full_x = np.zeros(shape=(1,) + rd.full_domain.shape + (1,))

def paint_in_domain(x, domain, value):
    x[:, domain.t:domain.b, domain.l:domain.r] = x[:, domain.t:domain.b, domain.l:domain.r] + value
    return x

full_x = paint_in_domain(full_x, rd.full_domain, -1.0)
full_x = paint_in_domain(full_x, rd.parent_domain, 2.0)

x_parent = tf.keras.layers.Cropping2D(domains.get_domain_padding(rd.full_domain, rd.parent_domain))(full_x)
print("Parent values:")
print(np.unique(x_parent))
assert np.unique(x_parent) == 1.0
assert x_parent.shape[1] == x_shape[0]
assert x_parent.shape[2] == x_shape[1]

x_hat_parent = tf.keras.layers.Cropping2D(domains.get_domain_padding(rd.full_domain, rd.parent_domain, buffer=x_buffer))(full_x)
print("Parent and buffer values:")
print(np.unique(x_hat_parent))
print("Parent with buffer shape:")
print(x_hat_parent.shape)

full_x = paint_in_domain(full_x, rd.daughter_domains[0], 1.1)
full_x = paint_in_domain(full_x, rd.daughter_domains[1], 2.1)

x_daughter_0 = tf.keras.layers.Cropping2D(domains.get_domain_padding(rd.full_domain, rd.daughter_domains[0]))(full_x)
print("Daughter 1 values:")
print(np.unique(x_daughter_0))

assert x_daughter_0.shape[1] == x_shape[0]
assert x_daughter_0.shape[2] == x_shape[1]

x_daughter_1 = tf.keras.layers.Cropping2D(domains.get_domain_padding(rd.full_domain, rd.daughter_domains[1]))(full_x)
print("Daughter 2 values:")
print(np.unique(x_daughter_1))

assert x_daughter_1.shape[1] == x_shape[0]
assert x_daughter_1.shape[2] == x_shape[1]

fig = plt.figure()
plt.imshow(full_x[0, :, :, 0])
plt.show()

j=1