import numpy as np
import matplotlib.pyplot as plt

def periodic_boundary(position, box_size):
    new_position = position % box_size
    return new_position

box_size = np.array([10.0, 10.0, 10.0])  #此处以L = 10举例说明
particle_positions = (np.array
([
    [12.5,  -3.2,   5.7],
    [ -2.8,  15.1,  -1.3],
    [  5.0,   5.0,  12.0],
    [  3.1,   2.4,   4.6],
    [  0.0,  10.0,   0.0]
]))

mapped_positions = periodic_boundary(particle_positions, box_size)


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(particle_positions[:, 0], particle_positions[:, 1], particle_positions[:, 2],
           color='red', marker='x', s=150, label='Original Coordinates', alpha=0.8)
ax.scatter(mapped_positions[:, 0], mapped_positions[:, 1], mapped_positions[:, 2],
           color='blue', marker='o', s=80, label='Mapped Coordinates', alpha=0.8)
vertices = (np.array
([
    [0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0],
    [0, 0, 10], [10, 0, 10], [10, 10, 10], [0, 10, 10]
]))
edges = \
[
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7]
]
for edge in edges:
    (ax.plot3D
    (
        [vertices[edge[0]][0], vertices[edge[1]][0]],
        [vertices[edge[0]][1], vertices[edge[1]][1]],
        [vertices[edge[0]][2], vertices[edge[1]][2]],
        color='black', linewidth=3
    ))
ax.set_xlim(-5, 15)
ax.set_ylim(-5, 16)
ax.set_zlim(-5, 15)
ax.set_xlabel('X Coordinate', fontsize=12)
ax.set_ylabel('Y Coordinate', fontsize=12)
ax.set_zlabel('Z Coordinate', fontsize=12)
ax.set_title('3D Coordinate Mapping with Periodic Boundary Conditions', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.show()