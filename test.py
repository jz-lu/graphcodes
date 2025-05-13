import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Generate or input your 3D points
points = np.random.rand(30, 3)  # 30 random points in 3D

# Step 2: Compute the convex hull
hull = ConvexHull(points)

# Step 3: Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Step 4: Plot the convex hull as a solid
ax.plot_trisurf(points[:,0], points[:,1], points[:,2], triangles=hull.simplices, cmap='viridis', alpha=0.8)

# Optional: Plot the points
ax.scatter(points[:,0], points[:,1], points[:,2], c='r', marker='o')

# Step 5: Customize the plot
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('3D Convex Hull')

# Step 6: Show the plot
plt.show()