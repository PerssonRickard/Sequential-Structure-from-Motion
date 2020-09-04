import cv2 as cv
import numpy as np
from scipy.linalg import null_space
import matplotlib.pyplot as plt
import open3d as o3d

def convertFromHomogeneousPoints(x):

    # Convert from homogeneous coordinates by dividing with the last element in the vector
    for i in range(len(x)):
        x[i,:] = np.divide(x[i,:], x[-1,:])

    return x

def getMatchingPointsFromSIFTKeyPointsAndDescriptors(kp1, kp2, des1, des2):

    # Match the SIFT features
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    goodMatches = []
    # Ratio test to find stable matches
    for m,n in matches:
        if m.distance < 0.8*n.distance:
            goodMatches.append([m])

    # Save the image points corresponding to good matches
    points1 = np.asarray([kp1[goodMatch[0].queryIdx].pt for goodMatch in goodMatches]).transpose()
    points2 = np.asarray([kp2[goodMatch[0].trainIdx].pt for goodMatch in goodMatches]).transpose()

    # Convert to homogeneous coordinates
    x1 = np.concatenate([points1, np.ones([1, points1.shape[1]])], 0)
    x2 = np.concatenate([points2, np.ones([1, points2.shape[1]])], 0)

    return x1, x2, goodMatches

def sequential3DReconstruction(sift, img, P, points3D, desPnP, kp, des, featureTo3DIndices):

    # Find features in the new image corresponding to features in the previous image
    kp2, des2 = sift.detectAndCompute(img,None)

    # Perform matching
    bf = cv.BFMatcher()
    matches = bf.knnMatch(desPnP, des2, k=2)
    
    goodMatches = []
    # Ratio test to find stable matches
    for m,n in matches:
        if m.distance < 0.8*n.distance:
            goodMatches.append([m])

    # Extract the 3D points from the previous image corresponding to the matched features in the new image
    temp = np.zeros([3, len(goodMatches)])
    for i, goodMatch in enumerate(goodMatches):
        temp[:, i] = points3D[:, goodMatch[0].queryIdx]
    orig3DPoints = points3D
    points3D = temp

    # Extract the image points in the new image corresponding to the 3D points
    points3 = np.asarray([kp2[goodMatch[0].trainIdx].pt for goodMatch in goodMatches]).transpose()
    x3 = np.concatenate([points3, np.ones([1, points3.shape[1]])], 0)

    # Find the extrinsic camera parameters from the 3D points and corresponding image points
    r = cv.solvePnPRansac(points3D.transpose(), x3[0:2,:].transpose(), K.astype(np.float32), None, reprojectionError=2)
    R, _ = cv.Rodrigues(r[1])
    t = r[2]

    # Construct the new camera
    P2 = np.concatenate([R, t], axis=1)

    # Unnormalize
    P2_un = np.matmul(K, P2)

    # Once the camera has been found for the new image, use corresponding image features in the previous
    # and new image to triangulate more 3D points.
    # Match the features for the previous and current image (now using all features from the previous image and
    # not only the ones we found corresponding 3D points for)
    xNew, x2New, goodMatches = getMatchingPointsFromSIFTKeyPointsAndDescriptors(kp, kp2, des, des2)

    # Triangulate the points in the new camera
    points3D = convertFromHomogeneousPoints(cv.triangulatePoints(P, P2_un, xNew[0:2,:], x2New[0:2,:]))

    # Save the SIFT features corresponding to the reconstructed 3D points, construct an array defining the mapping
    # from SIFT features to 3D points and construct a boolean array indicating which 3D points correspond to new 
    # 3D points (points which haven't already been reconstructed)
    new3DPointIndices = np.zeros(len(goodMatches), np.bool)
    temp = np.empty([points3D.shape[1], 128], dtype=np.float32)
    featureTo3DIndices2 = np.empty(len(goodMatches))
    for i, match in enumerate(goodMatches):
        temp[i, :] = des2[match[0].trainIdx, :]
        featureTo3DIndices2[i] = match[0].trainIdx

        if match[0].queryIdx not in featureTo3DIndices:
            new3DPointIndices[i] = True
    des2PnP = temp

    # Compute reprojection errors for the 3D points in the cameras
    reproj_x = convertFromHomogeneousPoints(np.matmul(P, points3D))
    reproj_x2 = convertFromHomogeneousPoints(np.matmul(P2_un, points3D))
    reproj_err_x = np.linalg.norm(xNew[0:2,:] - reproj_x[0:2,:], axis=0)
    reproj_err_x2 = np.linalg.norm(x2New[0:2,:] - reproj_x2[0:2,:], axis=0)

    # Transfrom the 3D points to the camera coordinate system of the new camera
    cameraCoordinateSystem = np.matmul(P2, points3D)

    # Points which have a sufficiently low reprojection error and which are in front of the new camera are considered inliers
    reproj_threshold = 0.5
    inliers = np.logical_and(np.logical_and((reproj_err_x < reproj_threshold), (reproj_err_x2 < reproj_threshold)), cameraCoordinateSystem[2,:] >= 0)

    # Remove outliers
    points3D = points3D[:,inliers]
    des2PnP = des2PnP[inliers,:]
    new3DPointIndices = new3DPointIndices[inliers]

    # Print statistical results
    print('Size of largest consensus set found:', inliers.sum(), 'out of', xNew.shape[1], 'total features (' + str(new3DPointIndices.sum()) + ' new 3D points constructed)')
    
    return points3D, new3DPointIndices, P2_un, kp2, des2, des2PnP, featureTo3DIndices2


# Plotting functions
def calculateCameraCenterAndPrincipalAxis(P):
    # Determine the camera center and orientation from the camera matrix
    camera_center = cv.convertPointsFromHomogeneous(np.transpose(null_space(P))).squeeze()
    principal_axis = P[2, 0:3]

    return camera_center, principal_axis

def plotCamera(P, ax, color='r', length=1):
    # Plot a camera using the camera matrix
    camera_center, principal_axis = calculateCameraCenterAndPrincipalAxis(P)
    ax.scatter(camera_center[0], camera_center[1], camera_center[2], marker='o', c=color)
    ax.quiver(camera_center[0], camera_center[1], camera_center[2], principal_axis[0], principal_axis[1], principal_axis[2], length=length, color=color)

    return camera_center, principal_axis

def plotCameras(Ps, ax, color='r', length=1):
    # Plot multiple cameras using an array of camera matrices
    sequentialCameraPositions = np.empty([3, len(Ps)])

    # Plot cameras
    for i, P in enumerate(Ps):
        camera_center, principal_axis = calculateCameraCenterAndPrincipalAxis(P)
        sequentialCameraPositions[:,i] = camera_center
        ax.scatter(camera_center[0], camera_center[1], camera_center[2], marker='o', c=color)
        ax.quiver(camera_center[0], camera_center[1], camera_center[2], principal_axis[0], principal_axis[1], principal_axis[2], length=length, color=color)
    
    # Plot a trajectory (line) showing the sequence of the cameras
    ax.plot(sequentialCameraPositions[0,:], sequentialCameraPositions[1,:], sequentialCameraPositions[2,:], c=color)

def set_axes_equal(ax):
    # Center the plot and make the scale of the axes equal
    x_middle = -1
    y_middle = 3
    z_middle = 10
    x_middle = -1.25
    y_middle = 1.25
    z_middle = 10

    plot_radius = 10

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

# Initial image pair to find initial cameras and 3D points
img1 = cv.imread("images/statue/DSC_0351.JPG", cv.IMREAD_GRAYSCALE)
img2 = cv.imread("images/statue/DSC_0352.JPG", cv.IMREAD_GRAYSCALE)

# Inner camera parameters
K = np.array([[2393.952166119461, -3.410605131648481e-13, 932.3821770809047], [0, 2398.118540286656, 628.2649953288065], [0, 0, 1]])

# Compute SIFT features
sift = cv.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1,None)
kp, des = sift.detectAndCompute(img2,None)

# Find the good matches
x1, x2, goodMatches = getMatchingPointsFromSIFTKeyPointsAndDescriptors(kp1, kp, des1, des)

# Construct the first camera matrix which is arbritary
P1 = np.zeros([3, 4])
P1[0:3,0:3] = np.eye(3)

# Find the essential matrix
result, _ = cv.findEssentialMat(x1[0:2,:].transpose(), x2[0:2,:].transpose(), K)
result = np.divide(result, result[2,2])

# Find cameras and 3D points using the essential matrix
result = cv.recoverPose(result, x1[0:2,:].transpose(), x2[0:2,:].transpose(), K, distanceThresh=10, triangulatedPoints=True)

# Extract the second camera from the result
P = np.matmul(K, np.concatenate([result[1], result[2]], axis=1))

# Extract inlier indices from result
numberOfInliers = result[0]
temp = result[3]
inliers = np.empty([temp.shape[0]], dtype=np.bool)
for i in range(temp.shape[0]):
    if temp[i] > 0:
        inliers[i] = True
    else:
        inliers[i] = False

# Extract 3D points from the result
X = result[4]
X = convertFromHomogeneousPoints(X)
X = X[0:3, :]

# Save the SIFT descriptors corresponding to the constructed 3D points as well as a mapping from SIFT features to 3D points
temp = np.empty([X.shape[1], 128], dtype=np.float32)
featureTo3DIndices = np.empty(len(goodMatches))
for i, match in enumerate(goodMatches):
    temp[i, :] = des[match[0].trainIdx, :]
    featureTo3DIndices[i] = match[0].trainIdx
desPnP = temp

# Remove the outliers
points3D = X[:,inliers]
x2 = x2[:,inliers]
desPnP = desPnP[inliers,:]

# Print statistical results of the initial image pair
print('Size of largest consensus set found:', numberOfInliers, 'out of', x1.shape[1], 'total features')

# Unnormalize the camera
P1_un = np.matmul(K, P1)

# Stack the cameras into an array
cameras = [P1_un, P]

# Stack the 3D points into an array
pointsFromPerspectives = [points3D]

# Perform sequential 3D reconstruction using the remaining images
for i in range(3, 58):
    imgName = "DSC_0" + str(350 + i) + ".JPG"
    img = cv.imread("images/statue/" + imgName, cv.IMREAD_GRAYSCALE)

    print("Processing image", imgName)

    points3D, new3DPointIndices, P, kp, des, desPnP, featureTo3DIndices = sequential3DReconstruction(sift, img, P, points3D, desPnP, kp, des, featureTo3DIndices)
    newPoints3D = points3D[:,new3DPointIndices]
    points3D = convertFromHomogeneousPoints(points3D)
    points3D = points3D[0:3,:]

    pointsFromPerspectives.append(newPoints3D)
    cameras.append(P)


# Collect all reconstructed 3D points into a single numpy array
numberOfPoints = 0
for i in range(len(pointsFromPerspectives)):
    numberOfPoints = numberOfPoints + pointsFromPerspectives[i].shape[1]
points = np.zeros((numberOfPoints, 3))
count = 0
for i in range(len(pointsFromPerspectives)):
    for j in range(pointsFromPerspectives[i].shape[1]):
        points[count,:] = pointsFromPerspectives[i][0:3,j]
        count = count + 1

# Plot the 3D points and cameras
fig = plt.figure(5)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:,0], points[:,1], points[:,2], marker='.', s=0.01, color='blue')
plotCameras(cameras, ax, color='lime', length=2)
set_axes_equal(ax)
fig.show()
plt.show()

# Remove points not close to the statue
statue_center_point = np.array([-1.25, 1.25, 10])
closePoints = np.linalg.norm(points - statue_center_point, axis=1) < 5
points = points[closePoints, :]

# Function to visualize the points while rotating
def visualize_3D_points_with_rotation(pcd):

    def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.rotate(10.0, 0.0)
        return False

    o3d.visualization.draw_geometries_with_animation_callback([pcd],
                                                              rotate_view)

# Print the total number of points in the final point cloud
print("Number of 3D points:", points.shape[0])

# Use open3D to visualize the 3D points smoothly
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Visualize the points
visualize_3D_points_with_rotation(pcd)
