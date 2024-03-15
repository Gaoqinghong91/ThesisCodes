import numpy as np
import scipy.sparse as s
import matplotlib.pyplot as plt
import networkx as nx
import cv2
import torch

grid = torch.linspace(
    0,10, 5,
    requires_grad=False).unsqueeze(0).unsqueeze(2).unsqueeze(3)

grid = grid.repeat(2, 1, 4, 4).float()
print(grid)
# z=torch.rand(2,1,4, 4)
# print(z)
# a=z*grid
# print(a)
# m=torch.sum(a, dim=1, keepdim=True)
# print(m)
# def connected_adjacency(image, connect, patch_size=(1, 1)):
#     """
#     Creates an adjacency matrix from an image where nodes are considered adjacent 
#     based on 4-connected or 8-connected pixel neighborhoods.

#     :param image: 2 or 3 dim array
#     :param connect: string, either '4' or '8'
#     :param patch_size: tuple (n,m) used if the image will be decomposed into 
#                    contiguous, non-overlapping patches of size n x m. The 
#                    adjacency matrix will be formed from the smaller sized array
#                    e.g. original image size = 256 x 256, patch_size=(8, 8), 
#                    then the image under consideration is of size 32 x 32 and 
#                    the adjacency matrix will be of size 
#                    32**2 x 32**2 = 1024 x 1024
#     :return: adjacency matrix as a sparse matrix (type=scipy.sparse.csr.csr_matrix)
#     """

#     r, c = image.shape[:2]

#     r = int(r / patch_size[0])
#     c = int(c / patch_size[1])
#     print(r,c)
#     if connect == '4':
#         # constructed from 2 diagonals above the main diagonal
#         d1 = np.tile(np.append(np.ones(c-1), [0]), r)[:-1]
#         d2 = np.ones(c*(r-1))
#         upper_diags = s.diags([d1, d2], [1, c])
#         return upper_diags + upper_diags.T

#     elif connect == '8':
#         # constructed from 4 diagonals above the main diagonal
#         d1 = np.tile(np.append(np.ones(c-1), [0]), r)[:-1]
#         d2 = np.append([0], d1[:c*(r-1)])
#         d3 = np.ones(c*(r-1))
#         d4 = d2[1:-1]
#         upper_diags = s.diags([d1, d2, d3, d4], [1, c-1, c, c+1])
#         retua = np.arange(400).reshape((20,20))
# # adj = connected_adjacency(a, '4').toarray()
# # subnodelist = a.flatten('F')
# # print(adj.shape)
# # labels = {k:k for k in subnodelist} 
# # show_graph_with_labels(adj,labels)e ValueError('Invalid parameter \'connect\'={connect}, must be "4" or "8".'
#                     #  .format(connect=repr(connect)))

# def show_graph_with_labels(adjacency_matrix, mylabels):
#     rows, cols = np.where(adjacency_matrix == 1)
#     edges = zip(rows.tolist(), cols.tolist())
#     gr = nx.Graph()
#     gr.add_edges_from(edges)
#     nx.draw(gr, node_size=500, labels=mylabels, with_labels=True)
#     plt.show()


# img_path= "/home/walle/endo_SCARED/dataset_1/keyframe_1/data/left/000001.png"
# image= cv2.imread(img_path)
# # r_img=image.resize(32,40)
# a = np.arange(1280).reshape((40,32))
# adj = connected_adjacency(a, '4').toarray()
# src, dst = np.nonzero(adj)

# # print(src)
# # print(dst)
# r_img = cv2.resize(image,(40,32))
# import torch
# a_1=r_img.reshape(1280,3)
# tensor_1 = torch.FloatTensor(a_1)
# b1=tensor_1[src,:]
# b2=tensor_1[dst,:]
# print(abs(b1 - b2))

#     # grayMat = final_x.type(torch.FloatTensor)
# # grayMat = np.matrix(a_1 ,dtype=np.float64)

# print(b1,b2)
# l1=torch.norm((b1 - b2), dim=1)
# # l1=np.linalg.norm((b1 - b2), axis=1)
# print(l1)
# # r_img = cv2.resize(image,(40,32))


# # cv2.imshow("show_img", r_img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# # a = np.arange(400).reshape((20,20))
# # adj = connected_adjacency(a, '4').toarray()
# # subnodelist = a.flatten('F')
# # print(adj.shape)
# # labels = {k:k for k in subnodelist} 
# # show_graph_with_labels(adj,labels)
