import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
import random
import os

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
# !pip install mayavi


plt.style.use('ggplot') # fivethirtyeight


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def visualize_clusterer_surface_and_gradient(classifier, x_test, y_test, cluster_id, outdir):

    if cluster_id is not None:
        clusters_to_plot = cluster_id
    else:
        clusters_to_plot = list(range(classifier.k))

    mesh_lim_m = int(x_test.min() * 1.2)
    mesh_lim_M = int(x_test.max() * 1.2)

    mesh_x, mesh_y = torch.meshgrid(torch.linspace(mesh_lim_m, mesh_lim_M, 150), torch.linspace(mesh_lim_m, mesh_lim_M, 150))
    concated = torch.cat([mesh_x.reshape(-1,1), mesh_y.reshape(-1,1)], dim=1) #.double()

    for cluster_id in clusters_to_plot:
        probs_mesh = classifier.label_guide(concated.cuda(), cluster_id)
        probs_mesh = probs_mesh.reshape(mesh_x.size()).detach().cpu().numpy()

        fig = plt.figure(figsize=(11,8))
        ax = plt.axes(projection='3d')

        fig.set_facecolor('white')
        ax.set_facecolor('white') 
        ax.grid(True) 
        # ax.w_xaxis.pane.fill = False
        # ax.w_yaxis.pane.fill = False
        # ax.w_zaxis.pane.fill = False

        X, Y, Z = mesh_x, mesh_y, probs_mesh
        offset = probs_mesh.min()

        ax.plot_surface(X, Y, Z, cmap="viridis", lw=0.5, rstride=1, cstride=1, alpha=0.75, edgecolors=None, antialiased=True)
        ax.contour(X, Y, Z, 14, cmap="viridis", linestyles="solid", offset=offset)
        ax.contour(X, Y, Z, 23, colors="k", linestyles="solid", alpha = 0.85)

        # Compute the gradients
        from torch.autograd import Variable
        concated_var = Variable(concated, requires_grad = True).cuda()
        concated_var.retain_grad()
        loss = classifier.label_guide(concated_var, cluster_id).sum()
        
        loss.backward()
        concated_grad = concated_var.grad.detach().cpu().numpy()

        U, V, W = concated_grad[:, 0].reshape(X.shape), concated_grad[:, 1].reshape(X.shape), np.array([0.0]*concated_grad.shape[0]).reshape(X.shape)
        Z_offset = np.array([offset]*concated_grad.shape[0]).reshape(X.shape)
        
        ## Plot the gradient field
        def down_sample(x=None):
            return x[::17,::17]

        X, Y, Z_offset, U, V, W = down_sample(X), down_sample(Y), down_sample(Z_offset), down_sample(U), down_sample(V), down_sample(W)
        # ax.quiver(X, Y, Z_offset, U, V, W, length=10.5, normalize=False, pivot='tip', arrow_length_ratio=0.3, color='r', linewidths=1.0)
        
        for i_ in range(X.shape[0]):
            for j_ in range(X.shape[1]):
                ij_ = (i_,j_)

                vec_coord = [[X[ij_] - U[ij_], X[ij_]],
                            [Y[ij_] - V[ij_], Y[ij_]],
                            [Z_offset[ij_] - W[ij_], Z_offset[ij_]]]

                vec_norm = sum([(a[0]-a[1])**2 for a in vec_coord])**0.5
                vec_norm2 = min(10 * np.log(1.1 + 1000 * vec_norm), 0.11 * (mesh_lim_M - mesh_lim_m)) 

                # print(vec_norm2, 0.11 * (mesh_lim_M - mesh_lim_m))

                if vec_norm2 < 0.011 * (mesh_lim_M - mesh_lim_m):
                    continue

                vec_coord = [[X[ij_] - vec_norm2/vec_norm * U[ij_], X[ij_]],
                            [Y[ij_] - vec_norm2/vec_norm * V[ij_], Y[ij_]],
                            [Z_offset[ij_] - vec_norm2/vec_norm * W[ij_], Z_offset[ij_]]]

                a = Arrow3D(*vec_coord, mutation_scale=12, 
                            lw=2, arrowstyle="-|>", color="blueviolet", alpha =0.8, zorder=1000)
                ax.add_artist(a)

        # ax.plot([4.], [0.], [0.], markerfacecolor='k', markeredgecolor='k', marker='x', markersize=10, alpha=0.6)

        ## Plot the dataset on X-Y plane
        indx_cluster_points = y_test.detach().cpu().numpy() == cluster_id 
        ax.scatter(x_test.detach().cpu().numpy()[~indx_cluster_points, 0], x_test.detach().cpu().numpy()[~indx_cluster_points, 1],  
                   zs=offset, zdir='z', c='silver', alpha=0.8, label='All data', zorder=0)
        ax.scatter(x_test.detach().cpu().numpy()[indx_cluster_points, 0], x_test.detach().cpu().numpy()[indx_cluster_points, 1], 
                   zs=offset, zdir='z', c='r', label='Dest. class')

        ## labels and title
        fontdict = {'fontsize': 18,
        'fontweight' : 1.0,
        'verticalalignment': 'baseline',
        'horizontalalignment': 'center'}
        ax.set_xlabel('$\mathbf{X}$', fontdict=fontdict)
        ax.set_ylabel('$\mathbf{Y}$', fontdict=fontdict)
        ax.set_zlabel('$\mathbf{R_{i}}$', fontdict=fontdict)
        # ax.set_title('%d' % cluster_id)
        # ax.set_xlim(-6, 6); ax.set_ylim(-6, 6); 
        ax.set_zlim(offset, 0.5)
        
        # ax.view_init(elev=40., azim=35 + 90 * 3)

        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])

        # plt.legend()
        plt.tight_layout()
        # plt.show()
        plt.savefig(os.path.join(outdir, 'guide', str(cluster_id) + '.png'))
        plt.clf()
