import torch
import numpy as np
import matplotlib.pyplot as plt

from utils import energy, entmax, normmax_bisect

def main():

    D = 2
    N = 5
    temp = 1
    normalize = True
    n_samples = 20

    torch.random.manual_seed(42)

    which = [0, 3, 8, 9]
    nplots = len(which)

    fig, axes = plt.subplots(nplots, 5, figsize=(10, 6),
                             constrained_layout=True)

    patterns = []
    queries = []

    for i in range(10):
        patterns.append(torch.randn(N, D, dtype=torch.float64))
        queries.append(torch.randn(D, 1, dtype=torch.float64))
    
    patterns[0] = torch.tensor([
        [-1, -1],
        [-1, +1],
        [+1, -1]], dtype=torch.float64)
    queries[0].zero_()


    for i in range(nplots):
        ii = which[i]
        X = patterns[ii]
        
        query = queries[ii]
    
        if normalize:
            #print(torch.sqrt(torch.sum(X*X, dim=1)))
            X = X / torch.sqrt(torch.sum(X*X, dim=1)).unsqueeze(1)

        xmin, xmax = X[:, 0].min(), X[:, 0].max()
        ymin, ymax = X[:, 1].min(), X[:, 1].max()

        xmin -= .1
        ymin -= .1
        xmax += .1
        ymax += .1

        xx = np.linspace(xmin, xmax, n_samples)
        yy = np.linspace(ymin, ymax, n_samples)

        mesh_x, mesh_y = np.meshgrid(xx, yy)

        Q = np.column_stack([mesh_x.ravel(), mesh_y.ravel()])
        Q = torch.from_numpy(Q)

        # cmap = 'OrRd_r'
        cmap = 'viridis'
        E1 = energy(Q, X, alpha=1, beta=1/temp).reshape(*mesh_x.shape)
        axes[i,0].contourf(mesh_x, mesh_y, E1, cmap=cmap)
        E15 = energy(Q, X, alpha=1.5, beta=1/temp).reshape(*mesh_x.shape)
        axes[i,1].contourf(mesh_x, mesh_y, E15, cmap=cmap)
        E2 = energy(Q, X, alpha=2, beta=1/temp).reshape(*mesh_x.shape)
        axes[i,2].contourf(mesh_x, mesh_y, E2, cmap=cmap)
        E3 = energy(Q, X, alpha=2, beta=1/temp, normmax=True).reshape(*mesh_x.shape)
        axes[i,3].contourf(mesh_x, mesh_y, E3, cmap=cmap)
        E4 = energy(Q, X, alpha=5, beta=1/temp, normmax=True).reshape(*mesh_x.shape)
        axes[i,4].contourf(mesh_x, mesh_y, E4, cmap=cmap)
        p = torch.softmax(X.mm(query), dim=0)
        query = X.T @ p

        for k, alpha in enumerate([1, 1.5, 2, "normmax2", "normmax5"]):
            num_iters = 1000
            xis = np.zeros((num_iters, D))
            xi = query
            for j in range(num_iters):
                xis[j, :] = xi[:, 0]
                if "normmax" in str(alpha):
                    p = normmax_bisect(X.mm(xi)/temp, alpha=int(alpha[-1]), dim=0)
                else:
                    p = entmax(X.mm(xi)/temp, alpha, dim=0)
                xi = X.T.mm(p)
            
            first_point = xis[0]

# Plot a marker at the first point to represent the circumference
            axes[i, k].scatter(first_point[0], first_point[1], marker='o',facecolors='none', s=75, edgecolors='C1', linewidths=1.5, label='$q_0$')
            axes[i, k].plot(xis[0:, 0], xis[0:, 1],
                            lw=2,
                            marker='.',
                            color='C1',
                            label='$q_t$')

        for ax in axes[i]:
            
            ax.plot(X[:, 0], X[:, 1], 's', markerfacecolor='w',
                    markeredgecolor='k', markeredgewidth=1, markersize=5,
                    label='$x_i$')
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            # ax.set_xlim(xmin-.2, xmax+.2)
            # ax.set_ylim(ymin-.2, ymax+.2)
            ax.set_xticks(())
            ax.set_yticks(())


    axes[0,0].set_title("$1$-entmax")
    axes[0,1].set_title("$1.5$-entmax")
    axes[0,2].set_title("$2$-entmax")
    axes[0,3].set_title("$2$-normmax")
    axes[0,4].set_title("$5$-normmax")
    axes[0, 0].legend()

    plt.savefig("contours.pdf")
    plt.show()


if __name__ == '__main__':
    main()

