import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from utils import energy, entmax, normmax_bisect

def main():


    D = 2
    N = 5
    # temp = 0.1
    temp = .1
    normalize = True
    n_samples = 100
    thresh = 0.001

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
            print(torch.sqrt(torch.sum(X*X, dim=1)))
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

        for k, alpha in enumerate([1, 1.5, 2, "normmax2", "normmax5"]):
            num_iters = 50

            # X is n by d. Xi is m by d.

            Xi = Q
            for _ in range(num_iters):
                if "normmax" in str(alpha):
                    p = normmax_bisect(Xi @ X.T / temp, alpha=int(alpha[-1]), dim=-1)
                else:
                    p = entmax(Xi @ X.T / temp, alpha=alpha, dim=-1)
                Xi = p @ X

            dists = torch.cdist(Xi, X)

            response = torch.zeros_like(dists[:, 0])
            for pp in range(len(X)):
                response[dists[:, pp] < thresh] = pp+1

            cols = ['w', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5'][:len(X)+1]
            cmap = matplotlib.colors.ListedColormap(cols)

            for pp in range(len(X)):
                response = response.reshape(*mesh_x.shape)
                axes[i,k].pcolormesh(mesh_x, mesh_y, response,
                                     vmin=0, vmax=len(X)+1,
                                     cmap=cmap)

                # axes[i,k].contourf(mesh_x, mesh_y, response,
                                   # levels=np.array([0, 1, 2, 3, 4, 5]) - .1,
                                   # colors=['w', 'C0', 'C1', 'C2', 'C3', 'C4',
                                   # 'C5']
                                   # )


        for ax in axes[i]:
            for pp in range(len(X)):
                ax.plot(X[pp, 0], X[pp, 1],
                        's',
                        markerfacecolor=f'C{pp}',
                        markeredgecolor='k',
                        markeredgewidth=1,
                        markersize=5)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_xticks(())
            ax.set_yticks(())
    plt.show()
    axes[0,0].set_title("$1$-entmax")
    axes[0,1].set_title("$1.5$-entmax")
    axes[0,2].set_title("$2$-entmax")
    axes[0,3].set_title("$2$-normmax")
    axes[0,4].set_title("$5$-normmax")
    plt.savefig("basins.png")


if __name__ == '__main__':
    main()



