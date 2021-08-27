import matplotlib
from matplotlib import pyplot
import os

COLORS = [
    'purple',
    'wheat',
    'maroon',
    'red',
    'powderblue',
    'dodgerblue',
    'magenta',
    'tan',
    'aqua',
    'yellow',
    'slategray',
    'blue',
    'rosybrown',
    'violet',
    'lightseagreen',
    'pink',
    'darkorange',
    'teal',
    'royalblue',
    'lawngreen',
    'gold',
    'navy',
    'darkgreen',
    'deeppink',
    'palegreen',
    'silver',
    'saddlebrown',
    'plum',
    'peru',
    'black',
]

assert (len(COLORS) == len(set(COLORS)))

def visualize_generated(fake, real, y, it, outdir):
    pyplot.plot(real[:, 0], real[:, 1], 'r.')
    pyplot.plot(fake[:, 0], fake[:, 1], 'b.')
    
    # plotted_labels = set()
    # for i in range(len(y)):
    #     if int(y[i]) not in plotted_labels:
    #         pyplot.text(real[i, 0], real[i, 1], '  ->'+str(int(y[i])))
    #         plotted_labels.add(int(y[i]))
    # print('plotted_labels', plotted_labels, 'num:', len(plotted_labels), 'num points:', len(y))
    fontdict = {'fontsize': 18,
        'fontweight' : 1.0,
        'verticalalignment': 'baseline',
        'horizontalalignment': 'center'}
    pyplot.xlabel('X', fontdict=fontdict)
    pyplot.ylabel('Y', fontdict=fontdict)
    pyplot.xticks([])
    pyplot.yticks([])

    pyplot.savefig(os.path.join(outdir, 'all', str(it) + '.png'),
                   dpi=100,
                   bbox_inches='tight')
    pyplot.clf()

    # lim = 6
    lim = 200 + 10
    axes = pyplot.gca()
    axes.set_aspect('equal', adjustable='box')
    axes.set_xlim([-lim, lim])
    axes.set_ylim([-lim, lim])

    pyplot.locator_params(nbins=4)
    pyplot.tight_layout()

    pyplot.plot(fake[:, 0], fake[:, 1], 'b.', alpha=0.1)
    pyplot.savefig(os.path.join(outdir, 'all',
                                str(it) + 'square.png'),
                   dpi=100,
                   bbox_inches='tight')
    pyplot.clf()


def visualize_clusters(x, y, it, outdir):
    y = y.detach().cpu().numpy()
    for i in range(y.max()):
        pyplot.plot(x[y == i, 0],
                    x[y == i, 1],
                    '.',
                    color=COLORS[i % len(COLORS)])
    pyplot.savefig(os.path.join(outdir, 'clusters', str(it) + '.png'))
    pyplot.clf()
