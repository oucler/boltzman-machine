# Import python modules
import numpy as np
import matplotlib.pyplot as plt
from dataset import MNISTLoader


def tick_params():
    """Tick params used in `plt.tick_params` or `im.axes.tick_params` to
    plot images without labels, borders etc.
    """
    return dict(axis='both', which='both',
                bottom='off', top='off', left='off', right='off',
                labelbottom='off', labelleft='off', labelright='off')


def im_plot(X, n_width=10, n_height=10, shape=None, title=None,
            title_params=None, imshow_params=None):
    """Plot batch of images `X` on a single graph."""
    # check params
    X = np.asarray(X)
    if shape is None:
        shape = X.shape[1:]

    title_params = title_params or {}
    title_params.setdefault('fontsize', 22)
    title_params.setdefault('y', 0.95)

    imshow_params = imshow_params or {}
    imshow_params.setdefault('interpolation', 'nearest')

    # plot
    for i in range(n_height * n_width):
        if i < len(X):
            img = X[i]
            if shape is not None:
                img = img.reshape(shape)
            ax = plt.subplot(n_height, n_width, i + 1)
            for d in ('bottom', 'top', 'left', 'right'):
                ax.spines[d].set_linewidth(2.)
            plt.tick_params(**tick_params())
            plt.imshow(img, **imshow_params)
    if title:
        plt.suptitle(title, **title_params)
    plt.subplots_adjust(wspace=0, hspace=0)


if __name__ == "__main__":
    loader = MNISTLoader(path='../data/')
    X = loader.training_data(loader.training_path_dir)
    fig = plt.figure(figsize=(10, 10))
    im_plot(X[:100], shape=(28, 28), title='Training examples',
            imshow_params={'cmap': plt.cm.gray})
    plt.savefig('mnist.png', dpi=196, bbox_inches='tight')