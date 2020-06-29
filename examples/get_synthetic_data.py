import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from random import seed, shuffle, sample
from scipy.stats import multivariate_normal
from scipy.stats import norm as univariate_normal

plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
plt.rc('font',**{'family':'serif','serif':['Palatino']})
plt.rcParams['text.usetex'] = True

SMALL_SIZE = 37
MEDIUM_SIZE = 40
BIGGER_SIZE = 45

MARKER_SIZE = 200
CROSS_SIZE = 3000

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE, labelsize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

def get_gaussian_data(n_samples=None, plot_data=False):
    """
        Code for generating the synthetic data.
        We will have two non-sensitive features and one sensitive feature.
        A sensitive feature value of -1 means the example is considered to be in protected group (e.g., female) and 1.0 means it's in non-protected group (e.g., male).
    """
    if n_samples is None:
        n_samples = 1000  # generate these many data points per class

    def gen_gaussian(mean_in, cov_in, class_label, sens_label, samples):
        nv = multivariate_normal(mean=mean_in, cov=cov_in)
        X = nv.rvs(samples)
        y = np.ones(samples, dtype=float) * class_label
        s = np.ones(samples, dtype=float) * sens_label
        return nv, X, y, s

    def gen_uni_gaussian(mean, std):
        nv = univariate_normal(loc=mean, scale=std)
        return nv

    """ Generate the non-sensitive features randomly """
    # We will generate one gaussian cluster for each class
    mu1, sigma1 = [2, -2], [[1, 0], [0, 1]]  # negative class, negative sens attr (protected)
    mu2, sigma2 = [4.5, -1.5], [[1, 0], [0, 1]]  # negative class, positive sens attr (unprotected)

    mu3, sigma3 = [3, -1], [[1, 0], [0, 1]]  # positive class, positive sens attr
    mu32, sigma32 = [1, 4], [[0.5, 0], [0, 0.5]]  # positive class, positive sens attr
    mu4, sigma4 = [2.5, 2.5], [[1, 0], [0, 1]] # positive class, negative sens attr

    nmb_1, nmb_2 = int(np.floor(n_samples/ 4)), int(np.floor(n_samples/ 8))

    gap = n_samples - 3 * nmb_1 - 2 * nmb_2

    nv1, X1, y1, s1 = gen_gaussian(mu1, sigma1, -1, -1,  nmb_1)  # negative class, negative sens attr
    nv2, X2, y2, s2 = gen_gaussian(mu2, sigma2, -1, 1, nmb_1)  # negative class, positive sens attr

    nv3, X3, y3, s3 = gen_gaussian(mu3, sigma3, 1, -1, nmb_2)  # positive class, negative sens attr
    nv32, X32, y32, s32 = gen_gaussian(mu32, sigma32, 1, -1, nmb_2)  # positive class, negative sens attr
    X3 = np.vstack((X3, X32))
    y3 = np.hstack((y3, y32))
    s3 = np.hstack((s3, s32))

    nv4, X4, y4, s4 = gen_gaussian(mu4, sigma4, 1, 1, nmb_1+gap)  # positive class, positive sens attr


    # join the posisitve and negative class clusters
    x_data = np.vstack((X1, X2, X3, X4))
    y_data = np.hstack((y1, y2, y3, y4))
    s_data = np.hstack((s1, s2, s3, s4))

    # shuffle the data
    perm = range(0, n_samples)
    perm = sample(perm, n_samples)
    x_data = x_data[perm]
    y_data = y_data[perm]
    s_data = s_data[perm]

    if plot_data:
        plot_data_(x_data, y_data, s_data, num_to_draw=400)

    return x_data, y_data, s_data


def plot_data_(x_data,y_data,s_data, num_to_draw=False):
    if not num_to_draw:
        num_to_draw = x_data.shape[0]  # we will only draw a small number of points to avoid clutter

    cmap_points = "coolwarm_r"
    fig, ax = plt.subplots(figsize=(10,10))

    x_draw = x_data[:num_to_draw]
    y_draw = y_data[:num_to_draw]
    s_draw = s_data[:num_to_draw]

    X_s_0 = x_draw[s_draw == -1.0]
    X_s_1 = x_draw[s_draw == 1.0]
    y_s_0 = y_draw[s_draw == -1.0]
    y_s_1 = y_draw[s_draw == 1.0]

    ax.scatter(X_s_0[y_s_0 == 1.0][:, 0], X_s_0[y_s_0 == 1.0][:, 1], c=np.ones_like(y_s_0[y_s_0 == 1.0]), cmap=cmap_points, alpha=0.9, marker='+', s=MARKER_SIZE, linewidth=2.5, norm=MidpointNormalize(midpoint=0,vmax=1,vmin=-1))
    ax.scatter(X_s_0[y_s_0 == -1.0][:, 0], X_s_0[y_s_0 == -1.0][:, 1], c=np.ones_like(y_s_0[y_s_0 == -1.0]), cmap=cmap_points, alpha=0.9,marker='_', s=MARKER_SIZE, linewidth=2.5, norm=MidpointNormalize(midpoint=0,vmax=1, vmin=-1))
    ax.scatter(X_s_1[y_s_1 == 1.0][:, 0], X_s_1[y_s_1 == 1.0][:, 1], c=-1*np.ones_like(y_s_1[y_s_1 == 1.0]), cmap=cmap_points, alpha=0.9,marker='+', s=MARKER_SIZE, linewidth=2.5, norm=MidpointNormalize(midpoint=0,vmax=1,vmin=-1))
    ax.scatter(X_s_1[y_s_1 == -1.0][:, 0], X_s_1[y_s_1 == -1.0][:, 1], c=-1*np.ones_like(y_s_1[y_s_1 == -1.0]), cmap=cmap_points,alpha=0.9, marker='_', s=MARKER_SIZE, linewidth=2.5, norm=MidpointNormalize(midpoint=0,vmax=1,vmin=-1))

    plt.locator_params(nbins=5)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    plt.show()


def plot_boundaries(model, x_data, y_data, s_data, num_to_draw=False):
    if not num_to_draw:
        num_to_draw = x_data.shape[0]  # we will only draw a small number of points to avoid clutter

    cmap_points = "coolwarm_r"
    cmap_contour = plt.get_cmap('RdYlGn')
    fig, ax = plt.subplots(figsize=(10,10))

    x_draw = x_data[:num_to_draw]
    y_draw = y_data[:num_to_draw]
    s_draw = s_data[:num_to_draw]

    X_s_0 = x_draw[s_draw == -1.0]
    X_s_1 = x_draw[s_draw == 1.0]
    y_s_0 = y_draw[s_draw == -1.0]
    y_s_1 = y_draw[s_draw == 1.0]

    ax.scatter(X_s_0[y_s_0 == 1.0][:, 0], X_s_0[y_s_0 == 1.0][:, 1], c=np.ones_like(y_s_0[y_s_0 == 1.0]), cmap=cmap_points, alpha=1, marker='+', s=MARKER_SIZE, linewidth=2.5, norm=MidpointNormalize(midpoint=0,vmax=1,vmin=-1))
    ax.scatter(X_s_0[y_s_0 == -1.0][:, 0], X_s_0[y_s_0 == -1.0][:, 1], c=np.ones_like(y_s_0[y_s_0 == -1.0]), cmap=cmap_points, alpha=1,marker='_', s=MARKER_SIZE, linewidth=2.5, norm=MidpointNormalize(midpoint=0,vmax=1, vmin=-1))
    ax.scatter(X_s_1[y_s_1 == 1.0][:, 0], X_s_1[y_s_1 == 1.0][:, 1], c=-1*np.ones_like(y_s_1[y_s_1 == 1.0]), cmap=cmap_points, alpha=1,marker='+', s=MARKER_SIZE, linewidth=2.5, norm=MidpointNormalize(midpoint=0,vmax=1,vmin=-1))
    ax.scatter(X_s_1[y_s_1 == -1.0][:, 0], X_s_1[y_s_1 == -1.0][:, 1], c=-1*np.ones_like(y_s_1[y_s_1 == -1.0]), cmap=cmap_points,alpha=1, marker='_', s=MARKER_SIZE, linewidth=2.5, norm=MidpointNormalize(midpoint=0,vmax=1,vmin=-1))

    # Create a mesh of points to plot in
    # h: Stepsize of meshgrid
    h = 0.2
    x_min, x_max = x_draw[:,0].min() - 1, x_draw[:,0].max() + 1
    y_min, y_max = x_draw[:,1].min() - 1, x_draw[:,1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    #Plot the decision boundaries for a classifier.
    zz = np.sign(model.predict(np.c_[xx.ravel(), yy.ravel()]))
    zz = zz.reshape(xx.shape)
    ax.contourf(xx, yy, zz, alpha=0.3, cmap=cmap_contour, norm=MidpointNormalize(midpoint=0))

    plt.locator_params(nbins=5)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    plt.show()

# set the colormap and centre the colorbar
class MidpointNormalize(mcolors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mcolors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


if __name__ == "__main__":

    n = 600
    x_train, y_train, s_train = get_gaussian_data(n_samples=n, plot_data=True)
