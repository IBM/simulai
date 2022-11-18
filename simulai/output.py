# (C) Copyright IBM Corp. 2019, 2020, 2021, 2022.

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#           http://www.apache.org/licenses/LICENSE-2.0

#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
import numpy as np
def plot_figures2D(u,t,x,z,nt,nx,nz,output_name):
    """

            :param u np.ndarray: array with data to plot
            :param t np.ndarray: uni-dimensional array with timestamps
            :param x np.ndarray: uni-dimensional array with x samples
            :param z np.ndarray: uni-dimensional array with z samples
            :param nt int: # of time samples
            :param nx int: # of x samples
            :param nz int: # of z samples
            :param output_name str :  name of final output.py
            :return:

    """


    tb = np.array([t.min(), t.max()])
    xb  = np.array([x.min(),x.max()])
    zb  = np.array([z.min(),z.max()])

    U = u.reshape(nt,nx,nz)


    data2 = U[0]

    data2.shape = -1, 1

    data2 = data2.reshape((nx, nz))

    fig = plt.figure()
    ax = plt.axes()

    im = ax.imshow(data2.T, interpolation='nearest', aspect='auto')

    xformatter = mpl.ticker.FuncFormatter(MeshFormatterHelper(xb[0], (xb[1]-xb[0]) / nx))
    ax.xaxis.set_major_formatter(xformatter)

    zformatter = mpl.ticker.FuncFormatter(MeshFormatterHelper(zb[0], (zb[1]-zb[0]) / nz))
    ax.yaxis.set_major_formatter(zformatter)

    plt.sci(im)
    cbar = plt.colorbar()

    plt.title('{0:4}/{1}'.format(0, len(data2) - 1))

    def _animate(i, im, data, cbar):
        data2 = data[i]

        data2.shape = -1, 1

        data2 = data2.reshape((nx, nz))

        im = im.set_data(data2.T)

        clim = (data2.min(), data2.max())

        cbar.mappable.set_clim(clim)

        return im,

    _animate_args = (im, U, cbar)

    time = tb[1]-tb[0]

    display_rate = int(nt / (30 * time)) if int(nt / (30 * time)) > 0 else 1

    anim = animation.FuncAnimation(fig, _animate, fargs=_animate_args, frames=range(0, len(U), display_rate),
                                   blit=False)

    anim.save(output_name, fps=30, extra_args=['-vcodec', 'libx264'])

    return

class MeshFormatterHelper(object):
    def __init__(self, lbound, delta):
        self.lbound = lbound
        self.delta = delta

    def __call__(self, grid_point, pos):
        return '{0:.3}'.format(self.lbound + self.delta * grid_point)

