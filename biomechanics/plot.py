import sympy as sm
import matplotlib.pyplot as plt


def plot_traj(t, x, syms):
    """Simple plot of state trajectories.

    Parameters
    ==========
    t : array_like, shape(n,)
        Time values.
    x : array_like, shape(n, m)
        State values at each time value.
    syms : sequence of Symbol, len(m)
        SymPy symbols associated with state.

    """
    num_rows = 8
    num_cols = (x.shape[1] // num_rows)
    if x.shape[1] % num_rows > 0:
        num_cols += 1

    fig, axes = plt.subplots(num_rows, num_cols, sharex=True)

    for ax, traj, sym in zip(axes.T.flatten(), x.T, syms):
        ax.plot(t, traj)
        ax.set_ylabel(sm.latex(sym, mode='inline'))

    # label the x axis only on the bottom row.
    for ax in axes[-1, :]:
        ax.set_xlabel('Time [s]')

    fig.tight_layout()

    return axes


def plot_config(x, y, z, xlim=(-0.5, 0.5), ylim=(-0.5, 0.5), zlim=(-0.5, 0.5)):

    # create a figure
    fig = plt.figure()
    fig.set_size_inches((10.0, 10.0))

    # setup the subplots
    ax_top = fig.add_subplot(2, 2, 1)
    ax_3d = fig.add_subplot(2, 2, 2, projection='3d')
    ax_front = fig.add_subplot(2, 2, 3)
    ax_right = fig.add_subplot(2, 2, 4)

    # common line and marker properties for each panel
    line_prop = {
        'color': 'black',
        'marker': 'o',
        'markerfacecolor': 'blue',
        'markersize': 4,
    }

    # top view
    lines_top, = ax_top.plot(x, y, **line_prop)
    ax_top.set_xlim(xlim)
    ax_top.set_ylim(ylim)
    ax_top.set_title('Top View')
    ax_top.set_xlabel('x')
    ax_top.set_ylabel('y')
    ax_top.set_aspect('equal')

    # 3d view
    lines_3d, = ax_3d.plot(x, y, z, **line_prop)
    ax_3d.set_xlim(xlim)
    ax_3d.set_ylim(ylim)
    ax_3d.set_zlim(zlim)
    ax_3d.set_xlabel('x')
    ax_3d.set_ylabel('y')
    ax_3d.set_zlabel('z')

    # front view
    lines_front, = ax_front.plot(y, z, **line_prop)
    ax_front.set_xlim(ylim)
    ax_front.set_ylim(zlim)
    ax_front.set_title('Front View')
    ax_front.set_xlabel('y')
    ax_front.set_ylabel('z')
    ax_front.set_aspect('equal')

    # right view
    lines_right, = ax_right.plot(x, z, **line_prop)
    ax_right.set_xlim(xlim)
    ax_right.set_ylim(zlim)
    ax_right.set_title('Right View')
    ax_right.set_xlabel('x')
    ax_right.set_ylabel('z')
    ax_right.set_aspect('equal')

    fig.tight_layout()

    return fig, lines_top, lines_3d, lines_front, lines_right
