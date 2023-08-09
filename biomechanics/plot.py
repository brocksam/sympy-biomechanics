import matplotlib.pyplot as plt


def plot_config(x, y, z):

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
    ax_top.set_xlim((-1.0, 1.0))
    ax_top.set_ylim((-1.0, 1.0))
    ax_top.set_title('Top View')
    ax_top.set_xlabel('x')
    ax_top.set_ylabel('y')
    ax_top.set_aspect('equal')

    # 3d view
    lines_3d, = ax_3d.plot(x, y, z, **line_prop)
    ax_3d.set_xlim((-1.0, 1.0))
    ax_3d.set_ylim((-1.0, 1.0))
    ax_3d.set_zlim((-1.0, 1.0))
    ax_3d.set_xlabel('x')
    ax_3d.set_ylabel('y')
    ax_3d.set_zlabel('z')

    # front view
    lines_front, = ax_front.plot(y, z, **line_prop)
    ax_front.set_xlim((-1.0, 1.0))
    ax_front.set_ylim((-1.0, 1.0))
    ax_front.set_title('Front View')
    ax_front.set_xlabel('y')
    ax_front.set_ylabel('z')
    ax_front.set_aspect('equal')

    # right view
    lines_right, = ax_right.plot(x, z, **line_prop)
    ax_right.set_xlim((-1.0, 1.0))
    ax_right.set_ylim((-1.0, 1.0))
    ax_right.set_title('Right View')
    ax_right.set_xlabel('x')
    ax_right.set_ylabel('z')
    ax_right.set_aspect('equal')

    fig.tight_layout()
