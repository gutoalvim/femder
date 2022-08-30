"""
Data visualization module.

This plot module receives organized raw data (numpy arrays, lists and strings) and plot it through its functions.

For further information check the function specific documentation.
"""
import os
import sys
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import ticker, gridspec
from IPython.core.getipython import get_ipython

import plotly.graph_objects as go
import plotly.io as pio
import plotly.figure_factory as ff
import plotly.graph_objs as go



def set_plotly_renderer():
    """
    Automatic Plotly renderer setup.
    """
    import re
    import psutil
    if get_ipython().__class__.__name__ == "ZMQInteractiveShell":
        if any(re.search('jupyter-notebook', x)
               for x in psutil.Process().parent().cmdline()):
            pio.renderers.default = "notebook"
        else:
            pio.renderers.default = "jupyterlab"

    elif 'google.colab' in sys.modules:
        pio.renderers.default = "colab"
    else:
        pio.renderers.default = "browser"

    print(f"Default Plotly renderer: {pio.renderers.default}")


def plot_2d_freq(x, y, xlabel=None, ylabel=None, save_fig=False, xlim=None, ylim=None, title=None, legend_list=[],
                 linestyle_list=None, project_folder=None):
    plt.style.use('seaborn-colorblind')

    fig = plt.figure(figsize=(16, 9))
    ax1 = None
    gs = gridspec.GridSpec(1, 1)
    base_fontsize = 18

    if isinstance(y, list):
        y = np.asarray(y).squeeze()

    if linestyle_list is None:
        linestyle_list = ["-"]

    freq = x
    mag = y.T

    ax1 = plt.subplot(gs[0, 0])
    ax1.semilogx(freq,
                 mag,
                 linewidth=4)
    if ylim is None:
        ax1.set_ylim([np.amin(mag) - np.amin(mag) * 0.1,
                      np.amax(mag) + np.amin(mag) * 0.1])
    else:
        ax1.set_ylim(ylim)
    ax1.set_xlim([min(freq), max(freq)])
    ax1.set_xlabel(xlabel, fontsize=base_fontsize - 1)
    ax1.set_ylabel(ylabel, fontsize=base_fontsize - 1)
    ax1.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax1.get_xaxis().set_minor_formatter(ticker.ScalarFormatter())
    ax1.tick_params(which="minor", length=5, rotation=-90, axis="x")
    ax1.tick_params(which="major", length=5, rotation=-90, axis="x")
    ax1.tick_params(axis="both", which="both", labelsize=base_fontsize - 2)
    ax1.minorticks_on()
    ax1.legend(legend_list, prop={'size': 18})
    for line, ls in zip(ax1.get_lines(), linestyle_list):
        line.set_linestyle(ls)
    ax1.grid("minor")
    ax1.set_title(title,
                  size=base_fontsize + 2,
                  fontweight="bold",
                  loc="left")

    gs.tight_layout(fig, pad=2)
    if save_fig:
        fig.savefig(project_folder + f'\\{title}.png',
                    dpi=300,
                    transparent=True,
                    bbox_inches='tight')

    return fig




def remove_bg_and_axis(fig, len_scene):
    for ic in range(len_scene):
        scene_text = f'scene{ic + 1}' if ic > 0 else 'scene'
        fig.layout[scene_text]['xaxis']['showbackground'] = False
        fig.layout[scene_text]['xaxis']['visible'] = False
        fig.layout[scene_text]['yaxis']['showbackground'] = False
        fig.layout[scene_text]['yaxis']['visible'] = False
        fig.layout[scene_text]['zaxis']['showbackground'] = False
        fig.layout[scene_text]['zaxis']['visible'] = False
    return fig



def set_all_cameras(fig, len_scene, camera_dict=None, axis='z'):
    eye_dict = {'x': [0., 0., -1.75],
                'y': [0., 0., -1.75],
                'z': [-0.5, -1.5, 0],
                "iso_z": [-1.2, -1.1, 0.4]}

    up_dict = {'x': [1, 0., 0.],
               'y': [0, 1, 0.],
               'z': [0., 0., 1],
               'iso_z': [0., 0., 1]}

    if camera_dict is None:
        camera_dict = dict(eye=dict(x=eye_dict[axis][0], y=eye_dict[axis][1], z=eye_dict[axis][2]),
                           up=dict(x=up_dict[axis][0], y=up_dict[axis][1], z=up_dict[axis][2]),
                           center=dict(x=0, y=0, z=0), projection_type="perspective")

    for ic in range(len_scene):
        scene_text = f'scene{ic + 1}_camera' if ic > 0 else 'scene_camera'
        fig.layout[scene_text] = camera_dict
    return fig


def plot_mesh(vertices, elements, fig=None):
    if fig is not None:
        fig.add_trace(go.Mesh3d(x=vertices[0, :],
                                y=vertices[1, :],
                                z=vertices[2, :],
                                i=elements[0, :], j=elements[1, :], k=elements[2, :], name='Mesh'
                                ), 1, 3)

        fig['layout']['scene2'].update(go.layout.Scene(aspectmode='data'))
        fig['layout']['scene2']['camera'] = dict(eye=dict(x=3.5, y=3.5, z=3.5),
                                                 up=dict(x=3, y=0, z=0),
                                                 center=dict(x=0, y=0, z=0), projection_type="perspective")

    else:
        fig = ff.create_trisurf(
            x=vertices[0, :],
            y=vertices[1, :],
            z=vertices[2, :],
            simplices=elements.T,
            color_func=elements.shape[1] * ["rgb(255, 222, 173)"],
        )
        fig['data'][0].update(opacity=1)

        # fig['layout']['scene'].update(go.layout.Scene(aspectmode='data'))
        fig.update_layout(
            width=1600, height=800, margin=dict(l=30, r=50, b=0, t=90),
            scene=dict(
                xaxis=dict(range=[None, None]),
                yaxis=dict(range=[None, None]),
                zaxis=dict(range=[None, None]),
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="data",

            )
        )
        # fig = remove_bg_and_axis(fig, 1)
        return fig

def plot_surfaces(vertices, elements, domain_index, admittance, fig=None):
    if fig is not None:
        fig.add_trace(go.Mesh3d(x=vertices[0, :],
                                y=vertices[1, :],
                                z=vertices[2, :],
                                i=elements[0, :], j=elements[1, :], k=elements[2, :], name='Mesh'
                                ), 1, 3)

        fig['layout']['scene2'].update(go.layout.Scene(aspectmode='data'))
        fig['layout']['scene2']['camera'] = dict(eye=dict(x=3.5, y=3.5, z=3.5),
                                                 up=dict(x=3, y=0, z=0),
                                                 center=dict(x=0, y=0, z=0), projection_type="perspective")

    else:
        fig = ff.create_trisurf(
            x=vertices[0, :],
            y=vertices[1, :],
            z=vertices[2, :],
            simplices=elements.T,
            color_func=elements.shape[1] * ["rgb(255, 255, 255)"],
        )
        fig['data'][0].update(opacity=0.3)

        for bl in list(admittance.keys()):
            indx = np.argwhere(domain_index == bl).ravel()
            con = elements[:, indx]  # [con,:].T
            # con = con.T
            fig.add_trace(go.Mesh3d(
                x=vertices[0, :],
                y=vertices[1, :],
                z=vertices[2, :],
                i=con[0, :], j=con[1, :], k=con[2, :], opacity=1, showlegend=True, visible=True, name=f'PG {int(bl)}'
            ))

        # fig['layout']['scene'].update(go.layout.Scene(aspectmode='data'))
        fig.update_layout(
            width=1600, height=800, margin=dict(l=30, r=50, b=0, t=90),
            scene=dict(
                xaxis=dict(range=[None, None]),
                yaxis=dict(range=[None, None]),
                zaxis=dict(range=[None, None]),
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="data",

            )
        )
        # fig = remove_bg_and_axis(fig, 1)
        return fig

def plot_mesh_dash(vertices, elements):
    vertices = vertices.T
    elements = elements.T
    fig = ff.create_trisurf(
        x=vertices[0, :],
        y=vertices[1, :],
        z=vertices[2, :],
        simplices=elements.T,
        color_func=elements.shape[1] * ["rgb(255, 222, 173)"],
    )
    fig['layout']['scene'].update(go.layout.Scene(aspectmode='data'))
    fig.update_layout(title="")

    return fig

def plot_setup(vertices, elements, source_coords=None, receiver_coords=None):
    fig = plot_mesh_dash(vertices, elements)

    fig['data'][0].update(opacity=0.3)

    fig['layout']['scene'].update(go.layout.Scene(aspectmode='data'))

    if receiver_coords is not None:
        fig.add_trace(
            go.Scatter3d(x=receiver_coords[:, 0], y=receiver_coords[:, 1], z=receiver_coords[:, 2], name="Receivers",
                         mode='markers'))

    if source_coords is not None:
        fig.add_trace(
            go.Scatter3d(x=source_coords[:, 0], y=source_coords[:, 1], z=source_coords[:, 2], name="Sources",
                         mode='markers'))

    return fig


def plot_boundary_pressure(vertices, elements, values, intensitymode, axis=None,):
    # fig = plot_mesh_dash(vertices, elements)
    # fig['data'][0].update(opacity=0.01)
    fig = go.Figure()
    values = np.real(values)
    # fig['layout']['scene'].update(go.layout.Scene(aspectmode='data'))
    fig.add_trace(go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=elements[:, 0],
        j=elements[:, 1],
        k=elements[:, 2],
        intensity=values,
        colorscale='Jet',
        intensitymode=intensitymode,

    ))
    fig['layout']['scene'].update(go.layout.Scene(aspectmode='data'))

    # fig.update_layout(
    #     width=1600, height=800, margin=dict(l=30, r=50, b=0, t=90),
    #     scene=dict(
    #         xaxis=dict(range=[None, None]),
    #         yaxis=dict(range=[None, None]),
    #         zaxis=dict(range=[None, None]),
    #         xaxis_title="X",
    #         yaxis_title="Y",
    #         zaxis_title="Z",
    #         aspectmode="data",
    #
    #     )
    # )

    fig = remove_bg_and_axis(fig, 1)
    if axis is not None:
        fig = set_all_cameras(fig, 1, axis=axis)
    # plotly.offline.iplot(fig)
    return fig



def plot_pressure_field(plane_data, pressure_lim, source_index, axis, freq, fig=None):

    colorbar_dict = {'title': 'Pressure [Pa]',
                     'titlefont': {'color': 'black'},
                     'title_side': 'right',
                     'tickangle': -90,
                     'tickcolor': 'black',
                     'tickfont': {'color': 'black'},
                     'x': -0.1}
    if fig is None:
        fig = go.Figure()

    for freq_i in range(len(freq)):
        for i in plane_data.keys():
            if i in axis:
                fig.add_trace(go.Mesh3d(x=plane_data[i]["vertices"][:,0], y=plane_data[i]["vertices"][:,1], z=plane_data[i]["vertices"][:,2],
                                        i=plane_data[i]["elements"][:,0], j=plane_data[i]["elements"][:,1], k=plane_data[i]["elements"][:,2],
                                        intensity=plane_data[i]["pressure"][freq_i, source_index, :],
                                        colorscale='jet', intensitymode='vertex', name=i, showlegend=True,
                                        visible=True, opacity=1,
                                        showscale=True, colorbar=colorbar_dict))

    fig.data[0].visible = True
    steps = []
    for i in range(len(freq)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(freq) + [None] * (len(fig.data) - len(freq))},
                  {"title": f'Frequency {freq[i] :.2f} Hz'}],  # layout attribute
            label=f"{freq[i] :.1f} Hz"
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=10,
        currentvalue={"prefix": "Frequency: ", "suffix": f" [Hz]"},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders
    )
    # fig = polar_axis_3d(fig, np.amax(balloon_data["intensity"][-1]))
    fig = remove_bg_and_axis(fig, 1)
    # fig = set_all_cameras(fig, 1, axis=axis)

    return fig



def layout_update_freq_plot(fig, xlabel= "Frequência [Hz]", ylabel= "NPS [dB]", width=920, height=600, legend_x=0.22, legend_y=0.22, font_size=20, font_size_legend=25,
                            tickangle=25, margin_l=0, margin_r=5, margin_b=50, margin_t=5, legend_orientation="v"):

    assert fig is not None

    fig.update_layout(

        font=dict(
            size=font_size,
        )
    )

    fig.update_layout(
        width=width, height=height,
        yaxis=dict(
            title=ylabel,
        ),
        xaxis=dict(
            title=xlabel,
        ),
        legend=dict(
            x=legend_x,
            y=legend_y,
            traceorder="normal",
            itemsizing='trace',
            orientation=legend_orientation,
            font=dict(
                family="sans-serif",
                size=font_size_legend,
                color="black"
            ),
        )
    )

    fig.update_xaxes(tickangle=tickangle)
    fig.update_layout(
        title="",
        margin=dict(l=margin_l, r=margin_r, b=margin_b, t=margin_t),
    )

    return fig
def freq_response_plotly(x_list, y_list, labels=None, visible=None, hover_data=None, linewidth=3, linestyle=None,
                         colors=None, alpha=1, mode="trace", fig=None, xlim=None, ylim=None, update_layout=True,
                         fig_size=(1200, 720), show_fig=True, save_fig=False, folder_path=None, ticks = None,
                         folder_name="Frequency Response", filename="freq_response", title='Frequency Response',
                         ylabel='Performance Metrics [-]', **kwargs):
    """
    Plot frequency response curves with Plotly.

    Parameters
    ----------
    x_list : list
        List of X values.
    y_list : list
        List of Y values.
    labels : list, optional
        List of curve labels as strings.
    visible : list, optional
        List of strings and/or booleans with visibility option for each curve.
    hover_data : list, optional
        List of strings that will appear on mouse hover.
    linewidth : int, optional
        Line width.
    linestyle : string, optional
        Line type description.
    colors : list or str, optional
        RGBA stings with color values.
    alpha : float, optional
        Line transparency.
    mode : str, optional
        String representing the plot style - 'trace' or 'fill_between'.
    fig : class or None, optional
        Plotly figure object that the points will be added to. If 'None' a new Plotly figure object will be created.
    xlim : tuple or list, optional
        Minimum and maximum limits of the X axis.
    ylim : tuple or list, optional
        Minimum and maximum limits of the Y axis.
    update_layout : bool, optional
        Option to update the figure layout or not.
    fig_size : tuple or list, optional
        Width and height of the figure.
    show_fig : bool, optional
        Option to display or not the figure once the plot is finished.
    save_fig : bool, optional
        Option to save a static image.
    folder_path : str, optional
        String containing the destination folder in which the image will be saved. If not defined the current active
        folder will be used.
    folder_name : str, optional
        String containing the name of the new folder that will be created in the destination folder in which the images
        will be saved. If not defined "backend" will be used as the new folder name.
    filename : str
        String that will be used as the partial file name.
    kwargs : keyword arguments, optional
        See rasta._plot.save_plot_fig.

    Returns
    -------
    Plotly figure object.
    """

    if fig is None:
        fig = go.Figure()
    if labels is None:
        labels = [f"Curve {i}" for i in range(len(x_list))]
    if visible is None:
        visible = [True for _ in range(len(x_list))]
    if colors is None:
        colors = 20*['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

        # colors = [f"rgb({col[0] * 255:.0f}, {col[1] * 255:.0f}, {col[2] * 255:.0f})" \
        #           for col in seaborn.color_palette('deep', len(y_list) + len(fig.data))][len(fig.data)::]

    if linestyle is None:
        linestyle = ['solid', 'dot', 'dashdot', "longdash", "dash", "longdashdot"]
        linestyle = linestyle * 20
    if mode == "trace":
        for i in range(len(x_list)):
            fig.add_trace(go.Scatter(x=x_list[i], y=y_list[i],
                                     name=labels[i],
                                     line=dict(width=linewidth, dash=linestyle[i], color=colors[i]),
                                     opacity=alpha, mode="lines",
                                     visible=visible[i],
                                     hovertext=f'Std. Dev.: {np.std(y_list[i]):0.2f} [dB]' + hover_data[
                                         i] if hover_data is not None
                                     else f'Std. Dev.: {np.std(y_list[i]):0.2f} [dB]',
                                     **kwargs
                                     ))
    elif mode == "fill_between":
        fig.add_trace(go.Scatter(x=x_list[0], y=np.min(y_list, axis=2)[0],
                                 visible="legendonly",
                                 showlegend=False,
                                 legendgroup=labels[0],
                                 line=dict(color=colors),
                                 hovertext=labels[0]
                                 ))

        fig.add_trace(go.Scatter(x=x_list[0], y=np.max(y_list, axis=2)[0],
                                 visible="legendonly",
                                 legendgroup=labels[0],
                                 name=labels[0],
                                 fillcolor=colors,
                                 line=dict(color=colors),
                                 fill='tonexty',
                                 hovertext=labels[0]
                                 ))

    if update_layout:
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black', gridcolor='grey', mirror=True)
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black', gridcolor='grey', mirror=True)
        fig.update_layout(title=dict(
            text=f"<b>{title}</b>",
            xanchor="left",
            xref="paper",
            x=0,
            yanchor="bottom",
            yref="paper",
            y=1,
            pad=dict(b=6, l=0, r=0, t=0)
        ),
            xaxis=dict(
                title="Frequency [Hz]",
                type="log",
                tickmode = 'array',
                tickvals = ticks,
                ticktext = ticks,
                range=[np.log10(xlim[0]), np.log10(xlim[1])] if xlim is not None else None,
                tickformat=".0f",
            ),
            yaxis=dict(
                title=ylabel,
                type="linear",
                range=[ylim[0], ylim[1]] if ylim is not None else None,
                tickformat=".1f",
            ),
            legend=dict(
                borderwidth=1,
                orientation="h",
                x=0.5, y=-0.2,
                xanchor="center",
                yanchor="top",
            ),
            template="seaborn",
            plot_bgcolor="rgba(0, 0, 0, 0)",
            width=fig_size[0], height=fig_size[1],
        )

    if save_fig:
        save_plotly_fig(fig, "plotly2d", filename, folder_path=folder_path, folder_name=folder_name)

    if show_fig:
        fig.show()

    return fig




def save_plotly_fig(fig, backend, filename, folder_path=None, folder_name=None, bg_transparency=True,
                    projection="perspective", scale=2, camera_angles=None, dpi=150, ax=None, ext=".png",
                    ):
    """
    Saves either a matplotlib or a Plotly figure as a static image.

    Parameters
    ----------
    fig : class
        Matplotlib or Plotly figure object.
    backend : str
        String to select backend - 'matplotlib', 'plotly2d' or 'plotly3d'.
    filename : str
        String that will be used as the full or partial file name depending on the backend.
    folder_path : str, optional
        String containing the destination folder in which the image will be saved. If not defined the current active
        folder will be used.
    folder_name : str, optional
        String containing the name of the new folder that will be created in the destination folder in which the images
        will be saved. If not defined "backend" will be used as the new folder name.
    bg_transparency : bool, option
        Boolean to save the figure with transparent background.
    projection : str, optional
        Projection type used in 3D Plotly figures - 'orthographic' or 'perspective'.
    scale : int, optional
        Scaling factor used in the Plotly images. Multiplies the Figure size to increase resolution.
    camera_angles : list, optional
        List of strings containing the camera angles that will be used to save different views of 3D Plotly images.
    dpi : int, optional
        Resolution of the Matplotlib images.
    ax : list, optional
        List containing matplotlib.axes objects. Used to set the inner background as transparent.
    ext : str, optional
        String containing the image format extension in which the figures will be saved to. Use '.png' for transparency.
    """
    if camera_angles is None:
        camera_angles = ["floorplan", "section", "diagonal_front", "diagonal_rear"]
    if folder_path is None:
        folder_path = os.getcwd() + os.sep
    if folder_name is None:
        folder_name = backend + "_figures" + os.sep
    folder_check = os.path.exists(folder_path + folder_name)
    if folder_check is False:
        os.mkdir(folder_path + folder_name)

    full_path = ""

    if backend == "plotly2d":
        if bg_transparency:
            fig.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)",
                               "paper_bgcolor": "rgba(0, 0, 0, 0)", }
                              )
        full_path = folder_path + folder_name + filename + ext
        fig.write_image(full_path, scale=scale)

    elif backend == "plotly3d":
        if bg_transparency:
            fig.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)",
                               "paper_bgcolor": "rgba(0, 0, 0, 0)", },
                              )
            fig.update_scenes(xaxis_showbackground=False,
                              yaxis_showbackground=False,
                              zaxis_showbackground=False)
        for camera in camera_angles:
            if camera == "top" or camera == "floorplan":
                camera_dict = dict(eye=dict(x=0., y=0., z=2.5),
                                   up=dict(x=0, y=1, z=0),
                                   center=dict(x=0, y=0, z=0),
                                   projection_type=projection)
            elif camera == "lateral" or camera == "side" or camera == "section":
                camera_dict = dict(eye=dict(x=2.5, y=0., z=0.0),
                                   up=dict(x=0, y=0, z=1),
                                   center=dict(x=0, y=0, z=0),
                                   projection_type=projection)
            elif camera == "front":
                camera_dict = dict(eye=dict(x=0., y=2.5, z=0.),
                                   up=dict(x=0, y=1, z=1),
                                   center=dict(x=0, y=0, z=0),
                                   projection_type=projection)
            elif camera == "rear" or camera == "back":
                camera_dict = dict(eye=dict(x=0., y=-2.5, z=0.),
                                   up=dict(x=0, y=1, z=1),
                                   center=dict(x=0, y=0, z=0),
                                   projection_type=projection)
            elif camera == "diagonal_front":
                camera_dict = dict(eye=dict(x=1.50, y=1.50, z=1.50),
                                   up=dict(x=0, y=0, z=1),
                                   center=dict(x=0, y=0, z=0),
                                   projection_type=projection)
            elif camera == "diagonal_rear":
                camera_dict = dict(eye=dict(x=-1.50, y=-1.50, z=1.50),
                                   up=dict(x=0, y=0, z=1),
                                   center=dict(x=0, y=0, z=0),
                                   projection_type=projection)
            else:
                camera_dict = None
            fig.update_layout(scene_camera=camera_dict, margin=dict(l=0, r=0, b=0, t=0))

            full_path = folder_path + folder_name + filename + f"_{camera}_{projection}" + ext
            fig.write_image(full_path, scale=scale)
            print("Figure saved at " + full_path)

    elif backend == "matplotlib":
        if bg_transparency:
            fig.patch.set_alpha(0.0)
            if ax:
                for axes in ax:
                    axes.patch.set_alpha(0.0)
        full_path = folder_path + folder_name + filename + ext
        fig.savefig(full_path, dpi=dpi, transparent=bg_transparency, bbox_inches=())
        print("Figure saved at " + full_path)


def plotly_fig_buttons(fig, y=1.08, x=0.80, x_diff=0.30):
    """
    Adds 2 buttons to change perspective and camera position in a Plotly figure.
    'y', 'x' and 'x_diff' may vary depending on the figure size. Currently configured for a 800x800 figure size.

    Parameters
    ----------
    fig : class
        Plotly figure object.
    y : float, optional
        Vertical coordinate of the first button in the Y axis of the figure.
    x : float, optional
        Horizontal coordinate of the first button in the X axis of the figure.
    x_diff : float, optional
        Horizontal spacing between the 2 buttons in the X axis of the figure.
    """
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=list([
                    dict(
                        args=[{"scene.camera.projection.type": "orthographic"}],
                        label="Orthographic",
                        method="relayout",
                    ),
                    dict(
                        args=[{"scene.camera.projection.type": "perspective"}],
                        label="Perspective",
                        method="relayout",
                    ),
                ]),
                type="dropdown",
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=x,
                xanchor="left",
                y=y,
                yanchor="top",
            ),

            dict(
                buttons=list([
                    dict(
                        args=[{"scene.camera.eye": {"x": 1.5, "y": 1.5, "z": 1.5},
                               "scene.camera.up": {"x": 0, "y": 0, "z": 1},
                               "scene.camera.center": {"x": 0, "y": 0, "z": 0}}],
                        label="Diagonal Front",
                        method="relayout",
                    ),
                    dict(
                        args=[{"scene.camera.eye": {"x": -1.5, "y": -1.5, "z": 1.5},
                               "scene.camera.up": {"x": 0, "y": 0, "z": 1},
                               "scene.camera.center": {"x": 0, "y": 0, "z": 0}}],
                        label="Diagonal Rear",
                        method="relayout",
                    ),
                    dict(
                        args=[{"scene.camera.eye": {"x": 0, "y": 0, "z": 2.5},
                               "scene.camera.up": {"x": 0, "y": 1, "z": 0},
                               "scene.camera.center": {"x": 0, "y": 0, "z": 0}}],
                        label="Top",
                        method="relayout",
                    ),
                    dict(
                        args=[{"scene.camera.eye": {"x": 2.5, "y": 0, "z": 0},
                               "scene.camera.up": {"x": 0, "y": 0, "z": 1},
                               "scene.camera.center": {"x": 0, "y": 0, "z": 0}}],
                        label="Lateral Right",
                        method="relayout",
                    ),
                    dict(
                        args=[{"scene.camera.eye": {"x": -2.5, "y": 0, "z": 0},
                               "scene.camera.up": {"x": 0, "y": 0, "z": 1},
                               "scene.camera.center": {"x": 0, "y": 0, "z": 0}}],
                        label="Lateral Left",
                        method="relayout",
                    ),
                    dict(
                        args=[{"scene.camera.eye": {"x": 0, "y": 2.5, "z": 0},
                               "scene.camera.up": {"x": 0, "y": 1, "z": 1},
                               "scene.camera.center": {"x": 0, "y": 0, "z": 0}}],
                        label="Front",
                        method="relayout",
                    ),
                    dict(
                        args=[{"scene.camera.eye": {"x": 0, "y": -2.5, "z": 0},
                               "scene.camera.up": {"x": 0, "y": 1, "z": 1},
                               "scene.camera.center": {"x": 0, "y": 0, "z": 0}}],
                        label="Rear",
                        method="relayout",
                    ),
                ]),
                type="dropdown",
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=x + x_diff,
                xanchor="left",
                y=y,
                yanchor="top",
            ),
        ]
    )


def hide_plotly_info(fig):
    """
    Hides title, axis and legends of the figure.

    Parameters
    ----------
    fig : class
        Plotly figure object.
    """
    fig.update_layout(title="",
                      showlegend=False,
                      scene={'xaxis_title': "",
                             'yaxis_title': "",
                             'zaxis_title': "",
                             'xaxis': dict(showticklabels=False,
                                           showgrid=False,
                                           showline=False,
                                           zeroline=False),
                             'yaxis': dict(showticklabels=False,
                                           showgrid=False,
                                           showline=False,
                                           zeroline=False),
                             'zaxis': dict(showticklabels=False,
                                           showgrid=False,
                                           showline=False,
                                           zeroline=False)}
                      )


def update_plotly_layout(fig, projection, default_camera, fig_title, fig_size, xlim, ylim, zlim, axis_titles,
                         aspect_mode):
    """
    Updates the layout of a Plotly figure.

    Parameters
    ----------
    fig : class
        Plotly figure object.
    projection : str
        Projection type used in 3D Plotly figures - 'orthographic' or 'perspective'.
    default_camera : str
        Default camera position that will be used at image display.
    fig_title : str
        Figure title.
    fig_size : tuple or list
        Width and height of the figure.
    xlim : tuple or list
        Minimum and maximum limits of the X axis.
    ylim : tuple or list
        Minimum and maximum limits of the Y axis.
    zlim : tuple or list
        Minimum and maximum limits of the Z axis.
    axis_titles : list
        List containing the string value for the names of the X, Y and Z axis.
    aspect_mode : str
        Aspect mode of the figure - 'auto', 'cube', 'data' or 'manual'.
    """
    if default_camera == "diagonal_front":
        camera_dict = dict(eye=dict(x=1.50, y=1.50, z=1.50),
                           up=dict(x=0, y=0, z=1),
                           center=dict(x=0, y=0, z=0),
                           projection_type=projection)
    elif default_camera == "top":
        camera_dict = dict(eye=dict(x=0., y=0., z=2.5),
                           up=dict(x=0, y=1, z=0),
                           center=dict(x=0, y=0, z=0),
                           projection_type=projection)
    else:
        camera_dict = None
    fig.update_layout(
        title=fig_title,
        width=fig_size[0], height=fig_size[1],
        scene=dict(
            xaxis=dict(range=[xlim[0], xlim[1]]),
            yaxis=dict(range=[ylim[0], ylim[1]]),
            zaxis=dict(range=[zlim[0], zlim[1]]),
            xaxis_title=axis_titles[0],
            yaxis_title=axis_titles[1],
            zaxis_title=axis_titles[2],
            aspectmode=aspect_mode,
        ),
        scene_camera=camera_dict,
    )


def points3d(single_points=None, single_labels=None, single_colors=None, single_opacity=0.7, single_symbol="diamond",
             group_points=None, group_labels=None, group_colors=None, group_opacity=0.7, group_symbol="circle",
             xlim=None, ylim=None, zlim=None, axis_titles=None, aspect_mode="auto", hide_info=False,
             fig_size=(800, 800), projection="orthographic", save_fig=False, fig_title="", fig=None,
             filename="points3d", folder_path=None, folder_name=None, fig_buttons=True, show_fig=True,
             update_layout=True, default_camera="diagonal_front", **kwargs
             ):
    """
    Display a 3D set of points in a 3D plot using Plotly.

    Parameters
    ----------
    single_points : array, optional
        (N, 3) array of points to be displayed with individual labels and colors.
    single_labels : list, optional
        List of strings containing a label for each value in 'singlePoints'.
    single_colors : list, optional
        List of strings containing the color of each value in 'singlePoints'.
    single_opacity : float, optional
        Transparency value for values passed in 'singlePoints'.
    single_symbol : str, optional
        Symbol used to display the values in 'singlePoints'.
    group_points : array, optional
        (N, 3) array of points to be displayed with a shared label and color.
    group_labels : str, optional
        Label for the values in 'group_points'.
    group_colors : str, optional
        Color for the values in 'group_points'.
    group_opacity : float, optional
        Transparency value for values passed in 'group_points'.
    group_symbol : str, optional
        Symbol used to display the values in 'group_points'.
    xlim : tuple or list, optional
        Minimum and maximum limits of the X axis.
    ylim : tuple or list, optional
        Minimum and maximum limits of the Y axis.
    zlim : tuple or list, optional
        Minimum and maximum limits of the Z axis.
    axis_titles : list, optional
        List containing the string value for the names of the X, Y and Z axis.
    aspect_mode : str, optional
        Aspect mode of the figure - 'auto', 'cube', 'data' or 'manual'.
    hide_info : bool, optional
        Option to hide title, axis and legends of the figure.
    fig_size : tuple or list, optional
        Width and height of the figure.
    projection : str, optional
        Projection type used in 3D Plotly figures - 'orthographic' or 'perspective'.
    save_fig : bool, optional
        Option to save a static image.
    filename : str
        String that will be used as the partial file name.
    folder_path : str, optional
        String containing the destination folder in which the image will be saved. If not defined the current active
        folder will be used.
    folder_name : str, optional
        String containing the name of the new folder that will be created in the destination folder in which the images
        will be saved. If not defined "backend" will be used as the new folder name.
    fig : class or None, optional
        Plotly figure object that the points will be added to. If 'None' a new Plotly figure object will be created.
    fig_buttons : bool, optional
        Option to add buttons to change perspective and camera angles.
    show_fig : bool, optional
        Option to display or not the figure once the plot is finished.
    fig_title : str, optional
        Figure title.
    update_layout : bool, optional
        Option to update the figure layout or not.
    default_camera : str, optional
        Default camera position that will be used at image display.
    kwargs : keyword arguments, optional
        See rasta._plot.save_plotly_fig.

    Returns
    -------
    Plotly figure object.
    """
    if axis_titles is None:
        axis_titles = ["X", "Y", "Z"]
    if zlim is None:
        zlim = [None, None]
    if ylim is None:
        ylim = [None, None]
    if xlim is None:
        xlim = [None, None]
    if fig is None:
        fig = go.Figure()

    if update_layout:
        update_plotly_layout(fig, projection, default_camera, fig_title, fig_size, xlim, ylim, zlim, axis_titles,
                             aspect_mode)

    if single_points is not None:
        if single_labels is None:
            single_labels = [f"Point {i}" for i in range(len(single_points))]
        if single_colors is None:
            single_colors = [None for _ in range(len(single_points))]
        for i in range(len(single_points)):
            fig.add_trace(go.Scatter3d(
                x=[single_points[i, 0]],
                y=[single_points[i, 1]],
                z=[single_points[i, 2]],
                mode="markers",
                marker=dict(
                    symbol=single_symbol,
                    color=single_colors[i],
                ),
                name=single_labels[i],
                opacity=single_opacity,
            ))

    if group_points is not None:
        if group_labels is None:
            group_labels = "Group Points"
        fig.add_trace(go.Scatter3d(
            x=group_points[:, 0].tolist(),
            y=group_points[:, 1].tolist(),
            z=group_points[:, 2].tolist(),
            mode="markers",
            marker=dict(
                symbol=group_symbol,
                color=group_colors,
            ),
            name=group_labels,
            opacity=group_opacity,
        ))

    if hide_info:
        hide_plotly_info(fig)

    if save_fig:
        save_plotly_fig(fig, "plotly3d", filename, folder_path=folder_path, folder_name=folder_name,
                        projection=projection)

    if show_fig:
        if fig_buttons and not hide_info:
            plotly_fig_buttons(fig, **kwargs)
        fig.show()

    return fig


def geometry3d(point_data, line_data, mesh_data, surface_labels, surface_colors,
               line_color="rgba(0, 0, 0, 0.05)", line_width=4,
               show_mesh=True, show_lines=True, show_elements=True,
               xlim=None, ylim=None, zlim=None, axis_titles=None,
               aspect_mode="auto", hide_info=False, fig_size=(800, 800), projection="orthographic",
               save_fig=False, filename="volume3d", folder_path=None, folder_name=None,
               fig=None, fig_buttons=True, show_fig=True, fig_title="",
               update_layout=True, default_camera="diagonal_front",
               axis=None, measurement=None, **kwargs
               ):
    """
    Display a 3D geometry in a 3D plot using Plotly.

    Parameters
    ----------
    point_data : dict
        Dictionary containing points index and coordinates.
    line_data : dict
        Dictionary containing lines connectivity data.
    mesh_data : dict
        Dictionary containing mesh elements, vertices and domain indices.
    surface_labels : dict
        Dictionary containing the surface label of each surface.
    surface_colors : dict
        Dictionary containing the RGB color of each surface.
    line_color : str, optional
        String of RGBA values for lines.
    line_width : int, optional
        Line width value.
    show_mesh : bool, optional
        Option to display wire mesh.
    show_lines : bool, optional
        Option to display surface lines.
    show_elements : bool, optional
        Option to display mesh elements.
    xlim : tuple or list, optional
        Minimum and maximum limits of the X axis.
    ylim : tuple or list, optional
        Minimum and maximum limits of the Y axis.
    zlim : tuple or list, optional
        Minimum and maximum limits of the Z axis.
    axis_titles : list, optional
        List containing the string value for the names of the X, Y and Z axis.
    aspect_mode : str, optional
        Aspect mode of the figure - 'auto', 'cube', 'data' or 'manual'.
    hide_info : bool, optional
        Option to hide title, axis and legends of the figure.
    fig_size : tuple or list, optional
        Width and height of the figure.
    projection : str, optional
        Projection type used in 3D Plotly figures - 'orthographic' or 'perspective'.
    save_fig : bool, optional
        Option to save a static image.
    filename : str
        String that will be used as the partial file name.
    folder_path : str, optional
        String containing the destination folder in which the image will be saved. If not defined the current active
        folder will be used.
    folder_name : str, optional
        String containing the name of the new folder that will be created in the destination folder in which the images
        will be saved. If not defined "backend" will be used as the new folder name.
    fig : class or None, optional
        Plotly figure object that the points will be added to. If 'None' a new Plotly figure object will be created.
    fig_buttons : bool, optional
        Option to add buttons to change perspective and camera angles.
    show_fig : bool, optional
        Option to display or not the figure once the plot is finished.
    fig_title : str, optional
        Figure title.
    update_layout : bool, optional
        Option to update the figure layout or not.
    default_camera : str, optional
        Default camera position that will be used at image display.

    kwargs : keyword arguments, optional
        See rasta._plot.save_plotly_fig.

    Returns
    -------
    Plotly figure object.
    """

    if axis_titles is None:
        axis_titles = ["X", "Y", "Z"]
    if zlim is None:
        zlim = [None, None]
    if ylim is None:
        ylim = [None, None]
    if xlim is None:
        xlim = [None, None]
    if fig is None:
        fig = go.Figure()

    if update_layout:
        update_plotly_layout(fig, projection, default_camera, fig_title, fig_size, xlim, ylim, zlim, axis_titles,
                             aspect_mode)

    displayed_names = [fig.data[i].name for i in range(len(fig.data))]

    if mesh_data is not None:
        # Unpacking data
        vertices = mesh_data["vertices"]
        elements = mesh_data["elements"]
        domain_indices = mesh_data["domain_indices"]

        # Adding mesh grid
        if show_mesh:
            trisurf = ff.create_trisurf(x=vertices[0, :],
                                        y=vertices[1, :],
                                        z=vertices[2, :],
                                        simplices=elements.T,
                                        color_func=elements.shape[1] * ["rgba(0, 0, 0, 0.05)"],
                                        edges_color="rgba(0, 0, 0, 0.05)",
                                        showbackground=False,
                                        )
            trisurf.data[1]["name"] = "Mesh"
            trisurf.data[1]["showlegend"] = True if "Mesh" not in displayed_names else False
            fig.add_trace(trisurf.data[1])  # Mesh lines
            displayed_names = [fig.data[i].name for i in range(len(fig.data))]

        # Adding mesh elements
        if show_elements:
            for di in set(domain_indices):
                indx = np.argwhere(domain_indices == di)
                con = elements.T[indx, :][:, 0, :]
                con = con.T
                fig.add_trace(go.Mesh3d(x=vertices[0, :], y=vertices[1, :], z=vertices[2, :],
                                        i=con[0, :], j=con[1, :], k=con[2, :],
                                        showlegend=True if surface_labels[di - 1] not in displayed_names else False,
                                        legendgroup=surface_labels[di - 1],
                                        visible=True,
                                        hovertext=f"PG: {di:0.0f}",
                                        color=surface_colors[di - 1] if surface_colors is not None else None,
                                        name=surface_labels[di - 1],
                                        opacity=0.3 if surface_colors is None else None
                                        ))
                displayed_names = [fig.data[i].name for i in range(len(fig.data))]

    # Adding lines
    if show_lines:
        for key, value in line_data.items():
            fig.add_trace(go.Scatter3d(
                x=[point_data[value[0]][0], point_data[value[1]][0]],
                y=[point_data[value[0]][1], point_data[value[1]][1]],
                z=[point_data[value[0]][2], point_data[value[1]][2]],
                mode="lines",
                legendgroup="lines",
                name="Lines",
                showlegend=True if "Lines" not in displayed_names else False,
                marker=dict(size=2),
                line=dict(width=line_width, color=line_color),
            ))
            displayed_names = [fig.data[i].name for i in range(len(fig.data))]

    if axis is not False:
        max_x = np.max(np.asarray(list(point_data.values()))[:, 0]) + 0.25
        max_y = np.max(np.asarray(list(point_data.values()))[:, 1]) + 0.25
        max_z = np.max(np.asarray(list(point_data.values()))[:, 2]) + 0.25
        axis = {"origin": (0, 0, 0), "size": (max_x, max_y, max_z), "shift": 0.25} if axis is None else axis
        fig = axis3d(fig, **axis)

    if measurement:
        max_y = np.max(np.asarray(list(point_data.values()))[:, 1])
        measurement = {"origin": (0, 0, 0), "axis": "y", "size": max_y} \
            if measurement is True else measurement
        fig = measurement3d(fig, **measurement)

    if hide_info:
        hide_plotly_info(fig)

    if save_fig:
        save_plotly_fig(fig, "plotly3d", filename, folder_path=folder_path, folder_name=folder_name,
                        projection=projection, **kwargs)

    if show_fig:
        if fig_buttons and not hide_info:
            plotly_fig_buttons(fig)
        fig.show()

    return fig


def axis3d(fig, origin=(0, 0, 0), size=(1, 1, 1), shift=0.25):
    """
    Add 3D reference axis to Plotly figure.

    Parameters
    ----------
    fig : class or None, optional
        Plotly figure object that the points will be added to. If 'None' a new Plotly figure object will be created.
    origin : tuple, optional
        X, Y and Z origin values.
    size : tuple, optional
        Tuple of floats containing the X, Y and Z arrow size in meters.
    shift : float, optional
        Size shift in meters for label placement.

    Returns
    -------
    Plotly figure object.
    """
    if fig is None:
        fig = go.Figure()

    x = [origin[0] + size[0], origin[0], origin[0]]
    y = [origin[1], origin[1] + size[1], origin[1]]
    z = [origin[2], origin[2], origin[2] + size[2]]
    norm_x = [1, 0, 0]
    norm_y = [0, 1, 0]
    norm_z = [0, 0, 1]
    colors = ["rgba(150, 0, 0, 0.3)",
              "rgba(0, 150, 0, 0.3)",
              "rgba(0, 0, 150, 0.3)"]
    text = ["<b>X</b>", "<b>Y</b>", "<b>Z</b>"]
    for i in range(3):
        fig.add_trace(go.Scatter3d(x=[origin[0], x[i]],
                                   y=[origin[1], y[i]],
                                   z=[origin[2], z[i]],
                                   mode="lines",
                                   name="Axes",
                                   legendgroup="axes",
                                   showlegend=False,
                                   hovertext=text[i],
                                   marker=dict(size=2),
                                   line=dict(width=4, color=colors[i]),
                                   ))
        fig.add_trace(go.Cone(x=[x[i]],
                              y=[y[i]],
                              z=[z[i]],
                              u=[norm_x[i]],
                              v=[norm_y[i]],
                              w=[norm_z[i]],
                              name=text[i],
                              legendgroup="axes",
                              showlegend=False,
                              sizeref=0.2,
                              showscale=False,
                              colorscale=[[0, colors[i]], [1, colors[i]]],
                              ))
    fig["layout"]["scene"]["annotations"] += tuple([dict(x=x[0] + shift,
                                                         y=y[0],
                                                         z=z[0],
                                                         text=text[0],
                                                         font=dict(color=colors[0]),
                                                         showarrow=False),
                                                    dict(x=x[1],
                                                         y=y[1] + shift,
                                                         z=z[1],
                                                         text=text[1],
                                                         font=dict(color=colors[1]),
                                                         showarrow=False),
                                                    dict(x=x[2],
                                                         y=y[2],
                                                         z=z[2] + shift,
                                                         text=text[2],
                                                         font=dict(color=colors[2]),
                                                         showarrow=False)])
    return fig


def measurement3d(fig, origin=(0, 0, 0), axis="y", size=1, shift=0.25, fontsize=16, anchors=None):
    """
    Add 3D reference measurement to Plotly figure.

    Parameters
    ----------
    fig : class or None, optional
        Plotly figure object that the points will be added to. If 'None' a new Plotly figure object will be created.
    origin : tuple, optional
        X, Y and Z origin values.
    axis : str, optional
        String representation of what axis te measurement is parallel to.
    size : float, optional
        Measurement size in meters.
    shift : float, optional
        Shift value of the line in relation to the measurement points.
    fontsize : int, optional
        Font size used to display the measurement value.
    anchors : list, optional
        List of strings containing the X and Y anchors of the annotation.

    Returns
    -------
    Plotly figure object.
    """
    if fig is None:
        fig = go.Figure()

    origin = list(origin)
    origin_shift = origin.copy()

    if axis == "x":
        origin_shift[1] += shift
        end = [origin[0] + size, origin[1], origin[2]]
        end_shift = [origin[0] + size, origin_shift[1], origin_shift[2]]
        norm = [1, 0, 0]
        anchors = ["center", "top"] if anchors is None else anchors
    elif axis == "y":
        origin_shift[2] += shift
        end = [origin[0], origin[1] + size, origin[2]]
        end_shift = [origin_shift[0], origin[1] + size, origin_shift[2]]
        norm = [0, 1, 0]
        anchors = ["center", "bottom"] if anchors is None else anchors

    elif axis == "z":
        origin_shift[1] += shift
        end = [origin[0], origin[1], origin[2] + size]
        end_shift = [origin_shift[0], origin_shift[1], origin[2] + size]
        norm = [0, 0, 1]
        anchors = ["right", "middle"] if anchors is None else anchors

    color = "rgba(0, 0, 0, 1)"
    text = f"<b>{size:0.2f} [m]</b>"

    # Measurement line
    fig.add_trace(go.Scatter3d(x=[origin_shift[0], end_shift[0]],
                               y=[origin_shift[1], end_shift[1]],
                               z=[origin_shift[2], end_shift[2]],
                               mode="lines",
                               name="Measurement",
                               legendgroup="measurement",
                               showlegend=False,
                               hovertext=text,
                               marker=dict(size=2),
                               line=dict(width=4, color=color),
                               ))
    # Front shift line
    fig.add_trace(go.Scatter3d(x=[origin[0], origin_shift[0]],
                               y=[origin[1], origin_shift[1]],
                               z=[origin[2], origin_shift[2]],
                               mode="lines",
                               name="Measurement",
                               legendgroup="measurement",
                               showlegend=False,
                               hovertext=text,
                               marker=dict(size=2),
                               line=dict(width=4, color=color),
                               ))
    # Rear shift line
    fig.add_trace(go.Scatter3d(x=[origin[0] + end[0], end_shift[0]],
                               y=[origin[1] + end[1], end_shift[1]],
                               z=[origin[2] + end[2], end_shift[2]],
                               mode="lines",
                               name="Measurement",
                               legendgroup="measurement",
                               showlegend=False,
                               hovertext=text,
                               marker=dict(size=2),
                               line=dict(width=4, color=color),
                               ))
    # Front arrow
    fig.add_trace(go.Cone(x=[end_shift[0]],
                          y=[end_shift[1]],
                          z=[end_shift[2]],
                          u=[norm[0]],
                          v=[norm[1]],
                          w=[norm[2]],
                          sizeref=0.2,
                          name="Measurement",
                          legendgroup="measurement",
                          showlegend=False,
                          showscale=False,
                          colorscale=[[0, color], [1, color]],
                          anchor="tip",
                          ))
    # Rear arrow
    fig.add_trace(go.Cone(x=[origin_shift[0]],
                          y=[origin_shift[1]],
                          z=[origin_shift[2]],
                          u=[-norm[0]],
                          v=[-norm[1]],
                          w=[-norm[2]],
                          sizeref=0.2,
                          name="Measurement",
                          legendgroup="measurement",
                          showscale=False,
                          colorscale=[[0, color], [1, color]],
                          anchor="tip",
                          ))

    fig["layout"]["scene"]["annotations"] += tuple([dict(x=(origin_shift[0] + end_shift[0]) / 2,
                                                         y=(origin_shift[1] + end_shift[1]) / 2,
                                                         z=(origin_shift[2] + end_shift[2]) / 2,
                                                         text=text,
                                                         xanchor=anchors[0],
                                                         yanchor=anchors[1],
                                                         font=dict(color=color, size=fontsize),
                                                         showarrow=False)])
    return fig
