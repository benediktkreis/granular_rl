from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import LinearSegmentedColormap
from shapely import plotting, Polygon, MultiPolygon
import numpy as np

heightmap_color_map="terrain"
minval = 0.5 #0.6
maxval = 1.0 #0.8
n = 256
h_cmap = plt.get_cmap(heightmap_color_map)
h_cmap_selection = LinearSegmentedColormap.from_list(
    f'trunc({h_cmap.name},{minval:.2f},{maxval:.2f})',
    h_cmap(np.linspace(minval, maxval, n))
)
goal_grid_size = np.array([26,26], dtype=np.int32)

def plot_geometry(geometries: list, ax, field_size, grid_cell_size, plot_centroids=False, colors = ["green", "black"]):
    if type(geometries) != list:
        geometries = [geometries]

    mask_index = 0
    mesh = None
    for geometry in geometries:
        if type(geometry) == Polygon or type(geometry) == MultiPolygon:
            plot_polygon(geometry, ax=ax, plot_centroids=plot_centroids)
        if type(geometry) == np.ndarray:
            if geometry.ndim == 1 and geometry.shape == (2,):
                plot_position(geometry, ax=ax)
            elif geometry.ndim == 2:
                if np.issubdtype(geometry.dtype, np.integer) and np.all(np.isin(geometry, [0, 1])):
                    plot_mask(geometry, ax=ax, color=colors[mask_index])
                    mask_index += 1
                else:
                    _, mesh = plot_heightmap(geometry, ax=ax, field_size=field_size, grid_cell_size=grid_cell_size, heightmap_boundary_color=colors[1])
            else:
                print("Numpy array dimension is not plotable.")
                print("geometry.ndim =", geometry.ndim)
                raise ValueError
    
    return ax, mesh

def plot_position(arr, ax, color="black"):
    x = arr[0]
    y = arr[1]
    ax.scatter(x, y, color=color, zorder=1)

    return ax

def plot_mask(arr, ax, color="gray"):
    ax.contour(arr, levels=[0.7], colors=[color], linewidths=2.0)
    return ax

def plot_heightmap(arr, ax, field_size, grid_cell_size, heightmap_boundary_color="red", heightmap_boundary_line_width=2.0):
    if (arr < 0).any():
        heightmap_min_value = -30.0
        heightmap_max_value = 30.0
    else:
        heightmap_min_value = 50.0
        heightmap_max_value = 70.0
           
    mesh = ax.pcolormesh(arr, cmap=h_cmap_selection, vmin=heightmap_min_value, vmax=heightmap_max_value ,zorder=0)

    (x1_s, y1_s) = [0,0]
    (x2_s, y2_s) = [arr.shape[1],arr.shape[0]]
    ax.plot([x1_s, x2_s, x2_s, x1_s, x1_s], [y1_s, y1_s, y2_s, y2_s, y1_s], color=heightmap_boundary_color, linewidth=heightmap_boundary_line_width, zorder=0)
    
    half_goal_grid_size = (goal_grid_size/2).astype(np.int32)
    half_hfield_grid_size = (np.array((arr.shape[1],arr.shape[0]), dtype=np.int32)/2).astype(np.int32)
    (x1_g, y1_g) = [half_hfield_grid_size[0]-half_goal_grid_size[0], half_hfield_grid_size[1]-half_goal_grid_size[1]]
    (x2_g, y2_g) = [half_hfield_grid_size[0]+half_goal_grid_size[0], half_hfield_grid_size[1]+half_goal_grid_size[1]]
    ax.plot([x1_g, x2_g, x2_g, x1_g, x1_g], [y1_g, y1_g, y2_g, y2_g, y1_g], color=heightmap_boundary_color, linewidth=heightmap_boundary_line_width, zorder=0)
    
    return ax, mesh

def plot_polygon(polygons, ax=None, polygon_color="C0", line_width=2.0, plot_centroids=False, centroids_color="black",):
    # Plot fresco
    plotting.plot_polygon(
        polygons,
        ax,
        alpha=0.5,
        add_points=False,
        color=polygon_color
    )
    # Plot centroids
    if plot_centroids:
        if type(polygons) == MultiPolygon:
            for polygon in polygons.geoms:
                plotting.plot_points(
                    polygon.centroid,
                    ax,
                    color=centroids_color
                )
        else:
            plotting.plot_points(
                polygons.centroid,
                ax,
                color=centroids_color
            )
    
    return ax