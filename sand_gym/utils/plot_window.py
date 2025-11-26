from matplotlib import pyplot as plt
import numpy as np
from sand_gym.utils.plot import plot_geometry
from cycler import cycler

dpi = 300
class PlotGraphWindow:
    def __init__(
        self,
        name="Graph plot",
        plot_size=None,
        axis_labels=None,
        legend_labels=None,
        plot_grid=True,
        visualize=True,  
    ):
        # Setup attributes
        self.name=name
        self.plot_size = plot_size
        self.plot_grid=plot_grid
        self.axis_labels=axis_labels
        self.legend_labels = legend_labels
        self.visualize=visualize
        self.initialize_line_styles(len(legend_labels))
        self.initialize_window()

    def initialize_line_styles(self, N):
        # Generate colors from a colormap
        cmap = plt.cm.viridis
        colors = [cmap(i) for i in np.linspace(0, 1, N)]

        # Define a list of linestyles (cycle if needed)
        linestyles = ['-', '--', '-.', ':']
        linestyles_cycle = [linestyles[i % len(linestyles)] for i in range(N)]

        # Create a custom property cycle that pairs each color with a linestyle.
        # Here we use the addition operator (+) to combine two cyclers with equal lengths.
        self.custom_cycle = cycler(color=colors) + cycler(linestyle=linestyles_cycle)
    
    def initialize_window(self):
        # Enable interactive mode
        plt.ion()

        self.fig, self.ax = plt.subplots()
        self.set_plot_params()
        plt.show(block=False)
        self.fig.canvas.manager.set_window_title(self.name)

    def set_plot_params(self):
        if self.plot_size is not None:
            if len(self.plot_size) >= 1:
                self.ax.set_xlim(self.plot_size[0][0], self.plot_size[0][1])
            if len(self.plot_size) == 2:
                self.ax.set_ylim(self.plot_size[1][0], self.plot_size[1][1])
        if self.axis_labels is None:
            self.axis_labels=["$x$", "$y$"]
        self.ax.set_xlabel(self.axis_labels[0])
        self.ax.set_ylabel(self.axis_labels[1])

        self.ax.set_prop_cycle(self.custom_cycle)

        if self.legend_labels is not None:
            self.ax.legend(loc='upper left')

        if self.plot_grid:
            self.ax.grid(True)
        else:
            self.ax.grid(False)

        self.ax.set_title(self.name)

    def close_window(self):
        plt.ioff()
        plt.close(self.fig)

    def update(self, x_data, y_data, markers='x', clear_plot=True):
        if clear_plot:
            self.ax.clear()
        self.ax.plot(x_data, y_data, marker=markers, label=self.legend_labels)
        if clear_plot:
            self.set_plot_params()
        self.fig.canvas.flush_events()

    def save_plot(self, save_path):
        self.fig.savefig(save_path+".png", dpi=dpi,bbox_inches='tight')

    def wait(self, wait_seconds=0.05):
        plt.pause(wait_seconds)

class PlotSandWindow:

    def __init__(
        self,
        name="Debug plot",
        field_size_grid=32,
        field_size_real=0.32,
        grid_cell_size=1,
        reference="grid",
        axis_labels=None,
        plot_grid=True,
        visualize=True,  
    ):
        # Setup attributes
        self.name=name
        if type(field_size_grid) == list:
            self.field_size_grid= max(field_size_grid)
        elif type(field_size_grid) == np.ndarray:
            self.field_size_grid= np.max(field_size_grid)
        else:
            self.field_size_grid=field_size_grid
        if type(field_size_real) == list:
            self.field_size_real = max(field_size_real)
        elif type(field_size_grid) == np.ndarray:
            self.field_size_real = np.max(field_size_real)
        else:
            self.field_size_real = field_size_real
        self.grid_cell_size = grid_cell_size
        self.plot_grid=plot_grid
        self.axis_labels=axis_labels
        self.visualize=visualize
        self.reference = reference

        if self.reference == "grid":
            self.plot_boundary = self.field_size_grid/4
            self.plot_size=[-self.plot_boundary, self.field_size_grid+self.plot_boundary] # Assumed to be quadratic
        elif self.reference == "world":
            self.plot_boundary = self.field_size_real/4
            self.plot_size=[-(self.field_size_real/2+self.plot_boundary), (self.field_size_real/2+self.plot_boundary)] # Assumed to be quadratic

        self.mesh = None
        self.cbar = None

        self.initialize_window()

    def initialize_window(self):
        # Enable interactive mode
        plt.ion()

        self.fig, self.ax = plt.subplots()
        self.set_plot_params()
    
        plt.show(block=False)
        self.fig.canvas.manager.set_window_title(self.name)

    def set_plot_params(self):
        self.ax.set_aspect('equal', 'box')
        self.ax.set_xlim(self.plot_size[0], self.plot_size[1])
        self.ax.set_ylim(self.plot_size[0], self.plot_size[1])
        if self.axis_labels is None:
            self.axis_labels=["$x$", "$y$"]
        self.ax.set_xlabel(self.axis_labels[0])
        self.ax.set_ylabel(self.axis_labels[1])
        
        # Major ticks every 20, minor ticks every 5
        if self.reference == "grid":
            major_size = 4
            minor_size = 1        
        elif self.reference == "world":
            major_size = 0.04
            minor_size = 0.01

        major_ticks = np.arange(self.plot_size[0], self.plot_size[1], major_size)
        minor_ticks = np.arange(self.plot_size[0], self.plot_size[1], minor_size)

        self.ax.set_xticks(major_ticks)
        self.ax.set_xticks(minor_ticks, minor=True)
        self.ax.set_yticks(major_ticks)
        self.ax.set_yticks(minor_ticks, minor=True)

        if self.plot_grid:
            grid_line_width = 1.0
            grid_line_color="black"
            self.ax.grid(which='minor', color=grid_line_color, alpha=0.2, linewidth=grid_line_width, zorder=2)
            self.ax.grid(which='major', color=grid_line_color, alpha=0.5, linewidth=grid_line_width, zorder=3)

        if self.mesh is not None and self.cbar is None:
            self.cbar = self.fig.colorbar(self.mesh, ax=self.ax)
        self.ax.set_title(self.name)

    def close_window(self):
        plt.ioff()
        plt.close(self.fig)

    def update(self, geometry, clear_plot=True):
        if clear_plot:
            self.ax.clear()
        _, self.mesh = plot_geometry(geometry, self.ax, self.field_size_grid, self.grid_cell_size)
        if clear_plot:
            self.set_plot_params()
        self.fig.canvas.flush_events()

    def save_plot(self, save_path):
        self.fig.savefig(save_path+".png", dpi=dpi,bbox_inches='tight')
        
    def wait(self, wait_seconds=0.05):
        plt.pause(wait_seconds)

class PlotEvalSandWindow:
    def __init__(
        self,
        name="Debug plot",
        field_size_grid=32,
        field_size_real=0.32,
        grid_cell_size=1,
        reference="grid",
        label_axis=False,
        plot_grid=False,
        visualize=True,  
    ):
        # Setup attributes
        self.name=name
        if type(field_size_grid) == list:
            self.field_size_grid= max(field_size_grid)
        elif type(field_size_grid) == np.ndarray:
            self.field_size_grid= np.max(field_size_grid)
        else:
            self.field_size_grid=field_size_grid
        if type(field_size_real) == list:
            self.field_size_real = max(field_size_real)
        elif type(field_size_grid) == np.ndarray:
            self.field_size_real = np.max(field_size_real)
        else:
            self.field_size_real = field_size_real
        self.grid_cell_size = grid_cell_size
        self.plot_grid=plot_grid
        self.label_axis=label_axis
        self.visualize=visualize
        self.reference = reference

        if self.reference == "grid":
            self.plot_boundary = self.field_size_grid/8
            self.plot_size=[-self.plot_boundary, self.field_size_grid+self.plot_boundary] # Assumed to be quadratic
        elif self.reference == "world":
            self.plot_boundary = self.field_size_real/8
            self.plot_size=[-(self.field_size_real/2+self.plot_boundary), (self.field_size_real/2+self.plot_boundary)] # Assumed to be quadratic

        self.initialize_window()

    def initialize_window(self):
        # Enable interactive mode
        plt.ion()

        self.fig, self.ax = plt.subplots()
        self.set_plot_params()
    
        plt.show(block=False)

    def set_plot_params(self):
        self.ax.set_aspect('equal', 'box')
        self.ax.set_xlim(self.plot_size[0], self.plot_size[1])
        self.ax.set_ylim(self.plot_size[0], self.plot_size[1])
        
        if self.label_axis:
            self.axis_labels=["$x$", "$y$"]
            self.ax.set_xlabel(self.axis_labels[0])
            self.ax.set_ylabel(self.axis_labels[1])
            
            # Major ticks every 20, minor ticks every 5
            if self.reference == "grid":
                major_size = 4
                minor_size = 1        
            elif self.reference == "world":
                major_size = 0.04
                minor_size = 0.01

            major_ticks = np.arange(self.plot_size[0], self.plot_size[1], major_size)
            minor_ticks = np.arange(self.plot_size[0], self.plot_size[1], minor_size)

            self.ax.set_xticks(major_ticks)
            self.ax.set_xticks(minor_ticks, minor=True)
            self.ax.set_yticks(major_ticks)
            self.ax.set_yticks(minor_ticks, minor=True)
        else:
            self.ax.set_xticks([])
            self.ax.set_yticks([])

        if self.plot_grid:
            grid_line_width = 1.0
            grid_line_color="black"
            self.ax.grid(which='minor', color=grid_line_color, alpha=0.2, linewidth=grid_line_width, zorder=2)
            self.ax.grid(which='major', color=grid_line_color, alpha=0.5, linewidth=grid_line_width, zorder=3)
        else:
            self.ax.grid(False)

    def close_window(self):
        plt.ioff()
        plt.close(self.fig)

    def update(self, geometry, clear_plot=True):
        if clear_plot:
            self.ax.clear()
        plot_geometry(geometry, self.ax, self.field_size_grid, self.grid_cell_size)
        if clear_plot:
            self.set_plot_params()
        self.fig.canvas.flush_events()

    def save_plot(self, save_path):
        self.fig.savefig(save_path+".png", dpi=dpi,bbox_inches='tight')

    def wait(self, wait_seconds=0.05):
        plt.pause(wait_seconds)