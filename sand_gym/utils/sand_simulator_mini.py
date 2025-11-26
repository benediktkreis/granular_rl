import numpy as np

class SandSimulatorMini:

    def __init__(
            self,
            field_size=[32,32], # list [nrow, ncol] or int [nrow,nrow] (if nrow=ncol) of hfield elevation defined in sand_shaping_arena.xml
            grid_cell_size_in_m=0.01,
            sand_range_in_m=0.2,
            angle_of_repose=34,
            flow_rate=0.125,
            unstable_cells_threshold = 1e-3,
            verbose=0,
            ):

        self.angle_of_repose_deg = angle_of_repose
        self.angle_of_repose_rad = np.deg2rad(self.angle_of_repose_deg)
        self.flow_rate = flow_rate
        self.unstable_cells_threshold = unstable_cells_threshold

        if type(field_size) != list:
            self.field_size = np.array([field_size, field_size], dtype=np.int32)
        else:
            self.field_size = np.asarray(field_size, dtype=np.int32)

        self.sand_range_in_mm = sand_range_in_m*1000
        self.grid_cell_size_in_mm = grid_cell_size_in_m*1000
        self.array_grid_cell_size_in_mm = np.ones(self.field_size)*self.grid_cell_size_in_mm

        # Debug
        self.verbose = verbose

    def simulate_sand(self, heightfield_grid=None):
        # Assume the heightfield is unstable
        is_unstable = True

        iteration_idx = 0
        while is_unstable:
            occupancy_mask = np.full(self.field_size, False, dtype="bool")

            iteration_idx += 1
            delta_h = np.array(self.get_unstable_cells(heightfield_grid, occupancy_mask)).astype(float)

            # If there are no cells that need adjusting, the field is stable
            abs_delta_h_sum = np.sum(np.array(abs(delta_h)))
            is_unstable = abs_delta_h_sum >= self.unstable_cells_threshold

            # Update the heightfield with the calculated delta_h
            heightfield_grid = heightfield_grid + delta_h

        # Only forward the simulation result if all values are valid
        # Setting negative values would result in a change of sand volume,
        # that is why it is not feasible
        if np.all((heightfield_grid >= 0.0) & (heightfield_grid <= self.sand_range_in_mm)):
            return heightfield_grid
        else:
            print("Error: Invalid sand value!")
            raise ValueError

    def set_boundary_occupancy_mask(self, occupancy_mask):
        occupancy_mask_copy = np.copy(occupancy_mask)
        occupancy_mask_copy[0, :] = 1
        occupancy_mask_copy[:, 0] = 1
        occupancy_mask_copy[self.field_size[0] - 1, :] = 1
        occupancy_mask_copy[:, self.field_size[1] - 1] = 1
        return occupancy_mask_copy

    def np_divide(self, a, b):
        #a/b
        return np.divide(a, b, out=np.zeros_like(a), where=b!=0.0)

    def get_unstable_cells(self, heightfield_copy, occupancy_mask):
        # Returns the delta_h of each cell
        bounded_occupancy_mask = self.set_boundary_occupancy_mask(occupancy_mask)

        # Roll the array so each cell moves 1 step in the indicated direction. This is used to later calculated the#
        # differences between the cells in each direction
        north = np.roll(heightfield_copy, -1, axis=0)
        northeast = np.roll(north, 1, axis=1)
        northwest = np.roll(north, -1, axis=1)

        south = np.roll(heightfield_copy, 1, axis=0)
        southeast = np.roll(south, 1, axis=1)
        southwest = np.roll(south, -1, axis=1)

        west = np.roll(heightfield_copy, -1, axis=1)
        east = np.roll(heightfield_copy, 1, axis=1)

        # Calculate the occupancy for each direction, i.e. if a cell is occupied by the tool
        occ_north = np.invert(np.logical_or(np.roll(bounded_occupancy_mask, -1, axis=0), bounded_occupancy_mask))
        occ_north_east = np.invert(
            np.logical_or(np.roll(np.roll(bounded_occupancy_mask, -1, axis=0), 1, axis=1), bounded_occupancy_mask))
        occ_north_west = np.invert(
            np.logical_or(np.roll(np.roll(bounded_occupancy_mask, -1, axis=0), -1, axis=1), bounded_occupancy_mask))

        occ_south = np.invert(np.logical_or(np.roll(bounded_occupancy_mask, 1, axis=0), bounded_occupancy_mask))
        occ_south_east = np.invert(
            np.logical_or(np.roll(np.roll(bounded_occupancy_mask, 1, axis=0), 1, axis=1), bounded_occupancy_mask))
        occ_south_west = np.invert(
            np.logical_or(np.roll(np.roll(bounded_occupancy_mask, 1, axis=0), -1, axis=1), bounded_occupancy_mask))

        occ_west = np.invert(np.logical_or(np.roll(bounded_occupancy_mask, -1, axis=1), bounded_occupancy_mask))
        occ_east = np.invert(np.logical_or(np.roll(bounded_occupancy_mask, 1, axis=1), bounded_occupancy_mask))

        # Compute the differences between the cell and its neighbors and multiply with the occupancy masks so cell
        # differences to occupied cells are not taken into account
        diff_down = (heightfield_copy - north) * occ_north
        diff_down_right = (heightfield_copy - northwest) * occ_north_west
        diff_down_left = (heightfield_copy - northeast) * occ_north_east

        diff_up = (heightfield_copy - south) * occ_south
        diff_up_right = (heightfield_copy - southwest) * occ_south_west
        diff_up_left = (heightfield_copy - southeast) * occ_south_east

        diff_right = (heightfield_copy - west) * occ_west
        diff_left = (heightfield_copy - east) * occ_east

        # Calculate the slope from the difference between cells
        # !! This only works because the cells are assumed to be square !!
        # This comes out as radians
        down_value = self.np_divide(diff_down, self.array_grid_cell_size_in_mm)
        slope_down = -np.arctan(down_value)
        down_right_value = self.np_divide(diff_down_right, self.array_grid_cell_size_in_mm)
        slope_down_right = -np.arctan(down_right_value)
        down_left_value = self.np_divide(diff_down_left, self.array_grid_cell_size_in_mm)
        slope_down_left = -np.arctan(down_left_value)

        up_value = self.np_divide(diff_up, self.array_grid_cell_size_in_mm)
        slope_up = -np.arctan(up_value)
        up_right_value = self.np_divide(diff_up_right, self.array_grid_cell_size_in_mm)
        slope_up_right = -np.arctan(up_right_value)
        up_left_value = self.np_divide(diff_up_left, self.array_grid_cell_size_in_mm)
        slope_up_left = -np.arctan(up_left_value)

        right_value = self.np_divide(diff_right, self.array_grid_cell_size_in_mm)
        slope_right = -np.arctan(right_value)
        left_value = self.np_divide(diff_left, self.array_grid_cell_size_in_mm)
        slope_left = -np.arctan(left_value)

        # Calculate the q-value as indicated in the paper
        q_down = self.get_q(slope_down)
        q_down_right = self.get_q(slope_down_right)
        q_down_left = self.get_q(slope_down_left)

        q_up = self.get_q(slope_up)
        q_up_right = self.get_q(slope_up_right)
        q_up_left = self.get_q(slope_up_left)

        q_right = self.get_q(slope_right)
        q_left = self.get_q(slope_left)

        # Calculate as is done in the paper
        delta_h = q_down + q_down_right + q_down_left + q_up + q_up_right + q_up_left + q_right + q_left
        delta_h /= 8  # A in the paper

        # Convert to degrees
        delta_h = np.rad2deg(delta_h)
        return delta_h

    def get_q(self, input_array):
        q = np.where(input_array < -self.angle_of_repose_rad, (input_array + self.angle_of_repose_rad) * self.flow_rate, 0)
        q += np.where(input_array > self.angle_of_repose_rad, (input_array - self.angle_of_repose_rad) * self.flow_rate, 0)

        # Round this to the 15th decimal place, because otherwise it results in weird floating point errors.
        return np.round(q, 15)