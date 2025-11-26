class MujocoSimUtils():
    def __init__(self, env):
        self.env = env

    def add_visual_element(self, vis_elem):
        self.env.add_visual_elements_to_mujoco_passive_viewer(vis_elem)

    def update_mujoco_heightfield(self):
        if self.env.sand_simulator is not None:
            self.env.sim.model._model.hfield_data = self.env.sand_simulator.get_normed_heightfield(flat=True, correct_render=self.env.correct_render)
            self.env.sim.forward()
            self.env.sim.step()
            self.env.sync_mujoco_passive_viewer()