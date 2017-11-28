class TraverserSettings(object):
    def __init__(self):
        pass

class Traverser(object):
    def __init__(self, f, Df, c, compute_step_size, terminate, settings):
        self.f = f
    def copy(self):
        pass
    def reset(self, x):
        pass
    def traverse(self, num_steps):
        pass
    def get_status(self):
        pass
    def get_fiber(self):
        pass
