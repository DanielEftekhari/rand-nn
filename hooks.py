class Hook():
    def __init__(self, track, flag_hook=False):
        self.track = track
        self.flag_hook = flag_hook
        self.handle, self.layers = {}, {}
    
    def init_hook(self, names, layers):
        if self.track:
            for name in names:
                self.handle[name] = layers[name].register_forward_hook(self._get_hook(name))
    
    def _get_hook(self, name):
        def hook(model, input, output):
            if self.flag_hook:
                self.layers[name] = output.detach()
        return hook
    
    def clear_hook(self):
        for name in self.handle:
            self.handle[name].remove()
        self.layers = {}
