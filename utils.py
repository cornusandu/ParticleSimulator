__forces = []

def force(func):
    __forces.append(func)
    return func

def get_forces():
    return __forces.copy()
