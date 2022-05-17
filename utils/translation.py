def str2bool(value):
    if value == "True":
        return True
    if value == "False":
        return False
    raise KeyError(value)
