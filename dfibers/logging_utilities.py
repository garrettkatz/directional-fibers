import os

def hardwrite(f,data):
    """
    Force file write to disk
    """
    if f.name == os.devnull: return
    f.write(data)
    f.flush()
    os.fsync(f)
