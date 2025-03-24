from torch.utils.tensorboard import SummaryWriter

_writers = {}  # dictionary, not a single writer

def get_writer(log_dir):
    global _writers
    if log_dir not in _writers:
        _writers[log_dir] = SummaryWriter(log_dir=log_dir)
    return _writers[log_dir]