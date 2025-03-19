from torch.utils.tensorboard import SummaryWriter

_writer = None

def get_writer(log_dir=None):
    global _writer
    if _writer is None:
        assert log_dir is not None, "log_dir must be provided for the first writer initialization."
        _writer = SummaryWriter(log_dir)
    return _writer
