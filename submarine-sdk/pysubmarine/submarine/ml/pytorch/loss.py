from torch.nn.modules import loss


def get_loss_fn(key):
    def _condition_fn(x):
        k, v = x
        return isinstance(v, type) and issubclass(v, loss._Loss) and (k.lower() == key.lower())
    _, loss_fn = next(iter(filter(_condition_fn, vars(loss).items())))
    return loss_fn
