def _iterate_val_loaders(val_loader):
    """Handle both dict of dataloaders or single dataloader cases."""
    if isinstance(val_loader, dict):
        iterators = {k: iter(v) for k, v in val_loader.items()}
        while True:
            batch_dict = {}
            exhausted = False
            for name, it in iterators.items():
                try:
                    batch_dict[name] = next(it)
                except StopIteration:
                    exhausted = True
                    break
            if exhausted:
                break
            yield batch_dict
    else:
        yield from val_loader
