def load_pth(p, checkpoint):
    if p['debug'] == True:
        new_checkpoint = {}
        for key in checkpoint.keys():
            new_checkpoint[key.replace('module.', '')] = checkpoint[key]
        return new_checkpoint
    else:
        return checkpoint