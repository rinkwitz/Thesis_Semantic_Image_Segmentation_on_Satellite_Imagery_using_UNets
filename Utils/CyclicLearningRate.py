def get_cyclic_lr(learning_rate_dict, num_epoch, num_batches, num_batch):
    assert learning_rate_dict['base_lr'] <= learning_rate_dict['max_lr']
    base_lr, max_lr = learning_rate_dict['base_lr'], learning_rate_dict['max_lr']
    num_current = (num_epoch % 4) * num_batches + num_batch
    stepsize = 2 * num_batches
    if num_current < stepsize:
        return base_lr + (max_lr - base_lr) * (num_current / stepsize)
    else:
        return max_lr - (max_lr - base_lr) * ((num_current - stepsize) / stepsize)
