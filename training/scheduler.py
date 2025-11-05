import torch.optim as optim

class NoamLR(optim.lr_scheduler._LRScheduler):
    """
    实现调度器
    """
    def __init__(self, optimizer, d_model, warmup_steps, last_epoch=-1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.num_steps = 0
        super(NoamLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        self.num_steps += 1
        arg1 = self.num_steps ** -0.5
        arg2 = self.num_steps * (self.warmup_steps ** -1.5)
        lr = (self.d_model ** -0.5) * min(arg1, arg2)
        return [lr for _ in self.optimizer.param_groups]