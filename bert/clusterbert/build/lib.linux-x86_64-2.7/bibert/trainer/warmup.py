'''
Author: CAI Zhijie
Date: 2021-09-14 03:37:46
LastEditTime: 2022-01-18 09:21:35
LastEditors: CAI Zhijie
Description: In User Settings Edit
FilePath: /workspace/MG-DL/MG-tools/ClusterBERT/clusterbert/bibert/trainer/warmup.py
'''

from torch.optim.lr_scheduler import _LRScheduler
import warnings
import numpy as np


EPOCH_DEPRECATION_WARNING = (
    "The epoch parameter in `scheduler.step()` was not necessary and is being "
    "deprecated where possible. Please use `scheduler.step()` to step the "
    "scheduler. During the deprecation, if epoch is different from None, the "
    "closed form is used instead of the new chainable form, where available. "
    "Please open an issue if you are unable to replicate your use case: "
    "https://github.com/pytorch/pytorch/issues/new/choose."
)


class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, gamma, lr_adjust_gamma, verbose=True, patience=6):
        self.optimizer = optimizer

        self.warmup_epochs = warmup_epochs

        super(WarmUpLR, self).__init__(optimizer)


        self.gamma = gamma
        
        self.p = 0
        self.m = 9999

        self.patience = patience

        self.lr_adjust_gamma = lr_adjust_gamma
        self.optimizer = optimizer
        self.loss_buffer = 9

        self.verbose = verbose

    def get_metric_lr(self, metric,tolerate = 0.78):
        '''
        @msg:在 patience 个batch内，loss 不低于原有loss 的tolerate，则按lr_adjust_gamma指数衰减学习率
        @param:metric loss ,
        @return:
            根据loss变化的学习率
        '''
        # self.optimizer.param_groups[0]['lr'] *= self.gamma

        if metric is not None and metric < self.loss_buffer * tolerate:
            self.m = float(metric)
            self.p = 0
        else:
            self.p += 1
            if self.p > self.patience:
                self.p = 0
                if self.verbose == True:
                    print(
                        f'\nlearning rate adjusted. {metric: .7f} > {self.loss_buffer * tolerate: .7f}\n')
                self.optimizer.param_groups[0]['lr'] *= self.lr_adjust_gamma
        return [self.optimizer.param_groups[0]['lr']]

    def get_trap_lr(self,smallest_lr=1e-7,weight_decay = 0.005):
        '''
        @msg:
        @param:
            smallest_lr:动态学习率的最小lr
            weigth_decay:逐渐衰减的权重
        @return:
            按梯形变化的学习率
        '''
        a = (self.last_epoch + 1) % (self.warmup_epochs)
        b = self.warmup_epochs - a
        self.optimizer.param_groups[0]['lr'] =  2 * max(min(a, b) / (
            self.warmup_epochs) * self.base_lrs[0], smallest_lr) * np.exp(-weight_decay * self.last_epoch)
        return [self.optimizer.param_groups[0]['lr']]
    
    def get_trig_lr(self,extend_weight = 1,smallest_lr=1e-8,weight_decay = 0.006):
        '''
        @msg:
        @param:
            smallest_lr:动态学习率的最小lr
            weigth_decay:逐渐衰减的权重
        @return:
            按三角形变化的学习率
        '''
        self.optimizer.param_groups[0]['lr'] = max(
            (1 - np.cos(extend_weight * self.last_epoch / self.warmup_epochs * np.pi)) * self.base_lrs[0], smallest_lr) * np.exp(-weight_decay * self.last_epoch)
        return self.optimizer.param_groups[0]['lr']
    
    def get_exp_lr(self):
        '''
        @msg:
        @param: 
            gamma 是指数率
        @return:
            按指数变化的学习率
        '''
        if self.last_epoch > self.warmup_epochs:
            self.optimizer.param_groups[0]['lr'] *= self.gamma
        else:
            self.optimizer.param_groups[0]['lr'] = (self.last_epoch + 1) / (self.warmup_epochs) * self.base_lrs[0]
        return [self.optimizer.param_groups[0]['lr']]

    # if you are trying to use the below 2 lr scheduler, make it named 'get_lr()' and make sure that the scheduler.step() has no feed.

    def get_lr(self):
        return self.get_trap_lr()


    def step(self, metric=None, epoch=None):
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn("Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                              "initialization. Please, make sure to call `optimizer.step()` before "
                              "`lr_scheduler.step()`. See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)

            # Just check if there were two first lr_scheduler.step() calls before optimizer.step()
            elif self.optimizer._step_count < 1:
                warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                              "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                              "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
                              "will result in PyTorch skipping the first value of the learning rate schedule. "
                              "See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
        self._step_count += 1

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False

        with _enable_get_lr_call(self):
            if epoch is None:
                self.last_epoch += 1
                if metric is None:
                    if self.last_epoch < self.warmup_epochs:
                        values = self.get_lr()
                    else:
                        values = self.get_trap_lr()
                elif metric is not None:
                    if self.last_epoch < self.warmup_epochs:
                        values = self.get_lr()
                    else:
                        values = self.get_metric_lr(metric)

            else:
                warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
                self.last_epoch = epoch
                if hasattr(self, "_get_closed_form_lr"):
                    values = self._get_closed_form_lr()
                elif metric is not None:
                    if self.last_epoch < self.warmup_epochs:
                        values = self.get_lr()
                    else:
                        values = self.get_metric_lr(metric)
                else:
                    if self.last_epoch < self.warmup_epochs:
                        values = self.get_lr()
                    else:
                        values = self.get_trap_lr()

        for param_group, lr in zip(self.optimizer.param_groups, values):
            param_group['lr'] = lr

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
