"""Utils for training and evaluating RNN model"""

from typing import Optional, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader






def perform_train_epoch(model: nn.Module,
                        loss_fn,
                        optimizer: optim.Optimizer,
                        dataloader_train: DataLoader,
                        device: Optional[int] = None,
                        max_grad_norm: Optional[float] = None) \
        -> float:
    """Perform one training epoch"""
    # if device is None:
    #     device = DEVICE

    is_training = model.training
    model.train()

    epoch_losses = []
    for i, (X, Y) in enumerate(dataloader_train):
        # if X.device != device and Y.device != device:
        #     X, Y = X.to(device), Y.to(device)
        model.zero_grad(set_to_none=True)
        _, loss = forward_pass_through_seq(model, X, Y=Y, loss_fn=loss_fn, return_Y_pred=False)#, device=device)
        loss.backward()
        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        epoch_losses.append(loss.detach().item())

    train_loss = torch.tensor(epoch_losses).sum().item()

    model.training = is_training

    return train_loss


@torch.no_grad()
def perform_test_epoch(model: nn.Module,
                       loss_fn,
                       dataloader_test: DataLoader,
                       device: Optional[int] = None) \
        -> float:
    """Evaluate on a test dataset"""
    # if device is None:
    #     device = DEVICE

    is_training = model.training
    model.eval()

    epoch_losses = []
    for i, (X, Y) in enumerate(dataloader_test):
        # if X.device != device and Y.device != device:
        #     X, Y = X.to(device), Y.to(device)
        _, loss = forward_pass_through_seq(model, X, Y=Y, loss_fn=loss_fn, return_Y_pred=False)#, device=device)

        epoch_losses.append(loss.detach().item())

    test_loss = torch.tensor(epoch_losses).sum().item()

    model.training = is_training

    return test_loss


def forward_pass_through_seq(model: nn.Module,
                             X: torch.Tensor,
                             Y: Optional[torch.Tensor] = None,
                             loss_fn = None,
                             return_Y_pred: bool = True,
                             device: Optional[int] = None) \
        -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """Full forward pass through sequence."""
    # if device is None:
    #     device = DEVICE

    # assert model._type in MODEL_TYPES

    # batch_size, num_frames, _ = X.shape
    # if Y is not None:
    #     assert Y.shape == (batch_size, num_frames, model._output_size)

    # perform forward pass, save outputs and loss
    loss = torch.zeros(1, dtype=torch.float)#, device=device)
    model.reset_hidden()

    # process
    out: torch.Tensor = model(X)
    Y = model._preprocess(Y, mode='outputs') # scale the output stats the same way as the input
    if Y is not None and loss_fn is not None:
        loss = loss_fn(out, Y)
    Y_pred = model._unpreprocess(out).detach().clone() if return_Y_pred else None # undo the effect of preprocessing

    return Y_pred, loss
