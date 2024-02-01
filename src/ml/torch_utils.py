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
    """Evaluate trained RNNoise denoisers on a test dataset"""
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
    """Full forward pass through sequence, accumulating loss along the way."""
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
    out = model(X)
    if Y is not None and loss_fn is not None:
        loss = loss_fn(out, Y)
    Y_pred = out.detach().clone() if return_Y_pred else None

    return Y_pred, loss




# def load_model(model_id: str,
#                epoch: Optional[int] = None) \
#         -> [nn.Module, str, str, str]:
#     """Load neural network model from model store"""
#     model_subdir = os.path.join(MODELS_DIR, model_id)
#     model_fnames = [fname for fname in sorted(os.listdir(model_subdir)) if '.pickle' in fname and 'meta' not in fname]
#
#     if epoch is None: # last snapshot
#         model_fname = model_fnames[-1]
#     else: # get closest one
#         epoch_nums = np.array([int(fname[:3]) for fname in model_fnames])
#         idx_ = np.argmin(np.abs(epoch_nums - epoch))
#         model_fname = model_fnames[idx_]
#
#     model_fpath = os.path.join(model_subdir, model_fname)
#     model_obj = load_pickle(model_fpath)
#
#     model = model_obj['model']
#
#     return model, model_fname, model_fpath, model_subdir
