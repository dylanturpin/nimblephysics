import nimblephysics_libs._nimblephysics as nimble
import torch
from typing import Tuple, Callable, List
import numpy as np
import math
import pdb


class BackpropSnapshotPointer:
  def __init__(self):
    self.backprop_snapshot = None


class ContactsLayer(torch.autograd.Function):
  """
  This implements a single, differentiable timestep of DART as a PyTorch layer
  """

  @staticmethod
  def forward(ctx, world, state, action):
    """
    We can't put type annotations on this declaration, because the supertype
    doesn't have any type annotations and otherwise mypy will complain, so here
    are the types:

    world: nimble.simulation.World
    state: torch.Tensor
    action: torch.Tensor
    -> torch.Tensor
    """

    world.setState(state.detach().numpy())
    world.setAction(action.detach().numpy())
    backprop_snapshot: nimble.neural.BackpropSnapshot = nimble.neural.forwardPass(world)
    ctx.backprop_snapshot = backprop_snapshot
    ctx.world = world

    return torch.tensor(np.concatenate((backprop_snapshot.getContactPositions(), backprop_snapshot.getContactNormals()), axis=1).flatten())

  @staticmethod
  def backward(ctx, grad_state):
    """
    In the backward pass we receive a Tensor containing the gradient of the loss
    with respect to the output, and we need to compute the gradient of the loss
    with respect to the input.
    """
    backprop_snapshot: nimble.neural.BackpropSnapshot = ctx.backprop_snapshot
    world: nimble.simulation.World = ctx.world

    #print(grad_state) 
    #print(grad_state.shape)

    position_J = backprop_snapshot.getContactPositionJacobians(world) # N_c*3 x N_q
    normal_J = backprop_snapshot.getContactNormalJacobians(world)     # N_c*3 x N_q
    combined_J = np.concatenate((position_J, normal_J), axis=0)       # N_c*6 x N_q
    # grad_state is N_c*6
    grad_wrt_q = torch.matmul(grad_state, torch.tensor(combined_J))
    # append 0s for grad_wrt_v

    return (
        None,
        torch.cat((grad_wrt_q, torch.zeros_like(grad_wrt_q)), dim=0),
        None
    )

def contacts(world: nimble.simulation.World, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
  """
  This does a forward pass on the `world` that gets passed in, storing information needed
  in order to do a backwards pass.
  """
  return ContactsLayer.apply(world, state, action)  # type: ignore