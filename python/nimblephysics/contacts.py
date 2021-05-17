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
  def forward(ctx, world, state, action, backprop_snapshot_holder):
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

    #dof_contact_types = backprop_snapshot.getDofContactTypes(world)
    #if np.any(dof_contact_types > 5) or np.any(dof_contact_types == 0):
      #pdb.set_trace()

    #positions = x.reshape(-1,6)[:,:3]
    #normals = x.reshape(-1,6)[:,3:]
    dof_types = backprop_snapshot.getDofContactTypes(world)
    body_names_A = [node.getName() for node in backprop_snapshot.getBodyNodesA()]
    body_names_B = [node.getName() for node in backprop_snapshot.getBodyNodesB()]
    print(body_names_A)
    print(body_names_B)
    results = world.getLastCollisionResult()

    indexes = []
    #filter_names = ['chain0link4', 'chain3link4', 'chain6link4']
    for i in range(len(body_names_A)):
      #if body_names_A[i] in filter_names or body_names_B[i] in filter_names:
      if True:
        indexes.append(i)
    ctx.indexes = np.array(indexes)
    x = torch.tensor(np.concatenate((backprop_snapshot.getContactNormals(), backprop_snapshot.getContactPositions()), axis=0).flatten())
    
    backprop_snapshot_holder.backprop_snapshot = backprop_snapshot
    #if results.getNumContacts() > 0:
      #pdb.set_trace()
    #result_positions = np.array([results.getContact(i).point for i in range(results.getNumContacts())])
    #if(x.shape[0]/6 >= 2):
      #pdb.set_trace()
      #print(result_positions)
    return x

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

    #indexes = np.concatenate((ctx.indexes*3, ctx.indexes*3+1, ctx.indexes*3+2))
    normal_J = backprop_snapshot.getContactNormalJacobians(world)#[indexes]     # N_c*3 x N_q
    position_J = backprop_snapshot.getContactPositionJacobians(world)#[indexes] # N_c*3 x N_q
    #force_J = backprop_snapshot.getContactForceJacobians(world)     # N_c*6 x N_q
    combined_J = np.concatenate((normal_J, position_J), axis=0)       # N_c* x N_q
    # grad_state is N_c*12
    grad_wrt_q = torch.matmul(grad_state, torch.tensor(combined_J))
    #pdb.set_trace()
    # append 0s for grad_wrt_v

    return (
        None,
        torch.cat((grad_wrt_q, torch.zeros_like(grad_wrt_q)), dim=0),
        None,
        None
    )

def contacts(world: nimble.simulation.World, state: torch.Tensor, action: torch.Tensor, backprop_snapshot_holder: BackpropSnapshotPointer) -> torch.Tensor:
  """
  This does a forward pass on the `world` that gets passed in, storing information needed
  in order to do a backwards pass.
  """
  return ContactsLayer.apply(world, state, action, backprop_snapshot_holder)  # type: ignore