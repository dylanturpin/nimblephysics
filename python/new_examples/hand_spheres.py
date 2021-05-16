
import nimblephysics as nimble
from nimblephysics.contacts import BackpropSnapshotPointer
import torch
import numpy as np
import pdb
import math
from pyquaternion import Quaternion
import transformations as trans
import spin

# Load the world
world: nimble.simulation.World = nimble.simulation.World()
world.setGravity([0, 0, 0.])

hand: nimble.dynamics.Skeleton = world.loadSkeleton("/workspaces/nimblephysics/scratch/GraspIt2URDF/urdf/Barrett.urdf")

sphereC = nimble.dynamics.Skeleton()
sphereC.setName('sphereC')
sphereCJoint, sphereCNode = sphereC.createWeldJointAndBodyNodePair()
sphereCShape = sphereCNode.createShapeNode(nimble.dynamics.SphereShape(0.8))
sphereCShape.createCollisionAspect()
sphereCVisual = sphereCShape.createVisualAspect()
sphereCVisual.setRGBA([0.,0.,1.,1.])
world.addSkeleton(sphereC)

initialState = np.zeros(world.getStateSize())
initialState = np.array([-6.15655259e-01, -6.15655291e-01,  1.48168677e+00,  5.65205104e-01,
       -7.62656156e-09, -5.65788127e-01,  1.04758439e+00,  9.29470046e-01,
        3.05423153e-01,  1.04758432e+00,  9.29470028e-01,  3.05423249e-01,
        6.49779693e-01,  3.77954798e-01,
        -1.32590563e-01, -2.41994962e-05,
        3.80826679e-05, -4.81696995e-05,  1.29930163e-01, -1.14384829e-01,
        4.32849548e-01,  3.68610074e-01, -4.48690629e-01,  4.32605647e-01,
        3.68631338e-01, -4.48714563e-01, -1.58233842e-01,  2.39882643e-02,])
initialState = np.array([-6.16106956e-01, -6.16106989e-01,  1.48155382e+00,  5.62272748e-01,
       -1.27471986e-08, -5.62094800e-01,  1.04749915e+00,  9.02816617e-01,
        3.20569350e-01,  1.04749909e+00,  9.02816742e-01,  3.20569185e-01,
        6.21738383e-01,  3.93429554e-01, -4.19993520e-03, -6.13846199e-08,
        1.30781705e-07, -3.43063887e-08, -4.04168060e-03,  3.44744373e-02,
       -6.44890520e-04, -1.92315233e-01,  1.09428848e-01, -6.44814247e-04,
       -1.92314201e-01,  1.09426972e-01, -2.02366290e-01,  1.11828985e-01])
# NO CONTACTS YET
initialState = np.array([-6.16812478e-01, -6.16812515e-01,  1.48134596e+00,  5.57418938e-01,
       -1.98676202e-08, -5.55994772e-01,  1.04734829e+00,  8.60886400e-01,
        3.44478295e-01,  1.04734825e+00,  8.60886750e-01,  3.44477723e-01,
        5.77606759e-01,  4.17870300e-01, -4.19799044e-03, -6.60195373e-08,
        1.23227776e-07, -2.88542445e-08, -4.31483982e-03,  3.68956802e-02,
       -7.47764795e-04, -1.94152672e-01,  1.10932534e-01, -7.47685112e-04,
       -1.94151627e-01,  1.10930661e-01, -2.04384083e-01,  1.13429265e-01])
# THREE CONTACTS
initialState = np.array([-6.16623010e-01, -6.16623044e-01,  1.48140181e+00,  5.57446734e-01,
       -2.07894933e-08, -5.56531756e-01,  1.04752868e+00,  8.61474464e-01,
        3.43615261e-01,  1.04752864e+00,  8.61474808e-01,  3.43614698e-01,
        5.80344310e-01,  4.16456658e-01,  2.00302117e-02,  1.78075073e-07,
       -9.00253112e-08, -7.92749478e-08,  2.97382445e-02, -3.00746911e-02,
        1.49567372e-02,  3.28091284e-02, -6.26517558e-02,  1.49571744e-02,
        3.28087041e-02, -6.26511418e-02,  2.11034106e-01, -1.08317769e-01])
initialState = torch.tensor(initialState, requires_grad=True)

state = initialState
backprop_snapshot_holder = BackpropSnapshotPointer()
contacts = nimble.contacts(world, state, torch.zeros(world.getActionSize()), backprop_snapshot_holder)

states = [state]
results = []
results.append(world.getLastCollisionResult())
for i in range(1000):
  print(i)
  ## UPDATE BASED ON LOSS
  n_contacts = contacts.shape[0] // 6
  normals = contacts[:n_contacts*3].reshape(-1,3)
  print(normals)
  positions = contacts[n_contacts*3:].reshape(-1,3)
  target_normals = torch.zeros_like(normals)
  target_normals[:,1] = 1.
  l = (((normals-target_normals)**2).sum(axis=1)**(0.5)).sum()
  print(l)
  l.backward()
  print(world.getLastCollisionResult().getNumContacts())

  with torch.no_grad():
    learning_rate = 10000.0
    next_action = torch.zeros(world.getActionSize())
    grad_step = -state.grad[:14] * learning_rate
    next_action[:14] += grad_step
    #next_action[6:14] = torch.tensor([0., 1., 0.33333, 0., 1., 0.333333, 1., 0.333333])*1000
    #next_action[6:14] *= -1

    state = state.detach()
    state[-world.getActionSize():] = 0.
    state = nimble.timestep(world, state.detach(), next_action)
    state = nimble.timestep(world, state.detach(), torch.zeros(world.getActionSize()))

  ## UPDATE BASED ON NORMAL
  # walk through contacts, add normal loss for any that have children of the hand object
  bodyNodesA = backprop_snapshot_holder.backprop_snapshot.getBodyNodesA()
  bodyNodesB = backprop_snapshot_holder.backprop_snapshot.getBodyNodesB()

  ikMap = nimble.neural.IKMapping(world)
  normal_sign = torch.ones(normals.shape[0])
  for j in range(normals.shape[0]):
    bodyA = bodyNodesA[j] 
    bodyB = bodyNodesB[j] 
    if bodyA.getSkeleton() == hand:
      ikMap.addLinearBodyNode(bodyA)
      normal_sign[j] = -1
    elif bodyB.getSkeleton() == hand:
      ikMap.addLinearBodyNode(bodyB)
  state = torch.tensor(state, requires_grad=True)
  state.grad=None
  world_pos = nimble.map_to_pos(world, ikMap, state).reshape(-1,3)
  l = (((world_pos - (world_pos.detach() - normals[:,:].detach()))**2).sum(axis=1)**(0.5)).sum()
  l.backward()
  with torch.no_grad():
    next_action = torch.zeros(world.getActionSize())
    next_action[:14] += - state.grad[:14] / ((state.grad[:14]**2).sum()**0.5) * (grad_step**2).sum()**(0.5)
    state = state.detach()
    state[-world.getActionSize():] = 0.
    state = nimble.timestep(world, state.detach(), next_action)
    state = nimble.timestep(world, state.detach(), torch.zeros(world.getActionSize()))

  #next_action = torch.zeros(world.getActionSize()) 
  #normal_step = normals[0,:]*(grad_step**2).sum()**(0.5)
  #next_action[3:] += normal_step
  #state = state.detach()
  #next_action *= -learning_rate
  #state[-world.getActionSize():] = 0.
  #state = nimble.timestep(world, state.detach(), next_action)
  #state = nimble.timestep(world, state.detach(), torch.zeros(world.getActionSize()))

  state = torch.tensor(state, requires_grad=True)
  state.grad = None
  contacts = nimble.contacts(world, state, torch.zeros(world.getActionSize()), backprop_snapshot_holder)

  states.append(state.detach().clone())
  results.append(world.getLastCollisionResult())

gui = nimble.NimbleGUI(world, results[0])
gui.loopStates(states, results)
gui.serve(8080)
gui.blockWhileServing()