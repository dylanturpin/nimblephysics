
import nimblephysics as nimble
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

sphereA = nimble.dynamics.Skeleton()
sphereA.setName('sphereA')
sphereAJoint, sphereANode = sphereA.createFreeJointAndBodyNodePair()
sphereAShape = sphereANode.createShapeNode(nimble.dynamics.SphereShape(0.1))
sphereAShape.createCollisionAspect()
sphereAVisual = sphereAShape.createVisualAspect()
sphereAVisual.setRGBA([1.,0.,0.,1.])
world.addSkeleton(sphereA)

sphereB = nimble.dynamics.Skeleton()
sphereB.setName('sphereB')
sphereBJoint, sphereBNode = sphereB.createWeldJointAndBodyNodePair()
sphereBShape = sphereBNode.createShapeNode(nimble.dynamics.SphereShape(0.1))
sphereBShape.createCollisionAspect()
sphereBVisual = sphereBShape.createVisualAspect()
sphereBVisual.setRGBA([1.,0.,0.,1.])
world.addSkeleton(sphereB)

initialState = np.zeros(world.getStateSize())
initialState[3:6] = [0.2, 0.0, 0.0]
initialState[9:12] = [0.0, 0.0, 0.0]
initialState = torch.tensor(initialState, requires_grad=True)

ikMap = nimble.neural.IKMapping(world)
ikMap.addLinearBodyNode(sphereA.getBodyNode(0))

state = initialState
contacts = nimble.contacts(world, state, torch.zeros(world.getActionSize()))

states = [state]
results = []
results.append(world.getLastCollisionResult())
for i in range(500):
  print(i)
  ## UPDATE BASED ON LOSS
  normals = contacts.reshape(-1,6)[:,3:]
  positions = contacts.reshape(-1,6)[:,:3]
  l = -(normals[:,1]-torch.ones_like(normals[:,1])**2).sum()
  l.backward()

  with torch.no_grad():
    learning_rate = 100.0
    next_action = torch.zeros(world.getActionSize())
    grad_step = -state.grad[:6] * learning_rate
    next_action[:6] += grad_step

    state = state.detach()
    state[-world.getActionSize():] = 0.
    state = nimble.timestep(world, state.detach(), next_action)
    state = nimble.timestep(world, state.detach(), torch.zeros(world.getActionSize()))

  ## UPDATE BASED ON NORMAL
  state = torch.tensor(state, requires_grad=True)
  state.grad=None
  world_pos = nimble.map_to_pos(world, ikMap, state)
  l = ((world_pos - (world_pos.detach() - normals[0,:].detach()))**2).sum()**(0.5)
  l.backward()
  with torch.no_grad():
    next_action = torch.zeros(world.getActionSize())
    next_action[:6] += - state.grad[:6] * (grad_step**2).sum()**(0.5)
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
  contacts = nimble.contacts(world, state, torch.zeros(world.getActionSize()))

  states.append(state.detach().clone())
  results.append(world.getLastCollisionResult())

gui = nimble.NimbleGUI(world, results[0])
gui.loopStates(states, results)
gui.serve(8080)
gui.blockWhileServing()