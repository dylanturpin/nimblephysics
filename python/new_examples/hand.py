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
#world.setGravity([0, 0, -9.81])
world.setGravity([0, 0, 0.])
#arm: nimble.dynamics.Skeleton = world.loadSkeleton("/workspaces/bhand_model/robots/bhand_model.urdf")
arm: nimble.dynamics.Skeleton = world.loadSkeleton("/workspaces/nimblephysics/scratch/GraspIt2URDF/urdf/Barrett.urdf")

ball = nimble.dynamics.Skeleton()
ballJoint, ballNode = ball.createFreeJointAndBodyNodePair()
ballShape = ballNode.createShapeNode(nimble.dynamics.SphereShape(.080))
ballShape.createCollisionAspect()
ballVisual = ballShape.createVisualAspect()
ballVisual.setRGBA([1.,0.,0.,1.])
world.addSkeleton(ball)

ball2 = nimble.dynamics.Skeleton()
ballJoint2, ballNode2 = ball2.createFreeJointAndBodyNodePair()
ballShape2 = ballNode2.createShapeNode(nimble.dynamics.SphereShape(.040))
ballShape2.createCollisionAspect()
ballVisual2 = ballShape2.createVisualAspect()
ballVisual2.setRGBA([0.,1.,0.,1.])
world.addSkeleton(ball2)

ball3 = nimble.dynamics.Skeleton()
ballJoint3, ballNode3 = ball3.createFreeJointAndBodyNodePair()
ballShape3 = ballNode3.createShapeNode(nimble.dynamics.SphereShape(.020))
ballShape3.createCollisionAspect()
ballVisual3 = ballShape3.createVisualAspect()
ballVisual3.setRGBA([0.,0.,1.,1.])
world.addSkeleton(ball3)

ball4 = nimble.dynamics.Skeleton()
ballJoint4, ballNode4 = ball4.createFreeJointAndBodyNodePair()
ballShape4 = ballNode4.createShapeNode(nimble.dynamics.SphereShape(.080))
ballShape4.createCollisionAspect()
ballVisual4 = ballShape4.createVisualAspect()
ballVisual4.setRGBA([0.5,0.5,0.5,0.5])
world.addSkeleton(ball4)

#square = nimble.dynamics.Skeleton()
#squareJoint, squareNode = square.createFreeJointAndBodyNodePair()
#squareShape = squareNode.createShapeNode(nimble.dynamics.BoxShape([0.01,0.01,0.01]))
#squareShape.createCollisionAspect()
#squareVisual = squareShape.createVisualAspect()
#world.addSkeleton(square)

# Set up initial state
#pdb.set_trace()
initialState = np.zeros(world.getStateSize())
quat = spin.Quaternion([+0.653281, -0.270598, -0.270598, +0.653281])
#quat = Quaternion([+0.653281, -0.270598, -0.270598, +0.653281])
#quat = Quaternion(0.3303333834279168, -0.8653453280301845, 0.21080696766652196, 0.312438380217033)
#base_to_palm_quat = Quaternion([ 4.51429801,  0.58911173, -0.87823112,  0.79981649])
#desired_palm_quat = Quaternion([+0.653281, -0.270598, -0.270598, +0.653281])
#desired_y, desired_p, desired_r = desired_palm_quat.yaw_pitch_roll
#base_joint_quat = desired_palm_quat / base_to_palm_quat
#y,p,r = base_joint_quat.yaw_pitch_roll
#pos = np.array([+56.5133, -6.40357e-14, -56.5133]) * 0.001
pos = np.array([+56.5133, -6.40357e-14, -56.5133]) * 0.001
#r,p,y = trans.euler_from_quaternion(quat.elements)
expmap = spin.ExponentialMap(quat.matrix).dofs

initialState[:6] = [expmap[0], expmap[1], expmap[2], pos[0], pos[1], pos[2]]
#initialState[6:14] = [1.0472, 1.50012, 0.50004, 1.0472, 2.32945, 0.776483, 0, 0]
#initialState[6:14] = [1.0472, 0.921875, 0.307292, 1.0472, 0.921875, 0.307292, 0.652344, 0.377603]
#initialState[6:14] = [0, 1.41795, 0.47265, 0, 0, 0, 0, 0] # barretOriginSpherePointing
initialState[6:14] = [1.0472, 0.921875, 0.307292, 1.0472, 0.921875, 0.307292, 0.652344, 0.377603] #barretSphereGrasp


quat = Quaternion([1., 0., 0., 0.])
y,p,r = quat.yaw_pitch_roll
#initialState[14:20] = [r, p, y, 0.080, 0.080, 0.080]
initialState[14:20] = [0, 0, 0, 0.2, 0.00, 0.00]
initialState[20:26] = [0, 0, 0, 0.00, 0.2, 0.00]
initialState[26:32] = [0, 0, 0, 0.00, 0.00, 0.2]
#initialState[32:38] = [0, 0, 0, 0.04, -0.00, -0.04] #looks not bad
initialState[32:38] = [0, 0, 0, 0.0, 0.0, 0.0] #looks not bad



#initialState[6:13] =0.5
#initialState[19] = 0.2
#initialState[18] = 0.05
initialState = torch.tensor(initialState, requires_grad=True)
#initialState = torch.zeros((world.getStateSize()), requires_grad=True)
action = torch.zeros((world.getActionSize()))
# apply gravity force only to ball
#action[37] = -9.81
#action[6:13] = -100.0
#action[19] = -10.0
#action[:] = 1.0


# Run a simulation for 300 timesteps
ikMap = nimble.neural.IKMapping(world)
#ikMap.addLinearBodyNode(arm.getBodyNode("palm"))
#ikMap.addAngularBodyNode(arm.getBodyNode("palm"))
ikMap.addSpatialBodyNode(arm.getBodyNode("base_link"))
ikMap.addSpatialBodyNode(arm.getBodyNode("palm"))

state = initialState
states = []
for i in range(300):
  #contacts = nimble.contacts(world, state, action)
  state = nimble.timestep(world, state, action)
  world_pos = nimble.map_to_pos(world, ikMap, state)
  print(world_pos)

  res = world.getLastCollisionResult()
  print(i)
  n_contacts = res.getNumContacts()
  print(n_contacts)
  #if n_contacts > 0:
      #c = res.getContact(0)
      #pdb.set_trace()
  #print(state)
  #l = contacts.sum()
  #l.backward()
  #print(contacts)
  states.append(state)

# Display our trajectory in a GUI

gui = nimble.NimbleGUI(world)
gui.loopStates(states)
gui.serve(8080)
gui.blockWhileServing()
