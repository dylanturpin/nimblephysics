#include "dart/trajectory/MultiShot.hpp"

#include <vector>

#include "dart/dynamics/Skeleton.hpp"
#include "dart/neural/BackpropSnapshot.hpp"
#include "dart/neural/NeuralUtils.hpp"
#include "dart/neural/RestorableSnapshot.hpp"
#include "dart/simulation/World.hpp"

using namespace dart;
using namespace dynamics;
using namespace simulation;
using namespace neural;

#define LOG_PERFORMANCE_MULTI_SHOT

namespace dart {
namespace trajectory {

//==============================================================================
MultiShot::MultiShot(
    std::shared_ptr<simulation::World> world,
    LossFn loss,
    int steps,
    int shotLength,
    bool tuneStartingState)
  : AbstractShot(world, loss, steps)
{
  mShotLength = shotLength;
  mTuneStartingState = tuneStartingState;

  int stepsRemaining = steps;
  bool isFirst = true;
  LossFn zeroLoss = LossFn();
  while (stepsRemaining > 0)
  {
    int shot = std::min(shotLength, stepsRemaining);
    mShots.push_back(std::make_shared<SingleShot>(
        world, zeroLoss, shot, !isFirst || tuneStartingState));
    stepsRemaining -= shot;
    isFirst = false;
  }
}

//==============================================================================
MultiShot::~MultiShot()
{
  // std::cout << "Freeing MultiShot: " << this << std::endl;
}

//==============================================================================
void MultiShot::setParallelOperationsEnabled(bool enabled)
{
  mParallelOperationsEnabled = enabled;
  if (enabled)
  {
    // Before using Eigen in a multi-threaded environment, we need to explicitly
    // call this (at least prior to Eigen 3.3)
    Eigen::initParallel();
  }
}

//==============================================================================
/// This sets the mapping we're using to store the representation of the Shot.
/// WARNING: THIS IS A POTENTIALLY DESTRUCTIVE OPERATION! This will rewrite
/// the internal representation of the Shot to use the new mapping, and if the
/// new mapping is underspecified compared to the old mapping, you may lose
/// information. It's not guaranteed that you'll get back the same trajectory
/// if you switch to a different mapping, and then switch back.
///
/// This will affect the values you get back from getStates() - they'll now be
/// returned in the view given by `mapping`. That's also the represenation
/// that'll be passed to IPOPT, and updated on each gradient step. Therein
/// lies the power of changing the representation mapping: There will almost
/// certainly be mapped spaces that are easier to optimize in than native
/// joint space, at least initially.
void MultiShot::switchRepresentationMapping(
    std::shared_ptr<simulation::World> world,
    const std::string& mapping,
    PerformanceLog* log)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("MultiShot.switchRepresentationMapping");
  }
#endif

  for (auto shot : mShots)
  {
    shot->switchRepresentationMapping(world, mapping, thisLog);
  }
  AbstractShot::switchRepresentationMapping(world, mapping, thisLog);

#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This adds a mapping through which the loss function can interpret the
/// output. We can have multiple loss mappings at the same time, and loss can
/// use arbitrary combinations of multiple views, as long as it can provide
/// gradients.
void MultiShot::addMapping(
    const std::string& key, std::shared_ptr<neural::Mapping> mapping)
{
  AbstractShot::addMapping(key, mapping);
  for (const std::shared_ptr<SingleShot> shot : mShots)
  {
    shot->addMapping(key, mapping);
  }
}

//==============================================================================
/// This removes the loss mapping at a particular key
void MultiShot::removeMapping(const std::string& key)
{
  AbstractShot::removeMapping(key);
  for (const std::shared_ptr<SingleShot> shot : mShots)
  {
    shot->removeMapping(key);
  }
}

//==============================================================================
/// Returns the length of the flattened problem state
int MultiShot::getFlatProblemDim() const
{
  int sum = 0;
  for (const std::shared_ptr<SingleShot> shot : mShots)
  {
    sum += shot->getFlatProblemDim();
  }
  return sum;
}

//==============================================================================
/// Returns the length of the knot-point constraint vector
int MultiShot::getConstraintDim() const
{
  return AbstractShot::getConstraintDim()
         + getRepresentationStateSize() * (mShots.size() - 1);
}

//==============================================================================
/// This computes the values of the constraints
void MultiShot::computeConstraints(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> constraints,
    PerformanceLog* log)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("MultiShot.computeConstraints");
  }
#endif

  int cursor = 0;
  int numParentConstraints = AbstractShot::getConstraintDim();
  AbstractShot::computeConstraints(
      world, constraints.segment(0, numParentConstraints), thisLog);
  cursor += numParentConstraints;
  for (int i = 1; i < mShots.size(); i++)
  {
    constraints.segment(cursor, getRepresentationStateSize())
        = mShots[i - 1]->getFinalState(world, thisLog)
          - mShots[i]->getStartState();
    cursor += getRepresentationStateSize();
  }

#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This copies a shot down into a single flat vector
void MultiShot::flatten(
    /* OUT */ Eigen::Ref<Eigen::VectorXd> flat, PerformanceLog* log) const
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("MultiShot.flatten");
  }
#endif

  int cursor = 0;
  for (const std::shared_ptr<SingleShot>& shot : mShots)
  {
    int dim = shot->getFlatProblemDim();
    shot->flatten(flat.segment(cursor, dim), thisLog);
    cursor += dim;
  }

#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This gets the parameters out of a flat vector
void MultiShot::unflatten(
    const Eigen::Ref<const Eigen::VectorXd>& flat, PerformanceLog* log)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("MultiShot.unflatten");
  }
#endif

  mRolloutCacheDirty = true;
  int cursor = 0;
  for (std::shared_ptr<SingleShot>& shot : mShots)
  {
    int dim = shot->getFlatProblemDim();
    shot->unflatten(flat.segment(cursor, dim), thisLog);
    cursor += dim;
  }

#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This gets the fixed upper bounds for a flat vector, used during
/// optimization
void MultiShot::getUpperBounds(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> flat,
    PerformanceLog* log) const
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("MultiShot.getUpperBounds");
  }
#endif

  int cursor = 0;
  for (const std::shared_ptr<SingleShot>& shot : mShots)
  {
    int dim = shot->getFlatProblemDim();
    shot->getUpperBounds(world, flat.segment(cursor, dim), thisLog);
    cursor += dim;
  }

#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This gets the fixed lower bounds for a flat vector, used during
/// optimization
void MultiShot::getLowerBounds(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> flat,
    PerformanceLog* log) const
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("MultiShot.getLowerBounds");
  }
#endif

  int cursor = 0;
  for (const std::shared_ptr<SingleShot>& shot : mShots)
  {
    int dim = shot->getFlatProblemDim();
    shot->getLowerBounds(world, flat.segment(cursor, dim), thisLog);
    cursor += dim;
  }

#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This gets the bounds on the constraint functions (both knot points and any
/// custom constraints)
void MultiShot::getConstraintUpperBounds(
    /* OUT */ Eigen::Ref<Eigen::VectorXd> flat, PerformanceLog* log) const
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("MultiShot.getConstraintUpperBounds");
  }
#endif

  flat.setZero();
  AbstractShot::getConstraintUpperBounds(
      flat.segment(0, AbstractShot::getConstraintDim()), thisLog);

#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This gets the bounds on the constraint functions (both knot points and any
/// custom constraints)
void MultiShot::getConstraintLowerBounds(
    /* OUT */ Eigen::Ref<Eigen::VectorXd> flat, PerformanceLog* log) const
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("MultiShot.getConstraintLowerBounds");
  }
#endif

  flat.setZero();
  AbstractShot::getConstraintLowerBounds(
      flat.segment(0, AbstractShot::getConstraintDim()), thisLog);

#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This returns the initial guess for the values of X when running an
/// optimization
void MultiShot::getInitialGuess(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> flat,
    PerformanceLog* log) const
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("MultiShot.getInitialGuess");
  }
#endif

  int cursor = 0;
  for (const std::shared_ptr<SingleShot>& shot : mShots)
  {
    int dim = shot->getFlatProblemDim();
    shot->getInitialGuess(world, flat.segment(cursor, dim), thisLog);
    cursor += dim;
  }

#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This computes the Jacobian that relates the flat problem to the
/// constraints. This returns a matrix that's (getConstraintDim(),
/// getFlatProblemDim()).
void MultiShot::backpropJacobian(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::MatrixXd> jac,
    PerformanceLog* log)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("MultiShot.backpropJacobian");
  }
#endif

  assert(jac.cols() == getFlatProblemDim());
  assert(jac.rows() == getConstraintDim());

  int rowCursor = 0;
  int colCursor = 0;

  jac.setZero();

  // Handle custom constraints
  int numParentConstraints = AbstractShot::getConstraintDim();
  int n = getFlatProblemDim();
  AbstractShot::backpropJacobian(
      world, jac.block(0, 0, numParentConstraints, n), thisLog);
  rowCursor += numParentConstraints;

  // Add in knot point constraints
  int stateDim = getRepresentationStateSize();
  for (int i = 1; i < mShots.size(); i++)
  {
    int dim = mShots[i - 1]->getFlatProblemDim();
    mShots[i - 1]->backpropJacobianOfFinalState(
        world, jac.block(rowCursor, colCursor, stateDim, dim), thisLog);
    colCursor += dim;
    jac.block(rowCursor, colCursor, stateDim, stateDim)
        = -1 * Eigen::MatrixXd::Identity(stateDim, stateDim);
    rowCursor += stateDim;
  }

  // We don't include the last shot in the constraints, cause it doesn't end in
  // a knot point
  assert(
      colCursor == jac.cols() - mShots[mShots.size() - 1]->getFlatProblemDim());
  assert(rowCursor == jac.rows());

#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This gets the number of non-zero entries in the Jacobian
int MultiShot::getNumberNonZeroJacobian()
{
  int nnzj = AbstractShot::getNumberNonZeroJacobian();
  int stateDim = getRepresentationStateSize();
  for (int i = 0; i < mShots.size() - 1; i++)
  {
    int shotDim = mShots[i]->getFlatProblemDim();
    // The main Jacobian block
    nnzj += shotDim * stateDim;
    // The -I at the end
    nnzj += stateDim;
  }

  return nnzj;
}

//==============================================================================
/// This gets the structure of the non-zero entries in the Jacobian
void MultiShot::getJacobianSparsityStructure(
    Eigen::Ref<Eigen::VectorXi> rows,
    Eigen::Ref<Eigen::VectorXi> cols,
    PerformanceLog* log)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("MultiShot.getJacobianSparsityStructure");
  }
#endif

  int sparseCursor = 0;
  int rowCursor = 0;
  int colCursor = 0;

  // Handle custom constraints
  int numParentConstraints = AbstractShot::getConstraintDim();
  int n = getFlatProblemDim();
  AbstractShot::getJacobianSparsityStructure(
      rows.segment(0, n * numParentConstraints),
      cols.segment(0, n * numParentConstraints),
      thisLog);
  rowCursor += numParentConstraints;
  sparseCursor += n * numParentConstraints;

  int stateDim = getRepresentationStateSize();
  // Handle knot point constraints
  for (int i = 1; i < mShots.size(); i++)
  {
    int dim = mShots[i - 1]->getFlatProblemDim();
    // This is the main Jacobian
    for (int col = colCursor; col < colCursor + dim; col++)
    {
      for (int row = rowCursor; row < rowCursor + stateDim; row++)
      {
        rows(sparseCursor) = row;
        cols(sparseCursor) = col;
        sparseCursor++;
      }
    }
    colCursor += dim;
    // This is the negative identity at the end
    for (int q = 0; q < stateDim; q++)
    {
      rows(sparseCursor) = rowCursor + q;
      cols(sparseCursor) = colCursor + q;
      sparseCursor++;
    }
    rowCursor += stateDim;
  }

#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This writes the Jacobian to a sparse vector
void MultiShot::getSparseJacobian(
    std::shared_ptr<simulation::World> world,
    Eigen::Ref<Eigen::VectorXd> sparse,
    PerformanceLog* log)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("MultiShot.getSparseJacobian");
  }
#endif

  int sparseCursor = AbstractShot::getNumberNonZeroJacobian();
  AbstractShot::getSparseJacobian(
      world, sparse.segment(0, sparseCursor), thisLog);
  int stateDim = getRepresentationStateSize();
  Eigen::VectorXd neg = Eigen::VectorXd::Ones(stateDim) * -1;
  for (int i = 1; i < mShots.size(); i++)
  {
    int dim = mShots[i - 1]->getFlatProblemDim();
    // This is the main Jacobian
    Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(stateDim, dim);
    mShots[i - 1]->backpropJacobianOfFinalState(world, jac, thisLog);
    for (int col = 0; col < dim; col++)
    {
      sparse.segment(sparseCursor, stateDim) = jac.col(col);
      sparseCursor += stateDim;
    }
    // This is the negative identity at the end
    sparse.segment(sparseCursor, stateDim) = neg;
    sparseCursor += stateDim;
  }

#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This populates the passed in matrices with the values from this trajectory
void MultiShot::getStates(
    std::shared_ptr<simulation::World> world,
    /* OUT */ TrajectoryRollout* rollout,
    PerformanceLog* log,
    bool useKnots)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("MultiShot.getStates");
  }
#endif

  int posDim = getRepresentation()->getPosDim();
  int velDim = getRepresentation()->getVelDim();
  int forceDim = getRepresentation()->getForceDim();
  assert(rollout->getPoses(mRepresentationMapping).cols() == mSteps);
  assert(rollout->getPoses(mRepresentationMapping).rows() == posDim);
  assert(rollout->getVels(mRepresentationMapping).cols() == mSteps);
  assert(rollout->getVels(mRepresentationMapping).rows() == velDim);
  assert(rollout->getForces(mRepresentationMapping).cols() == mSteps);
  assert(rollout->getForces(mRepresentationMapping).rows() == forceDim);
  int cursor = 0;
  if (useKnots)
  {
    for (int i = 0; i < mShots.size(); i++)
    {
      int steps = mShots[i]->getNumSteps();
      TrajectoryRolloutRef slice = rollout->slice(cursor, steps);
      mShots[i]->getStates(world, &slice, thisLog, true);
      cursor += steps;
    }
  }
  else
  {
    RestorableSnapshot snapshot(world);
    getRepresentation()->setPositions(world, mShots[0]->mStartPos);
    getRepresentation()->setVelocities(world, mShots[0]->mStartVel);
    for (int i = 0; i < mShots.size(); i++)
    {
      for (int j = 0; j < mShots[i]->mSteps; j++)
      {
        Eigen::VectorXd forces = mShots[i]->mForces.col(j);
        getRepresentation()->setForces(world, forces);
        world->step();
        for (auto pair : mMappings)
        {
          rollout->getPoses(pair.first).col(cursor)
              = pair.second->getPositions(world);
          rollout->getVels(pair.first).col(cursor)
              = pair.second->getVelocities(world);
          rollout->getForces(pair.first).col(cursor)
              = pair.second->getForces(world);
        }
        cursor++;
      }
    }
  }
  assert(cursor == mSteps);

#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This returns the concatenation of (start pos, start vel) for convenience
Eigen::VectorXd MultiShot::getStartState()
{
  return mShots[0]->getStartState();
}

//==============================================================================
/// This unrolls the shot, and returns the (pos, vel) state concatenated at
/// the end of the shot
Eigen::VectorXd MultiShot::getFinalState(
    std::shared_ptr<simulation::World> world, PerformanceLog* log)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("MultiShot.getFinalState");
  }
#endif

  Eigen::VectorXd ret
      = mShots[mShots.size() - 1]->getFinalState(world, thisLog);

#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif

  return ret;
}

//==============================================================================
/// This returns the debugging name of a given DOF
std::string MultiShot::getFlatDimName(int dim)
{
  for (int i = 0; i < mShots.size(); i++)
  {
    int shotDim = mShots[i]->getFlatProblemDim();
    if (dim < shotDim)
    {
      return "Shot " + std::to_string(i) + " " + mShots[i]->getFlatDimName(dim);
    }
    dim -= shotDim;
  }
  return "Error OOB";
}

//==============================================================================
/// This computes the gradient in the flat problem space, taking into accounts
/// incoming gradients with respect to any of the shot's values.
void MultiShot::backpropGradientWrt(
    std::shared_ptr<simulation::World> world,
    const TrajectoryRollout* gradWrtForces,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> grad,
    PerformanceLog* log)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("MultiShot.backpropGradientWrt");
  }
#endif

  int cursorDims = 0;
  int cursorSteps = 0;
  for (int i = 0; i < mShots.size(); i++)
  {
    int steps = mShots[i]->getNumSteps();
    int dim = mShots[i]->getFlatProblemDim();
    const TrajectoryRolloutConstRef slice
        = gradWrtForces->sliceConst(cursorSteps, steps);
    mShots[i]->backpropGradientWrt(
        world, &slice, grad.segment(cursorDims, dim), thisLog);
    cursorSteps += steps;
    cursorDims += dim;
  }

#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

} // namespace trajectory
} // namespace dart