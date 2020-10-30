#include "dart/trajectory/AbstractShot.hpp"

#include <iostream>

#include <coin/IpIpoptApplication.hpp>
#include <coin/IpReturnCodes.hpp>
#include <coin/IpSolveStatistics.hpp>

#include "dart/neural/IdentityMapping.hpp"
#include "dart/neural/Mapping.hpp"
#include "dart/simulation/World.hpp"

#define LOG_PERFORMANCE_ABSTRACT_SHOT

namespace dart {
namespace trajectory {

//==============================================================================
/// Default constructor
AbstractShot::AbstractShot(
    std::shared_ptr<simulation::World> world, LossFn loss, int steps)
  : mWorld(world), mLoss(loss), mSteps(steps), mRolloutCacheDirty(true)
{
  std::shared_ptr<neural::Mapping> identityMapping
      = std::make_shared<neural::IdentityMapping>(world);
  mRepresentationMapping = "identity";
  mMappings[mRepresentationMapping] = identityMapping;
}

//==============================================================================
AbstractShot::~AbstractShot()
{
  // std::cout << "Freeing AbstractShot: " << this << std::endl;
}

//==============================================================================
/// This updates the loss function for this trajectory
void AbstractShot::setLoss(LossFn loss)
{
  mLoss = loss;
}

//==============================================================================
/// Add a custom constraint function to the trajectory
void AbstractShot::addConstraint(LossFn loss)
{
  mConstraints.push_back(loss);
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
void AbstractShot::switchRepresentationMapping(
    std::shared_ptr<simulation::World> world,
    const std::string& mapping,
    PerformanceLog* log)
{
  // Reset the main representation mapping
  mRepresentationMapping = mapping;
  // Clear our cached trajectory
  mRolloutCacheDirty = true;
}

//==============================================================================
/// This adds a mapping through which the loss function can interpret the
/// output. We can have multiple loss mappings at the same time, and loss can
/// use arbitrary combinations of multiple views, as long as it can provide
/// gradients.
void AbstractShot::addMapping(
    const std::string& key, std::shared_ptr<neural::Mapping> mapping)
{
  mMappings[key] = mapping;
  // Clear our cached trajectory
  mRolloutCacheDirty = true;
}

//==============================================================================
/// This returns true if there is a loss mapping at the specified key
bool AbstractShot::hasMapping(const std::string& key)
{
  return mMappings.find(key) != mMappings.end();
}

//==============================================================================
/// This returns the loss mapping at the specified key
std::shared_ptr<neural::Mapping> AbstractShot::getMapping(
    const std::string& key)
{
  return mMappings[key];
}

//==============================================================================
/// This returns a reference to all the mappings in this shot
std::unordered_map<std::string, std::shared_ptr<neural::Mapping>>&
AbstractShot::getMappings()
{
  return mMappings;
}

//==============================================================================
/// This removes the loss mapping at a particular key
void AbstractShot::removeMapping(const std::string& key)
{
  mMappings.erase(key);
  // Clear our cached trajectory
  mRolloutCacheDirty = true;
}

//==============================================================================
/// Returns the sum of posDim() + velDim() for the current representation
/// mapping
int AbstractShot::getRepresentationStateSize() const
{
  return getRepresentation()->getPosDim() + getRepresentation()->getVelDim();
}

//==============================================================================
const std::string& AbstractShot::getRepresentationName() const
{
  return mRepresentationMapping;
}

//==============================================================================
/// Returns the representation currently being used
const std::shared_ptr<neural::Mapping> AbstractShot::getRepresentation() const
{
  return mMappings.at(mRepresentationMapping);
}

//==============================================================================
/// Returns the length of the flattened problem state
int AbstractShot::getFlatProblemDim(
    std::shared_ptr<simulation::World> world) const
{
  return getFlatStaticProblemDim(world) + getFlatDynamicProblemDim(world);
}

//==============================================================================
int AbstractShot::getFlatStaticProblemDim(
    std::shared_ptr<simulation::World> world) const
{
  return mWorld->getWrtMass()->dim(mWorld);
}

//==============================================================================
int AbstractShot::getFlatDynamicProblemDim(
    std::shared_ptr<simulation::World> world) const
{
  return 0;
}

//==============================================================================
/// This copies a shot down into a single flat vector
void AbstractShot::flatten(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> flatStatic,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> flatDynamic,
    PerformanceLog* log) const
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_ABSTRACT_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("AbstractShot.flatten");
  }
#endif

  flatStatic.segment(0, world->getWrtMass()->dim(world))
      = world->getWrtMass()->get(world);

#ifdef LOG_PERFORMANCE_ABSTRACT_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This copies a shot down into a single flat vector
void AbstractShot::flatten(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> flat,
    PerformanceLog* log) const
{
  int staticDim = getFlatStaticProblemDim(world);
  int dynamicDim = getFlatDynamicProblemDim(world);
  flatten(
      world,
      flat.segment(0, staticDim),
      flat.segment(staticDim, dynamicDim),
      log);
}

//==============================================================================
/// This gets the parameters out of a flat vector
void AbstractShot::unflatten(
    std::shared_ptr<simulation::World> world,
    const Eigen::Ref<const Eigen::VectorXd>& flatStatic,
    const Eigen::Ref<const Eigen::VectorXd>& flatDynamic,
    PerformanceLog* log)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_ABSTRACT_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("AbstractShot.unflatten");
  }
#endif

  world->getWrtMass()->set(
      world, flatStatic.segment(0, world->getWrtMass()->dim(world)));

#ifdef LOG_PERFORMANCE_ABSTRACT_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This gets the parameters out of a flat vector
void AbstractShot::unflatten(
    std::shared_ptr<simulation::World> world,
    const Eigen::Ref<const Eigen::VectorXd>& flat,
    PerformanceLog* log)
{
  int staticDim = getFlatStaticProblemDim(world);
  int dynamicDim = getFlatDynamicProblemDim(world);
  unflatten(
      world,
      flat.segment(0, staticDim),
      flat.segment(staticDim, dynamicDim),
      log);
}

//==============================================================================
/// This gets the fixed upper bounds for a flat vector, used during
/// optimization
void AbstractShot::getUpperBounds(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> flat,
    PerformanceLog* log) const
{
  int staticDim = getFlatStaticProblemDim(world);
  int dynamicDim = getFlatDynamicProblemDim(world);
  getUpperBounds(
      world,
      flat.segment(0, staticDim),
      flat.segment(staticDim, dynamicDim),
      log);
}

//==============================================================================
/// This gets the fixed lower bounds for a flat vector, used during
/// optimization
void AbstractShot::getLowerBounds(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> flat,
    PerformanceLog* log) const
{
  int staticDim = getFlatStaticProblemDim(world);
  int dynamicDim = getFlatDynamicProblemDim(world);
  getLowerBounds(
      world,
      flat.segment(0, staticDim),
      flat.segment(staticDim, dynamicDim),
      log);
}

//==============================================================================
/// This gets the fixed upper bounds for a flat vector, used during
/// optimization
void AbstractShot::getUpperBounds(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> flatStatic,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> flatDynamic,
    PerformanceLog* log) const
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_ABSTRACT_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("AbstractShot.getUpperBounds");
  }
#endif

  flatStatic.segment(0, world->getWrtMass()->dim(world))
      = world->getWrtMass()->upperBound(world);

#ifdef LOG_PERFORMANCE_ABSTRACT_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This gets the fixed lower bounds for a flat vector, used during
/// optimization
void AbstractShot::getLowerBounds(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> flatStatic,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> flatDynamic,
    PerformanceLog* log) const
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_ABSTRACT_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("AbstractShot.getLowerBounds");
  }
#endif

  flatStatic.segment(0, world->getWrtMass()->dim(world))
      = world->getWrtMass()->lowerBound(world);

#ifdef LOG_PERFORMANCE_ABSTRACT_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This returns the initial guess for the values of X when running an
/// optimization
void AbstractShot::getInitialGuess(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> flatStatic,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> flatDynamic,
    PerformanceLog* log) const
{
  flatStatic.segment(0, world->getWrtMass()->dim(world))
      = world->getWrtMass()->get(world);
}

//==============================================================================
/// This gets the bounds on the constraint functions (both knot points and any
/// custom constraints)
void AbstractShot::getConstraintUpperBounds(
    /* OUT */ Eigen::Ref<Eigen::VectorXd> flat, PerformanceLog* log) const
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_ABSTRACT_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("AbstractShot.getConstraintUpperBounds");
  }
#endif

  assert(flat.size() == mConstraints.size());
  for (int i = 0; i < mConstraints.size(); i++)
  {
    flat(i) = mConstraints[i].getUpperBound();
  }

#ifdef LOG_PERFORMANCE_ABSTRACT_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This gets the bounds on the constraint functions (both knot points and any
/// custom constraints)
void AbstractShot::getConstraintLowerBounds(
    /* OUT */ Eigen::Ref<Eigen::VectorXd> flat, PerformanceLog* log) const
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_ABSTRACT_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("AbstractShot.getConstraintLowerBounds");
  }
#endif

  assert(flat.size() == mConstraints.size());
  for (int i = 0; i < mConstraints.size(); i++)
  {
    flat(i) = mConstraints[i].getLowerBound();
  }

#ifdef LOG_PERFORMANCE_ABSTRACT_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This returns the initial guess for the values of X when running an
/// optimization
void AbstractShot::getInitialGuess(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> flat,
    PerformanceLog* log) const
{
  int staticDim = getFlatStaticProblemDim(world);
  int dynamicDim = getFlatDynamicProblemDim(world);
  getInitialGuess(
      world,
      flat.segment(0, staticDim),
      flat.segment(staticDim, dynamicDim),
      log);
}

//==============================================================================
int AbstractShot::getConstraintDim() const
{
  return mConstraints.size();
}

//==============================================================================
/// This computes the values of the constraints, assuming that the constraint
/// vector being passed in is only the size of mConstraints
void AbstractShot::computeConstraints(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> constraints,
    PerformanceLog* log)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_ABSTRACT_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("AbstractShot.computeConstraints");
  }
#endif

  assert(constraints.size() == mConstraints.size());

  for (int i = 0; i < mConstraints.size(); i++)
  {
    constraints(i)
        = mConstraints[i].getLoss(getRolloutCache(world, thisLog), thisLog);
  }

#ifdef LOG_PERFORMANCE_ABSTRACT_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This computes the Jacobian that relates the flat problem to the end state.
/// This returns a matrix that's (getConstraintDim(), getFlatProblemDim()).
void AbstractShot::backpropJacobian(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::MatrixXd> jac,
    PerformanceLog* log)
{
  int staticDim = getFlatStaticProblemDim(world);
  int dynamicDim = getFlatDynamicProblemDim(world);
  int numConstraints = getConstraintDim();
  assert(jac.rows() == numConstraints);
  assert(jac.cols() == staticDim + dynamicDim);
  backpropJacobian(
      world,
      jac.block(0, 0, numConstraints, staticDim),
      jac.block(0, staticDim, numConstraints, dynamicDim),
      log);
}

//==============================================================================
/// This computes the Jacobian that relates the flat problem to the end state.
/// This returns a matrix that's (getConstraintDim(), getFlatProblemDim()).
void AbstractShot::backpropJacobian(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::MatrixXd> jacStatic,
    /* OUT */ Eigen::Ref<Eigen::MatrixXd> jacDynamic,
    PerformanceLog* log)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_ABSTRACT_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("AbstractShot.backpropJacobian");
  }
#endif

  assert(jacStatic.rows() == mConstraints.size());
  assert(jacStatic.cols() == getFlatStaticProblemDim(world));
  assert(jacDynamic.rows() == mConstraints.size());
  assert(jacDynamic.cols() == getFlatDynamicProblemDim(world));

  Eigen::VectorXd gradStatic
      = Eigen::VectorXd::Zero(getFlatStaticProblemDim(world));
  Eigen::VectorXd gradDynamic
      = Eigen::VectorXd::Zero(getFlatDynamicProblemDim(world));
  for (int i = 0; i < mConstraints.size(); i++)
  {
    mConstraints[i].getLossAndGradient(
        getRolloutCache(world, thisLog),
        /* OUT */ getGradientWrtRolloutCache(world, thisLog),
        thisLog);
    gradStatic.setZero();
    gradDynamic.setZero();
    backpropGradientWrt(
        world,
        getGradientWrtRolloutCache(world, thisLog),
        /* OUT */ gradStatic,
        /* OUT */ gradDynamic,
        thisLog);
    jacDynamic.row(i) = gradDynamic;
    jacStatic.row(i) = gradStatic;
  }

#ifdef LOG_PERFORMANCE_ABSTRACT_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This gets the number of non-zero entries in the Jacobian
int AbstractShot::getNumberNonZeroJacobianStatic(
    std::shared_ptr<simulation::World> world)
{
  return mConstraints.size() * getFlatStaticProblemDim(world);
}

//==============================================================================
/// This gets the number of non-zero entries in the Jacobian
int AbstractShot::getNumberNonZeroJacobianDynamic(
    std::shared_ptr<simulation::World> world)
{
  return mConstraints.size() * getFlatDynamicProblemDim(world);
}

//==============================================================================
/// This gets the structure of the non-zero entries in the Jacobian
void AbstractShot::getJacobianSparsityStructure(
    std::shared_ptr<simulation::World> world,
    Eigen::Ref<Eigen::VectorXi> rows,
    Eigen::Ref<Eigen::VectorXi> cols,
    PerformanceLog* log)
{
  int nnzjStatic = getNumberNonZeroJacobianStatic(world);
  int nnzjDynamic = getNumberNonZeroJacobianDynamic(world);
  assert(
      nnzjStatic + nnzjDynamic == rows.size()
      && nnzjStatic + nnzjDynamic == cols.size());
  getJacobianSparsityStructureStatic(
      world, rows.segment(0, nnzjStatic), cols.segment(0, nnzjStatic), log);
  getJacobianSparsityStructureDynamic(
      world,
      rows.segment(nnzjStatic, nnzjDynamic),
      cols.segment(nnzjStatic, nnzjDynamic),
      log);
  // Bump all the dynamic elements over by `staticCols`
  int staticCols = getFlatStaticProblemDim(world);
  cols.segment(nnzjStatic, nnzjDynamic)
      += Eigen::VectorXi::Ones(nnzjDynamic) * staticCols;
}

//==============================================================================
/// This gets the structure of the non-zero entries in the Jacobian
void AbstractShot::getJacobianSparsityStructureDynamic(
    std::shared_ptr<simulation::World> world,
    Eigen::Ref<Eigen::VectorXi> rows,
    Eigen::Ref<Eigen::VectorXi> cols,
    PerformanceLog* log)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_ABSTRACT_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("AbstractShot.getJacobianSparsityStructure");
  }
#endif

  assert(rows.size() == AbstractShot::getNumberNonZeroJacobianDynamic(world));
  assert(cols.size() == AbstractShot::getNumberNonZeroJacobianDynamic(world));
  int cursor = 0;
  // Do row-major ordering
  for (int j = 0; j < mConstraints.size(); j++)
  {
    for (int i = 0; i < getFlatDynamicProblemDim(world); i++)
    {
      rows(cursor) = j;
      cols(cursor) = i;
      cursor++;
    }
  }
  assert(cursor == AbstractShot::getNumberNonZeroJacobianDynamic(world));

#ifdef LOG_PERFORMANCE_ABSTRACT_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This gets the structure of the non-zero entries in the Jacobian
void AbstractShot::getJacobianSparsityStructureStatic(
    std::shared_ptr<simulation::World> world,
    Eigen::Ref<Eigen::VectorXi> rows,
    Eigen::Ref<Eigen::VectorXi> cols,
    PerformanceLog* log)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_ABSTRACT_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("AbstractShot.getJacobianSparsityStructure");
  }
#endif

  assert(rows.size() == AbstractShot::getNumberNonZeroJacobianStatic(world));
  assert(cols.size() == AbstractShot::getNumberNonZeroJacobianStatic(world));
  int cursor = 0;
  // Do row-major ordering
  for (int j = 0; j < mConstraints.size(); j++)
  {
    for (int i = 0; i < getFlatStaticProblemDim(world); i++)
    {
      rows(cursor) = j;
      cols(cursor) = i;
      cursor++;
    }
  }
  assert(cursor == AbstractShot::getNumberNonZeroJacobianStatic(world));

#ifdef LOG_PERFORMANCE_ABSTRACT_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This writes the Jacobian to a pair of sparse vectors, separating out the
/// static and dynamic regions.
void AbstractShot::getSparseJacobian(
    std::shared_ptr<simulation::World> world,
    Eigen::Ref<Eigen::VectorXd> sparseStatic,
    Eigen::Ref<Eigen::VectorXd> sparseDynamic,
    PerformanceLog* log)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_ABSTRACT_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("AbstractShot.getSparseJacobian");
  }
#endif

  assert(
      sparseStatic.size()
      == AbstractShot::getNumberNonZeroJacobianStatic(world));
  assert(
      sparseDynamic.size()
      == AbstractShot::getNumberNonZeroJacobianDynamic(world));

  sparseStatic.setZero();
  sparseDynamic.setZero();

  int cursorDynamic = 0;
  int cursorStatic = 0;
  int nStatic = getFlatStaticProblemDim(world);
  int nDynamic = getFlatDynamicProblemDim(world);
  for (int i = 0; i < mConstraints.size(); i++)
  {
    mConstraints[i].getLossAndGradient(
        getRolloutCache(world, thisLog),
        /* OUT */ getGradientWrtRolloutCache(world, thisLog),
        thisLog);
    backpropGradientWrt(
        world,
        getGradientWrtRolloutCache(world, thisLog),
        /* OUT */ sparseStatic.segment(cursorStatic, nStatic),
        /* OUT */ sparseDynamic.segment(cursorDynamic, nDynamic),
        thisLog);
    cursorStatic += nStatic;
    cursorDynamic += nDynamic;
  }

  assert(cursorStatic == sparseStatic.size());
  assert(cursorDynamic == sparseDynamic.size());

#ifdef LOG_PERFORMANCE_ABSTRACT_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This writes the Jacobian to a sparse vector
void AbstractShot::getSparseJacobian(
    std::shared_ptr<simulation::World> world,
    Eigen::Ref<Eigen::VectorXd> sparse,
    PerformanceLog* log)
{
  int nnzjStatic = getNumberNonZeroJacobianStatic(world);
  int nnzjDynamic = getNumberNonZeroJacobianDynamic(world);
  // Simply concatenate the two results together
  getSparseJacobian(
      world,
      sparse.segment(0, nnzjStatic),
      sparse.segment(nnzjStatic, nnzjDynamic),
      log);
}

//==============================================================================
/// This computes the gradient in the flat problem space, automatically
/// computing the gradients of the loss function as part of the call
void AbstractShot::backpropGradient(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> grad,
    PerformanceLog* log)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_ABSTRACT_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("AbstractShot.backpropGradient");
  }
#endif

  int staticDim = getFlatStaticProblemDim(world);
  int dynamicDim = getFlatDynamicProblemDim(world);

  mLoss.getLossAndGradient(
      getRolloutCache(world, thisLog),
      /* OUT */ getGradientWrtRolloutCache(world, thisLog),
      thisLog);
  backpropGradientWrt(
      world,
      getGradientWrtRolloutCache(world, thisLog),
      /* OUT */ grad.segment(0, staticDim),
      /* OUT */ grad.segment(staticDim, dynamicDim),
      thisLog);

#ifdef LOG_PERFORMANCE_ABSTRACT_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This computes the gradient in the flat problem space, taking into accounts
/// incoming gradients with respect to any of the shot's values.
void AbstractShot::backpropGradientWrt(
    std::shared_ptr<simulation::World> world,
    const TrajectoryRollout* gradWrtRollout,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> grad,
    PerformanceLog* log)
{
  int staticDim = getFlatStaticProblemDim(world);
  int dynamicDim = getFlatDynamicProblemDim(world);
  backpropGradientWrt(
      world,
      gradWrtRollout,
      grad.segment(0, staticDim),
      grad.segment(staticDim, dynamicDim),
      log);
}

//==============================================================================
/// Get the loss for the rollout
double AbstractShot::getLoss(
    std::shared_ptr<simulation::World> world, PerformanceLog* log)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_ABSTRACT_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("AbstractShot.getLoss");
  }
#endif

  double val = mLoss.getLoss(getRolloutCache(world, thisLog), thisLog);

#ifdef LOG_PERFORMANCE_ABSTRACT_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif

  return val;
}

//==============================================================================
/// This gets called at the beginning of backpropGradientWrt(), as an
/// opportunity to zero out any static gradient values being managed by
/// AbstractShot.
void AbstractShot::initializeStaticGradient(
    std::shared_ptr<simulation::World> world,
    Eigen::Ref<Eigen::VectorXd> gradStatic,
    PerformanceLog* log)
{
  gradStatic.segment(0, world->getWrtMass()->dim(world)).setZero();
}

//==============================================================================
/// This adds anything to the static gradient that we need to. It needs to be
/// called for every timestep during backpropGradientWrt().
void AbstractShot::accumulateStaticGradient(
    std::shared_ptr<simulation::World> world,
    Eigen::Ref<Eigen::VectorXd> gradStatic,
    neural::LossGradient& thisTimestep,
    PerformanceLog* log)
{
  gradStatic.segment(0, world->getWrtMass()->dim(world))
      += thisTimestep.lossWrtMass;
}

//==============================================================================
/// This gets called at the beginning of backpropJacobianOfFinalState() in
/// SingleShot, as an opportunity to zero out any static jacobian values being
/// managed by AbstractShot.
void AbstractShot::initializeStaticJacobianOfFinalState(
    std::shared_ptr<simulation::World> world,
    Eigen::Ref<Eigen::MatrixXd> jacStatic,
    PerformanceLog* log)
{
  jacStatic.setZero();
}

//==============================================================================
/// This adds anything to the static gradient that we need to. It needs to be
/// called for every timestep during backpropJacobianOfFinalState() in
/// SingleShot.
void AbstractShot::accumulateStaticJacobianOfFinalState(
    std::shared_ptr<simulation::World> world,
    Eigen::Ref<Eigen::MatrixXd> jacStatic,
    TimestepJacobians& thisTimestep,
    PerformanceLog* log)
{
  jacStatic.block(
      0, 0, thisTimestep.massPos.rows(), thisTimestep.massPos.cols())
      += thisTimestep.massPos;
  jacStatic.block(
      thisTimestep.massPos.rows(),
      0,
      thisTimestep.massVel.rows(),
      thisTimestep.massVel.cols())
      += thisTimestep.massVel;
}

//==============================================================================
const TrajectoryRollout* AbstractShot::getRolloutCache(
    std::shared_ptr<simulation::World> world,
    PerformanceLog* log,
    bool useKnots)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_ABSTRACT_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("AbstractShot.getRolloutCache");
  }
#endif

  if (mRolloutCacheDirty)
  {
    mRolloutCache = std::make_shared<TrajectoryRolloutReal>(this);
    getStates(
        world,
        /* OUT */ mRolloutCache.get(),
        thisLog);
    mGradWrtRolloutCache = std::make_shared<TrajectoryRolloutReal>(this);
    mRolloutCacheDirty = false;
  }

#ifdef LOG_PERFORMANCE_ABSTRACT_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif

  return mRolloutCache.get();
}

//==============================================================================
TrajectoryRollout* AbstractShot::getGradientWrtRolloutCache(
    std::shared_ptr<simulation::World> world,
    PerformanceLog* log,
    bool useKnots)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_ABSTRACT_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("AbstractShot.getGradientWrtRolloutCache");
  }
#endif

  if (mRolloutCacheDirty)
  {
    mRolloutCache = std::make_shared<TrajectoryRolloutReal>(this);
    getStates(
        world,
        /* OUT */ mRolloutCache.get(),
        thisLog);
    mGradWrtRolloutCache = std::make_shared<TrajectoryRolloutReal>(this);
    mRolloutCacheDirty = false;
  }

#ifdef LOG_PERFORMANCE_ABSTRACT_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif

  return mGradWrtRolloutCache.get();
}

//==============================================================================
/// This computes finite difference Jacobians analagous to
/// backpropGradient()
void AbstractShot::finiteDifferenceGradient(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> grad)
{
  double originalLoss = mLoss.getLoss(getRolloutCache(world, nullptr), nullptr);

  int dims = getFlatProblemDim(world);
  Eigen::VectorXd flat = Eigen::VectorXd::Zero(dims);
  flatten(world, flat, nullptr);

  assert(grad.size() == dims);

  const double EPS = 1e-6;

  for (int i = 0; i < dims; i++)
  {
    flat(i) += EPS;
    unflatten(world, flat, nullptr);
    double posLoss = mLoss.getLoss(getRolloutCache(world, nullptr), nullptr);
    flat(i) -= EPS;

    flat(i) -= EPS;
    unflatten(world, flat, nullptr);
    double negLoss = mLoss.getLoss(getRolloutCache(world, nullptr), nullptr);
    flat(i) += EPS;

    grad(i) = (posLoss - negLoss) / (2 * EPS);
  }
}

//==============================================================================
int AbstractShot::getNumSteps()
{
  return mSteps;
}

//==============================================================================
/// This gets the total number of non-zero entries in the Jacobian
int AbstractShot::getNumberNonZeroJacobian(
    std::shared_ptr<simulation::World> world)
{
  return getNumberNonZeroJacobianStatic(world)
         + getNumberNonZeroJacobianDynamic(world);
}

//==============================================================================
/// This computes finite difference Jacobians analagous to backpropJacobians()
void AbstractShot::finiteDifferenceJacobian(
    std::shared_ptr<simulation::World> world, Eigen::Ref<Eigen::MatrixXd> jac)
{
  int dim = getFlatProblemDim(world);
  int numConstraints = getConstraintDim();
  assert(jac.cols() == dim);
  assert(jac.rows() == numConstraints);

  Eigen::VectorXd originalConstraints = Eigen::VectorXd::Zero(numConstraints);
  computeConstraints(world, originalConstraints, nullptr);
  Eigen::VectorXd flat = Eigen::VectorXd::Zero(dim);
  flatten(world, flat, nullptr);

  const double EPS = 1e-7;

  Eigen::VectorXd positiveConstraints = Eigen::VectorXd::Zero(numConstraints);
  Eigen::VectorXd negativeConstraints = Eigen::VectorXd::Zero(numConstraints);
  for (int i = 0; i < dim; i++)
  {
    flat(i) += EPS;
    unflatten(world, flat, nullptr);
    computeConstraints(world, positiveConstraints, nullptr);
    flat(i) -= EPS;

    flat(i) -= EPS;
    unflatten(world, flat, nullptr);
    computeConstraints(world, negativeConstraints, nullptr);
    flat(i) += EPS;

    jac.col(i) = (positiveConstraints - negativeConstraints) / (2 * EPS);
  }
}

} // namespace trajectory
} // namespace dart