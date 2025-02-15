/*
 * Copyright (c) 2011-2019, The DART development contributors
 * All rights reserved.
 *
 * The list of contributors can be found at:
 *   https://github.com/dartsim/dart/blob/master/LICENSE
 *
 * This file is provided under the following "BSD-style" License:
 *   Redistribution and use in source and binary forms, with or
 *   without modification, are permitted provided that the following
 *   conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 *   CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 *   INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *   MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 *   USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 *   AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *   LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *   ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *   POSSIBILITY OF SUCH DAMAGE.
 */

#include "dart/dynamics/ArrowShape.hpp"

#include "dart/math/Constants.hpp"

namespace dart {
namespace dynamics {

//==============================================================================
ArrowShape::Properties::Properties(
    s_t _radius,
    s_t _headRadiusScale,
    s_t _headLengthScale,
    s_t _minHeadLength,
    s_t _maxHeadLength,
    bool _s_tArrow)
  : mRadius(_radius),
    mHeadRadiusScale(_headRadiusScale),
    mHeadLengthScale(_headLengthScale),
    mMinHeadLength(_minHeadLength),
    mMaxHeadLength(_maxHeadLength),
    ms_tArrow(_s_tArrow)
{
}

//==============================================================================
ArrowShape::ArrowShape(
    const Eigen::Vector3s& _tail,
    const Eigen::Vector3s& _head,
    const Properties& _properties,
    const Eigen::Vector4s& _color,
    std::size_t _resolution)
  : MeshShape(Eigen::Vector3s::Ones(), nullptr),
    mTail(_tail),
    mHead(_head),
    mProperties(_properties)
{
  instantiate(_resolution);
  configureArrow(mTail, mHead, mProperties);
  setColorMode(MeshShape::COLOR_INDEX);
  notifyColorUpdated(_color);
}

//==============================================================================
void ArrowShape::setPositions(
    const Eigen::Vector3s& _tail, const Eigen::Vector3s& _head)
{
  configureArrow(_tail, _head, mProperties);
}

//==============================================================================
const Eigen::Vector3s& ArrowShape::getTail() const
{
  return mTail;
}

//==============================================================================
const Eigen::Vector3s& ArrowShape::getHead() const
{
  return mHead;
}

//==============================================================================
void ArrowShape::setProperties(const Properties& _properties)
{
  configureArrow(mTail, mHead, _properties);
}

//==============================================================================
void ArrowShape::notifyColorUpdated(const Eigen::Vector4s& _color)
{
  for (std::size_t i = 0; i < mMesh->mNumMeshes; ++i)
  {
    aiMesh* mesh = mMesh->mMeshes[i];
    for (std::size_t j = 0; j < mesh->mNumVertices; ++j)
    {
      mesh->mColors[0][j] = aiColor4D(
          static_cast<float>(_color[0]),
          static_cast<float>(_color[1]),
          static_cast<float>(_color[2]),
          static_cast<float>(_color[3]));
    }
  }
}

//==============================================================================
const ArrowShape::Properties& ArrowShape::getProperties() const
{
  return mProperties;
}

//==============================================================================
static void constructArrowTip(
    aiMesh* mesh, s_t base, s_t tip, const ArrowShape::Properties& properties)
{
  std::size_t resolution = (mesh->mNumVertices - 1) / 2;
  for (std::size_t i = 0; i < resolution; ++i)
  {
    s_t theta = (s_t)(i) / (s_t)(resolution)*2 * math::constantsd::pi();

    s_t R = properties.mRadius;
    s_t x = R * cos(theta);
    s_t y = R * sin(theta);
    s_t z = base;
    mesh->mVertices[2 * i].Set(
        static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));

    if (base != tip)
    {
      x *= properties.mHeadRadiusScale;
      y *= properties.mHeadRadiusScale;
    }

    mesh->mVertices[2 * i + 1].Set(
        static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
  }

  mesh->mVertices[mesh->mNumVertices - 1].Set(0, 0, static_cast<float>(tip));
}

//==============================================================================
static void constructArrowBody(
    aiMesh* mesh, s_t z1, s_t z2, const ArrowShape::Properties& properties)
{
  std::size_t resolution = mesh->mNumVertices / 2;
  for (std::size_t i = 0; i < resolution; ++i)
  {
    s_t theta = (s_t)(i) / (s_t)(resolution)*2 * math::constantsd::pi();

    s_t R = properties.mRadius;
    s_t x = R * cos(theta);
    s_t y = R * sin(theta);
    s_t z = z1;
    mesh->mVertices[2 * i].Set(
        static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));

    z = z2;
    mesh->mVertices[2 * i + 1].Set(
        static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
  }
}

//==============================================================================
void ArrowShape::configureArrow(
    const Eigen::Vector3s& _tail,
    const Eigen::Vector3s& _head,
    const Properties& _properties)
{
  mTail = _tail;
  mHead = _head;
  mProperties = _properties;

  mProperties.mHeadLengthScale
      = std::max(0.0, std::min(1.0, mProperties.mHeadLengthScale));
  mProperties.mMinHeadLength = std::max(0.0, mProperties.mMinHeadLength);
  mProperties.mMaxHeadLength = std::max(0.0, mProperties.mMaxHeadLength);
  mProperties.mHeadRadiusScale = std::max(1.0, mProperties.mHeadRadiusScale);

  s_t length = (mTail - mHead).norm();

  s_t minHeadLength = std::min(
      mProperties.mMinHeadLength,
      mProperties.ms_tArrow ? length / 2.0 : length);
  s_t maxHeadLength = std::min(
      mProperties.mMaxHeadLength,
      mProperties.ms_tArrow ? length / 2.0 : length);

  s_t headLength = mProperties.mHeadLengthScale * length;
  headLength = std::min(maxHeadLength, std::max(minHeadLength, headLength));

  // construct the tail
  if (mProperties.ms_tArrow)
  {
    constructArrowTip(mMesh->mMeshes[0], headLength, 0, mProperties);
  }
  else
  {
    constructArrowTip(mMesh->mMeshes[0], 0, 0, mProperties);
  }

  // construct the main body
  if (mProperties.ms_tArrow)
  {
    constructArrowBody(
        mMesh->mMeshes[1], headLength, length - headLength, mProperties);
  }
  else
  {
    constructArrowBody(mMesh->mMeshes[1], 0, length - headLength, mProperties);
  }

  // construct the head
  constructArrowTip(
      mMesh->mMeshes[2], length - headLength, length, mProperties);

  Eigen::Isometry3s tf(Eigen::Isometry3s::Identity());
  tf.translation() = mTail;
  Eigen::Vector3s v = mHead - mTail;
  Eigen::Vector3s z = Eigen::Vector3s::UnitZ();

  if (v.norm() > 0)
  {
    v.normalize();
    Eigen::Vector3s axis = z.cross(v);
    if (axis.norm() > 0)
      axis.normalize();
    else
      axis
          = Eigen::Vector3s::UnitY(); // Any vector in the X/Y plane can be used
    tf.rotate(Eigen::AngleAxis_s(acos(z.dot(v)), axis));
  }

  aiNode* node = mMesh->mRootNode;
  for (std::size_t i = 0; i < 4; ++i)
    for (std::size_t j = 0; j < 4; ++j)
      node->mTransformation[i][j] = static_cast<float>(tf(i, j));

  mIsBoundingBoxDirty = true;
  mIsVolumeDirty = true;

  incrementVersion();
}

//==============================================================================
void ArrowShape::instantiate(std::size_t resolution)
{
  aiNode* node = new aiNode;
  node->mNumMeshes = 3;
  node->mMeshes = new unsigned int[3];
  for (std::size_t i = 0; i < 3; ++i)
    node->mMeshes[i] = i;

  aiScene* scene = new aiScene;
  scene->mNumMeshes = 3;
  scene->mMeshes = new aiMesh*[3];
  scene->mRootNode = node;

  scene->mMaterials = new aiMaterial*[1];
  scene->mMaterials[0] = new aiMaterial;

  // allocate memory
  for (std::size_t i = 0; i < 3; ++i)
  {
    std::size_t numVertices
        = (i == 0 || i == 2) ? 2 * resolution + 1 : 2 * resolution;

    aiMesh* mesh = new aiMesh;
    mesh->mMaterialIndex = (unsigned int)(-1);

    mesh->mNumVertices = numVertices;
    mesh->mVertices = new aiVector3D[numVertices];
    mesh->mNormals = new aiVector3D[numVertices];
    mesh->mColors[0] = new aiColor4D[numVertices];

    std::size_t numFaces = (i == 0 || i == 2) ? 3 * resolution : numVertices;
    mesh->mNumFaces = numFaces;
    mesh->mFaces = new aiFace[numFaces];
    for (std::size_t j = 0; j < numFaces; ++j)
    {
      mesh->mFaces[j].mNumIndices = 3;
      mesh->mFaces[j].mIndices = new unsigned int[3];
    }

    scene->mMeshes[i] = mesh;
  }

  // set normals
  aiMesh* mesh = scene->mMeshes[0];
  for (std::size_t i = 0; i < resolution; ++i)
  {
    mesh->mNormals[2 * i].Set(0.0f, 0.0f, 1.0f);

    s_t theta = (s_t)(i) / (s_t)(resolution)*2 * math::constantsd::pi();
    mesh->mNormals[2 * i + 1].Set(
        static_cast<float>(cos(theta)), static_cast<float>(sin(theta)), 0.0f);
  }
  mesh->mNormals[mesh->mNumVertices - 1].Set(0.0f, 0.0f, -1.0f);

  mesh = scene->mMeshes[1];
  for (std::size_t i = 0; i < resolution; ++i)
  {
    s_t theta = (s_t)(i) / (s_t)(resolution)*2 * math::constantsd::pi();
    mesh->mNormals[2 * i].Set(
        static_cast<float>(cos(theta)), static_cast<float>(sin(theta)), 0.0f);
    mesh->mNormals[2 * i + 1].Set(
        static_cast<float>(cos(theta)), static_cast<float>(sin(theta)), 0.0f);
  }

  mesh = scene->mMeshes[2];
  for (std::size_t i = 0; i < resolution; ++i)
  {
    mesh->mNormals[2 * i].Set(0.0f, 0.0f, -1.0f);

    s_t theta = (s_t)(i) / (s_t)(resolution)*2 * math::constantsd::pi();
    mesh->mNormals[2 * i + 1].Set(
        static_cast<float>(cos(theta)), static_cast<float>(sin(theta)), 0.0f);
  }
  mesh->mNormals[mesh->mNumVertices - 1].Set(0.0f, 0.0f, 1.0f);

  // set faces
  mesh = scene->mMeshes[0];
  aiFace* face;
  for (std::size_t i = 0; i < resolution; ++i)
  {
    // Back of head
    face = &mesh->mFaces[3 * i];
    face->mIndices[0] = 2 * i;
    face->mIndices[1] = 2 * i + 1;
    face->mIndices[2] = (i + 1 < resolution) ? 2 * i + 3 : 1;

    face = &mesh->mFaces[3 * i + 1];
    face->mIndices[0] = 2 * i;
    face->mIndices[1] = (i + 1 < resolution) ? 2 * i + 3 : 1;
    face->mIndices[2] = (i + 1 < resolution) ? 2 * i + 2 : 0;

    // Tip
    face = &mesh->mFaces[3 * i + 2];
    face->mIndices[0] = 2 * i + 1;
    face->mIndices[1] = 2 * resolution;
    face->mIndices[2] = (i + 1 < resolution) ? 2 * i + 3 : 1;
  }

  mesh = scene->mMeshes[1];
  for (std::size_t i = 0; i < resolution; ++i)
  {
    face = &mesh->mFaces[2 * i];
    face->mIndices[0] = 2 * i;
    face->mIndices[1] = (i + 1 < resolution) ? 2 * i + 3 : 1;
    face->mIndices[2] = 2 * i + 1;

    face = &mesh->mFaces[2 * i + 1];
    face->mIndices[0] = 2 * i;
    face->mIndices[1] = (i + 1 < resolution) ? 2 * i + 2 : 0;
    face->mIndices[2] = (i + 1 < resolution) ? 2 * i + 3 : 1;
  }

  mesh = scene->mMeshes[2];
  for (std::size_t i = 0; i < resolution; ++i)
  {
    // Back of head
    face = &mesh->mFaces[3 * i];
    face->mIndices[0] = 2 * i;
    face->mIndices[1] = (i + 1 < resolution) ? 2 * i + 3 : 1;
    face->mIndices[2] = 2 * i + 1;

    face = &mesh->mFaces[3 * i + 1];
    face->mIndices[0] = 2 * i;
    face->mIndices[1] = (i + 1 < resolution) ? 2 * i + 2 : 0;
    face->mIndices[2] = (i + 1 < resolution) ? 2 * i + 3 : 1;

    // Tip
    face = &mesh->mFaces[3 * i + 2];
    face->mIndices[0] = 2 * i + 1;
    face->mIndices[1] = (i + 1 < resolution) ? 2 * i + 3 : 1;
    face->mIndices[2] = 2 * resolution;
  }

  mMesh = scene;

  // setColor(mColor);
  // TODO(JS)
}

} // namespace dynamics
} // namespace dart
