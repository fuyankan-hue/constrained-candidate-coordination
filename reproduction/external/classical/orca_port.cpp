/*
 * Agent.cc
 * RVO2 Library
 *
 * SPDX-FileCopyrightText: 2008 University of North Carolina at Chapel Hill
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please send all bug reports to <geom@cs.unc.edu>.
 *
 * The authors may be contacted via:
 *
 * Jur van den Berg, Stephen J. Guy, Jamie Snape, Ming C. Lin, Dinesh Manocha
 * Dept. of Computer Science
 * 201 S. Columbia St.
 * Frederick P. Brooks, Jr. Computer Science Bldg.
 * Chapel Hill, N.C. 27599-3175
 * United States of America
 *
 * <https://gamma.cs.unc.edu/RVO2/>
 */

// Extracted dynamic-agent RVO2 core; no static obstacles in this forecast adapter.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>
namespace RVO {
const float RVO_EPSILON=0.00001F;
static int degeneracies=0;
struct Vector2 {
 float a,b; Vector2(float x=0,float y=0):a(x),b(y){}
 float x() const{return a;} float y() const{return b;}
 Vector2 operator-() const{return Vector2(-a,-b);}
 Vector2& operator+=(const Vector2& v){a+=v.a;b+=v.b;return *this;}
};
Vector2 operator+(Vector2 x,Vector2 y){return Vector2(x.a+y.a,x.b+y.b);}
Vector2 operator-(Vector2 x,Vector2 y){return Vector2(x.a-y.a,x.b-y.b);}
Vector2 operator*(Vector2 x,float s){return Vector2(x.a*s,x.b*s);}
Vector2 operator*(float s,Vector2 x){return x*s;}
Vector2 operator/(Vector2 x,float s){return Vector2(x.a/s,x.b/s);}
float operator*(Vector2 x,Vector2 y){return x.a*y.a+x.b*y.b;}
float absSq(Vector2 x){return x*x;} float abs(Vector2 x){return std::sqrt(absSq(x));}
Vector2 normalize(Vector2 x){return x/abs(x);}
float det(Vector2 x,Vector2 y){return x.a*y.b-x.b*y.a;}
struct Line {Vector2 point,direction;};
Vector2 safeUnit(Vector2 w,float length,int i,int j){
 if(length>1e-12F)return w/length;
 ++degeneracies;return Vector2(i<j?1.F:-1.F,0.F);
}
bool linearProgram1(const std::vector<Line> &lines, std::size_t lineNo,
                    float radius, const Vector2 &optVelocity, bool directionOpt,
                    Vector2 &result) { /* NOLINT(runtime/references) */
  const float dotProduct = lines[lineNo].point * lines[lineNo].direction;
  const float discriminant =
      dotProduct * dotProduct + radius * radius - absSq(lines[lineNo].point);
  if (discriminant < 0.0F) {
    /* Max speed circle fully invalidates line lineNo. */
    return false;
  }

  const float sqrtDiscriminant = std::sqrt(discriminant);
  float tLeft = -dotProduct - sqrtDiscriminant;
  float tRight = -dotProduct + sqrtDiscriminant;

  for (std::size_t i = 0U; i < lineNo; ++i) {
    const float denominator = det(lines[lineNo].direction, lines[i].direction);
    const float numerator =
        det(lines[i].direction, lines[lineNo].point - lines[i].point);
    if (std::fabs(denominator) <= RVO_EPSILON) {
      /* Lines lineNo and i are (almost) parallel. */
      if (numerator < 0.0F) {
        return false;
      }

      continue;
    }

    const float t = numerator / denominator;

    if (denominator >= 0.0F) {
      /* Line i bounds line lineNo on the right. */
      tRight = std::min(tRight, t);
    } else {
      /* Line i bounds line lineNo on the left. */
      tLeft = std::max(tLeft, t);
    }
    if (tLeft > tRight) {
      return false;
    }
  }
  if (directionOpt) {
    /* Optimize direction. */
    if (optVelocity * lines[lineNo].direction > 0.0F) {
      /* Take right extreme. */
      result = lines[lineNo].point + tRight * lines[lineNo].direction;
    } else {
      /* Take left extreme. */
      result = lines[lineNo].point + tLeft * lines[lineNo].direction;
    }
  } else {
    /* Optimize closest point. */
    const float t =
        lines[lineNo].direction * (optVelocity - lines[lineNo].point);
    if (t < tLeft) {
      result = lines[lineNo].point + tLeft * lines[lineNo].direction;
    } else if (t > tRight) {
      result = lines[lineNo].point + tRight * lines[lineNo].direction;
    } else {
      result = lines[lineNo].point + t * lines[lineNo].direction;
    }
  }

  return true;
}
/**
 * @relates        Agent
 * @brief          Solves a two-dimensional linear program subject to linear
 *                 constraints defined by lines and a circular constraint.
 * @param[in]      lines        Lines defining the linear constraints.
 * @param[in]      radius       The radius of the circular constraint.
 * @param[in]      optVelocity  The optimization velocity.
 * @param[in]      directionOpt True if the direction should be optimized.
 * @param[in, out] result       A reference to the result of the linear program.
 * @return         The number of the line it fails on, and the number of lines
 *                 if successful.
 */
std::size_t linearProgram2(const std::vector<Line> &lines, float radius,
                           const Vector2 &optVelocity, bool directionOpt,
                           Vector2 &result) { /* NOLINT(runtime/references) */
  if (directionOpt) {
    /* Optimize direction. Note that the optimization velocity is of unit length
     * in this case.
     */
    result = optVelocity * radius;
  } else if (absSq(optVelocity) > radius * radius) {
    /* Optimize closest point and outside circle. */
    result = normalize(optVelocity) * radius;
  } else {
    /* Optimize closest point and inside circle. */
    result = optVelocity;
  }
  for (std::size_t i = 0U; i < lines.size(); ++i) {
    if (det(lines[i].direction, lines[i].point - result) > 0.0F) {
      /* Result does not satisfy constraint i. Compute new optimal result. */
      const Vector2 tempResult = result;

      if (!linearProgram1(lines, i, radius, optVelocity, directionOpt,
                          result)) {
        result = tempResult;

        return i;
      }
    }
  }

  return lines.size();
}
/**
 * @relates        Agent
 * @brief          Solves a two-dimensional linear program subject to linear
 *                 constraints defined by lines and a circular constraint.
 * @param[in]      lines        Lines defining the linear constraints.
 * @param[in]      numObstLines Count of obstacle lines.
 * @param[in]      beginLine    The line on which the 2-d linear program failed.
 * @param[in]      radius       The radius of the circular constraint.
 * @param[in, out] result       A reference to the result of the linear program.
 */
void linearProgram3(const std::vector<Line> &lines, std::size_t numObstLines,
                    std::size_t beginLine, float radius,
                    Vector2 &result) { /* NOLINT(runtime/references) */
  float distance = 0.0F;
  for (std::size_t i = beginLine; i < lines.size(); ++i) {
    if (det(lines[i].direction, lines[i].point - result) > distance) {
      /* Result does not satisfy constraint of line i. */
      std::vector<Line> projLines(
          lines.begin(),
          lines.begin() + static_cast<std::ptrdiff_t>(numObstLines));

      for (std::size_t j = numObstLines; j < i; ++j) {
        Line line;

        const float determinant = det(lines[i].direction, lines[j].direction);
        if (std::fabs(determinant) <= RVO_EPSILON) {
          /* Line i and line j are parallel. */
          if (lines[i].direction * lines[j].direction > 0.0F) {
            /* Line i and line j point in the same direction. */
            continue;
          }
          /* Line i and line j point in opposite direction. */
          line.point = 0.5F * (lines[i].point + lines[j].point);
        } else {
          line.point = lines[i].point + (det(lines[j].direction,
                                             lines[i].point - lines[j].point) /
                                         determinant) *
                                            lines[i].direction;
        }
        line.direction = normalize(lines[j].direction - lines[i].direction);
        projLines.push_back(line);
      }

      const Vector2 tempResult = result;
      if (linearProgram2(
              projLines, radius,
              Vector2(-lines[i].direction.y(), lines[i].direction.x()), true,
              result) < projLines.size()) {
        /* This should in principle not happen. The result is by definition
         * already in the feasible region of this linear program. If it fails,
         * it is due to small floating point error, and the current result is
         * kept. */
        result = tempResult;
      }
      distance = det(lines[i].direction, lines[i].point - result);
    }
  }
}

struct Agent {
 int id_;float radius_,timeHorizon_,maxSpeed_;
 Vector2 position_,velocity_,prefVelocity_,newVelocity_;
 std::vector<std::pair<float,const Agent*> > agentNeighbors_;
 std::vector<Line> orcaLines_;
 void computeNewVelocity(float timeStep){
  orcaLines_.clear();const std::size_t numObstLines=0;
  const float invTimeHorizon = 1.0F / timeHorizon_;

  /* Create agent ORCA lines. */
  for (std::size_t i = 0U; i < agentNeighbors_.size(); ++i) {
    const Agent *const other = agentNeighbors_[i].second;
    const Vector2 relativePosition = other->position_ - position_;
    const Vector2 relativeVelocity = velocity_ - other->velocity_;
    const float distSq = absSq(relativePosition);
    const float combinedRadius = radius_ + other->radius_;
    const float combinedRadiusSq = combinedRadius * combinedRadius;

    Line line;
    Vector2 u;
    if (distSq > combinedRadiusSq) {
      /* No collision. */
      const Vector2 w = relativeVelocity - invTimeHorizon * relativePosition;
      /* Vector from cutoff center to relative velocity. */
      const float wLengthSq = absSq(w);

      const float dotProduct = w * relativePosition;
      if (dotProduct < 0.0F &&
          dotProduct * dotProduct > combinedRadiusSq * wLengthSq) {
        /* Project on cut-off circle. */
        const float wLength = std::sqrt(wLengthSq);
        const Vector2 unitW = safeUnit(w, wLength, id_, other->id_);

        line.direction = Vector2(unitW.y(), -unitW.x());
        u = (combinedRadius * invTimeHorizon - wLength) * unitW;
      } else {
        /* Project on legs. */
        const float leg = std::sqrt(distSq - combinedRadiusSq);
        if (det(relativePosition, w) > 0.0F) {
          /* Project on left leg. */
          line.direction = Vector2(relativePosition.x() * leg -
                                       relativePosition.y() * combinedRadius,
                                   relativePosition.x() * combinedRadius +
                                       relativePosition.y() * leg) /
                           distSq;
        } else {
          /* Project on right leg. */
          line.direction = -Vector2(relativePosition.x() * leg +
                                        relativePosition.y() * combinedRadius,
                                    -relativePosition.x() * combinedRadius +
                                        relativePosition.y() * leg) /
                           distSq;
        }
        u = (relativeVelocity * line.direction) * line.direction -
            relativeVelocity;
      }
    } else {
      /* Collision. Project on cut-off circle of time timeStep. */
      const float invTimeStep = 1.0F / timeStep;

      /* Vector from cutoff center to relative velocity. */
      const Vector2 w = relativeVelocity - invTimeStep * relativePosition;

      const float wLength = abs(w);
      const Vector2 unitW = safeUnit(w, wLength, id_, other->id_);
      line.direction = Vector2(unitW.y(), -unitW.x());
      u = (combinedRadius * invTimeStep - wLength) * unitW;
    }

    line.point = velocity_ + 0.5F * u;
    orcaLines_.push_back(line);
  }

  const std::size_t lineFail =
      linearProgram2(orcaLines_, maxSpeed_, prefVelocity_, false, newVelocity_);

  if (lineFail < orcaLines_.size()) {
    linearProgram3(orcaLines_, numObstLines, lineFail, maxSpeed_, newVelocity_);
  }
}

};
}
extern "C" int orca_degeneracies(){return RVO::degeneracies;}
extern "C" int orca_lp(int n,const double* data,double radius,const double* preference,double* result){
 std::vector<RVO::Line> lines;
 for(int i=0;i<n;++i){RVO::Line l;l.point=RVO::Vector2(data[4*i],data[4*i+1]);l.direction=RVO::Vector2(data[4*i+2],data[4*i+3]);lines.push_back(l);}
 RVO::Vector2 r;int failed=RVO::linearProgram2(lines,radius,RVO::Vector2(preference[0],preference[1]),false,r);
 result[0]=r.x();result[1]=r.y();return failed;
}
extern "C" int orca_rollout(int n,int steps,const double* current,const double* velocity,const double* anchor,double horizon,double* output){
 RVO::degeneracies=0;std::vector<RVO::Agent> agents(n);
 for(int i=0;i<n;++i){
  auto& a=agents[i];a.id_=i;a.position_=RVO::Vector2(current[2*i],current[2*i+1]);a.velocity_=RVO::Vector2(velocity[2*i],velocity[2*i+1]);
  a.radius_=0.5F;a.timeHorizon_=horizon;
  double speed=std::hypot(velocity[2*i],velocity[2*i+1]);
  for(int t=0;t<steps;++t){int at=(i*steps+t)*2;double x=t?anchor[at-2]:current[2*i],y=t?anchor[at-1]:current[2*i+1];speed=std::max(speed,std::hypot(anchor[at]-x,anchor[at+1]-y)/0.1);}
  a.maxSpeed_=std::max(speed,0.1);
 }
 for(int t=0;t<steps;++t){
  for(int i=0;i<n;++i){
   auto& a=agents[i];a.agentNeighbors_.clear();
   for(int j=0;j<n;++j)if(i!=j){float d=RVO::absSq(agents[j].position_-a.position_);if(d<75.F*75.F)a.agentNeighbors_.push_back(std::make_pair(d,&agents[j]));}
   std::stable_sort(a.agentNeighbors_.begin(),a.agentNeighbors_.end(),[](const std::pair<float,const RVO::Agent*>& x,const std::pair<float,const RVO::Agent*>& y){return x.first<y.first;});
   int at=(i*steps+t)*2;a.prefVelocity_=RVO::Vector2((anchor[at]-a.position_.x())/0.1,(anchor[at+1]-a.position_.y())/0.1);
   a.computeNewVelocity(0.1F);
  }
  for(int i=0;i<n;++i){auto& a=agents[i];a.velocity_=a.newVelocity_;a.position_+=a.velocity_*0.1F;int at=(i*steps+t)*2;output[at]=a.position_.x();output[at+1]=a.position_.y();if(!std::isfinite(output[at])||!std::isfinite(output[at+1]))return 1;}
 }
 return 0;
}
