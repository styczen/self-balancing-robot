#!/usr/bin/env python
import numpy as np
import pybullet as p
from robot_parts.wheel import Wheel
from robot_parts.chassis import Chassis
from robot_parts.plane import Plane


class SelfBalancingRobotEnv:
    OBSERVATION_SIZE = 3
    ACTION_SIZE = 1

    def __init__(self, physics_client_id: int,
                 # action_scale=10
                 ):
        self._physics_client_id = physics_client_id
        # self._forces = [1, 1]
        self._robot_id = None
        # self._action_scale = action_scale
        self._vd = 0  # desired velocity
        self._vt = 0  # current velocity
        self._maxV = 24.6  # 235RPM = 24,609142453 rad/sec maximum velocity of robot

    def step(self, action) -> tuple:
        """Apply control and run one step of simulation."""
        self._apply_action(action)
        p.stepSimulation(physicsClientId=self._physics_client_id)
        observation = self._observation()
        done = self._done()
        reward = self._reward()
        return observation, reward, done, ''

    def reset(self):
        """Resets simulation and spawn every object in its initial position."""
        # Reset all simulation state
        p.resetSimulation(physicsClientId=self._physics_client_id)

        # After reset every parameter has to be set
        p.setGravity(gravX=0, gravY=0, gravZ=-9.81, physicsClientId=self._physics_client_id)

        # Plane
        plane_id = Plane(physics_client_id=self._physics_client_id)
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=plane_id.collision,
                          basePosition=[0, 0, 0], baseOrientation=[0, 0, 0, 1],
                          physicsClientId=self._physics_client_id)

        # Robot
        wheel_left = Wheel(radius=0.1, height=0.03, rgba=[0, 0.5, 0, 1], physics_client_id=self._physics_client_id)
        wheel_right = Wheel(radius=0.1, height=0.03, rgba=[0.5, 0, 0, 1], physics_client_id=self._physics_client_id)

        chassis = Chassis(x=0.05, y=0.2, z=0.4, rgba=[0, 0, 0.5, 1], physics_client_id=self._physics_client_id)

        robot_orn = p.getQuaternionFromEuler(np.deg2rad([0, SelfBalancingRobotEnv._starting_angle(1), 0]))
        robot_pos = [0, 0, 0.3]  # on the plane [0, 0, 0.3]
        self._robot_id = p.createMultiBody(baseMass=1,
                                           baseCollisionShapeIndex=chassis.collision,
                                           baseVisualShapeIndex=chassis.visual,
                                           basePosition=robot_pos, baseOrientation=robot_orn,
                                           linkMasses=[0.25, 0.25],
                                           linkCollisionShapeIndices=[wheel_left.collision, wheel_right.collision],
                                           linkVisualShapeIndices=[wheel_left.visual, wheel_right.visual],
                                           linkPositions=[[0, -0.12, -0.2], [0, 0.12, -0.2]],
                                           linkOrientations=[[0, 0, 0, 1], [0, 0, 0, 1]],
                                           linkInertialFramePositions=[[0, 0, 0], [0, 0, 0]],
                                           linkInertialFrameOrientations=[[0, 0, 0, 1], [0, 0, 0, 1]],
                                           linkParentIndices=[0, 0],
                                           linkJointTypes=[p.JOINT_REVOLUTE, p.JOINT_REVOLUTE],
                                           linkJointAxis=[[0, 1, 0], [0, 1, 0]],
                                           useMaximalCoordinates=False,
                                           physicsClientId=self._physics_client_id)

        return self._observation()

    def _observation(self):
        """Get robot's tilt angle."""
        robot_pos, robot_orn = p.getBasePositionAndOrientation(bodyUniqueId=self._robot_id,
                                                               physicsClientId=self._physics_client_id)
        robot_euler = p.getEulerFromQuaternion(robot_orn)
        linear_vel, angular_vel = p.getBaseVelocity(bodyUniqueId=self._robot_id,
                                                    physicsClientId=self._physics_client_id)
        return np.array([robot_euler[1], angular_vel[1], self._vt])

    def _reward(self):
        return 0.1 - abs(self._vt - self._vd) * 0.005

    def _apply_action(self, action):
        vt = self._vt + action  # dt = 1 so action is essentially acceleration so control with torque
        self._vt = np.clip(a=vt, a_min=-self._maxV, a_max=self._maxV)[0]
        p.setJointMotorControlArray(bodyUniqueId=self._robot_id, jointIndices=[0, 1],
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocities=[self._vt, self._vt],
                                    # forces=self._forces,
                                    physicsClientId=self._physics_client_id)

    def _done(self) -> bool:
        """When robot tilts to much, episode is done."""
        robot_pos, robot_orn = p.getBasePositionAndOrientation(bodyUniqueId=self._robot_id,
                                                               physicsClientId=self._physics_client_id)
        robot_euler = p.getEulerFromQuaternion(robot_orn)
        return abs(robot_euler[1]) > np.deg2rad(45)

    @staticmethod
    def _starting_angle(angle=1):
        """Randomly generates angle in range [-angle, angle) in degrees."""
        return np.random.rand() * 2 * angle - angle

    def wheels_state(self):
        """Get state of wheels i.e. angular velocities and applied torques."""
        state = p.getJointStates(bodyUniqueId=self._robot_id, jointIndices=[0, 1],
                                 physicsClientId=self._physics_client_id)
        velocities = (state[0][1], state[1][1])
        applied_torques = (state[0][3], state[1][3])
        return velocities, applied_torques

    # @property
    # def forces(self) -> list:
    #     """Getter for forces."""
    #     return self._forces
    #
    # @forces.setter
    # def forces(self, val: list):
    #     """Setter for forces on the wheel."""
    #     if len(val) != 2:
    #         raise ValueError('There should be two forces.')
    #     self._forces = val

    # @property
    # def action_scale(self) -> float:
    #     """Action scale coefficient getter."""
    #     return self._action_scale
    #
    # @action_scale.setter
    # def action_scale(self, val: float):
    #     """Action scale coefficient setter."""
    #     self._action_scale = val

    # @property
    # def noise_coeff(self) -> float:
    #     """Measurement noise coefficient getter."""
    #     return self._noise_coeff
    #
    # @noise_coeff.setter
    # def noise_coeff(self, val: float):
    #     """Measurement noise coefficient setter."""
    #     self._noise_coeff = val
