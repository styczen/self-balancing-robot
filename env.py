#!/usr/bin/env python
import numpy as np
import pybullet as p
from parts.wheel import Wheel
from parts.chassis import Chassis
from parts.plane import Plane


class SelfBalancingRobotEnv:
    def __init__(self, physics_client_id: int, measurement_noise=False):
        self._physics_client_id = physics_client_id
        self._forces = [1, 1]
        self._measurement_noise = measurement_noise
        self._robot_id = None

    def step(self, action: float) -> tuple:
        """Apply control and run one step of simulation."""
        p.setJointMotorControlArray(bodyUniqueId=self._robot_id, jointIndices=[0, 1],
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocities=[action, action], forces=self._forces,
                                    physicsClientId=self._physics_client_id)
        p.stepSimulation(physicsClientId=self._physics_client_id)
        observation = self._observation()
        done = SelfBalancingRobotEnv._done(observation)
        if done:
            reward = -100
        else:
            reward = 1
        return observation, reward, done

    def _observation(self):
        """Get robot's tilt angle."""
        robot_pos, robot_orn = p.getBasePositionAndOrientation(bodyUniqueId=self._robot_id,
                                                               physicsClientId=self._physics_client_id)
        robot_angle = p.getEulerFromQuaternion(robot_orn)[1]
        if not self._measurement_noise:
            robot_angle += 0.01 * np.random.randn()
        return robot_angle

    @staticmethod
    def _done(observation: float) -> bool:
        """When robot tilts to much, episode is done."""
        return True if abs(observation) > np.deg2rad(10) else False

    @property
    def forces(self) -> list:
        """Getter for forces."""
        return self._forces

    @forces.setter
    def forces(self, val: list):
        """Setter for forces on the wheel."""
        if len(val) != 2:
            raise ValueError('There should be two forces.')
        self._forces = val

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
