#!/usr/bin/env python
import time
import pybullet as p
from parts.plane import Plane
from parts.wheel import Wheel
from parts.chassis import Chassis

DT = 1 / 240

if __name__ == '__main__':
    pc = p.connect(p.GUI)
    p.setGravity(gravX=0, gravY=0, gravZ=-9.81, physicsClientId=pc)

    # Plane
    plane_id = Plane(physics_client_id=pc)
    plane = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=plane_id.collision,
                              basePosition=[0, 0, 0], baseOrientation=[0, 0, 0, 1],
                              physicsClientId=pc)

    # Robot
    wheel = Wheel(radius=0.1, height=0.01, rgba=[0, 0.5, 0, 1], physics_client_id=pc)
    chassis = Chassis(x=0.02, y=0.2, z=0.4, rgba=[0, 0, 0.5, 1], physics_client_id=pc)

    wheel_orn = p.getQuaternionFromEuler([1.5708, 0, 0])
    robot = p.createMultiBody(baseMass=1,
                              baseCollisionShapeIndex=chassis.collision,
                              baseVisualShapeIndex=chassis.visual,
                              basePosition=[0.1, 0, 0.3], baseOrientation=[0, 0, 0, 1],
                              linkMasses=[0.25, 0.25],
                              linkCollisionShapeIndices=[wheel.collision, wheel.collision],
                              linkVisualShapeIndices=[wheel.visual, wheel.visual],
                              linkPositions=[[0, -0.115, -0.2], [0, 0.115, -0.2]],
                              linkOrientations=[wheel_orn, wheel_orn],
                              linkInertialFramePositions=[[0, 0, 0], [0, 0, 0]],
                              linkInertialFrameOrientations=[[0, 0, 0, 1], [0, 0, 0, 1]],
                              linkParentIndices=[0, 0],
                              linkJointTypes=[p.JOINT_REVOLUTE, p.JOINT_REVOLUTE],
                              linkJointAxis=[[0, 0, 1], [0, 0, 1]],
                              useMaximalCoordinates=False,
                              physicsClientId=pc)

    # nr_joints = p.getNumJoints(bodyUniqueId=robot, physicsClientId=pc)
    # states = p.getJointStates(bodyUniqueId=robot, jointIndices=list(range(nr_joints)), physicsClientId=pc)
    # print(states)

    while True:
        keys = p.getKeyboardEvents(physicsClientId=pc)
        if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
            break

        p.setJointMotorControlArray(bodyUniqueId=robot, jointIndices=[0, 1],
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocities=[10, 10], forces=[1, 1],
                                    physicsClientId=pc)

        p.stepSimulation(physicsClientId=pc)
        time.sleep(DT)

    p.disconnect(physicsClientId=pc)