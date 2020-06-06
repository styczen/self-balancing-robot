#!/usr/bin/env python
from .robot_part import RobotPart
import pybullet as p
import numpy as np


class Wheel(RobotPart):
    def __init__(self, radius, height, rgba, specular_color=None, physics_client_id=0):
        super().__init__(physics_client_id)
        self._collision_id = p.createCollisionShape(
            shapeType=p.GEOM_CYLINDER, radius=radius, height=height,
            collisionFrameOrientation=p.getQuaternionFromEuler(np.deg2rad([90, 0, 0])),
            physicsClientId=physics_client_id)
        self._visual_id = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER, radius=radius, length=height,
            rgbaColor=rgba, specularColor=[0, 0, 0] if not specular_color else specular_color,
            visualFrameOrientation=p.getQuaternionFromEuler(np.deg2rad([90, 0, 0])),
            physicsClientId=physics_client_id)
