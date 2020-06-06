#!/usr/bin/env python
from .robot_part import RobotPart
import pybullet as p


class Plane(RobotPart):
    def __init__(self, plane_normal=None, physics_client_id=0):
        super().__init__(physics_client_id)
        self._collision_id = p.createCollisionShape(shapeType=p.GEOM_PLANE,
                                                    planeNormal=[0, 0, 1] if not plane_normal else plane_normal,
                                                    physicsClientId=physics_client_id)
