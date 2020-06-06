#!/usr/bin/env python
from .robot_part import RobotPart
import pybullet as p


class Chassis(RobotPart):
    def __init__(self, x, y, z, rgba, specular_color=None, physics_client_id=0):
        super().__init__(physics_client_id)
        self._collision_id = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                                    halfExtents=[x / 2, y / 2, z / 2],
                                                    physicsClientId=physics_client_id)
        self._visual_id = p.createVisualShape(shapeType=p.GEOM_BOX,
                                              halfExtents=[x / 2, y / 2, z / 2],
                                              rgbaColor=rgba,
                                              specularColor=[0, 0, 0] if not specular_color else specular_color,
                                              physicsClientId=physics_client_id)
