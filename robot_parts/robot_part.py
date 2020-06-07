#!/usr/bin/env python


class RobotPart:
    def __init__(self, physics_client_id=0):
        self._collision_id = None
        self._visual_id = None
        self._physics_client_id = physics_client_id

    @property
    def collision(self):
        return self._collision_id

    @property
    def visual(self):
        return self._visual_id

    @property
    def physics_client_id(self):
        return self._physics_client_id
