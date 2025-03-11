# -*- coding: utf-8 -*-
"""
Created on Tue May 13 19:38:43 2014

@author: henrik
"""


class MarkerPose:
    def __init__(self, x, y, theta, quality, order = None, number = -1):
        self.x = x
        self.y = y
        self.theta = theta
        self.quality = quality
        self.order = order
        self.number = number

    def scale_position(self, scale_factor):
        self.x = self.x * scale_factor
        self.y = self.y * scale_factor

    def __repr__(self):
        return "MarkerPose(x=%f, y=%f, theta=%f, quality=%f, number=%f)" % (self.x, self.y, self.theta, self.quality, self.number)


