#!/usr/bin/env python
# coding: utf-8

import numpy as np
import time
from sys import stderr
from functools import lru_cache
from queue import PriorityQueue
from types import MethodType

class gLEC(object):

    def __init__(
            self,
            mesh,
            max_fuel = 2000,
            horizontal_distance_cost_weight = 0.004,
            travel_cost_function = None,
            neighbouring_triangles_function = None,
            neighbouring_points_function = None,
            neighbours_cache_size = None,
            other_cache_size = None):

        self.mesh = mesh
        self.max_fuel = max_fuel
        self.horizontal_distance_cost_weight = horizontal_distance_cost_weight

        if self.horizontal_distance_cost_weight > 0.:
            self.normalised_area = self.max_fuel / self.horizontal_distance_cost_weight
        else:
            print("WARNING: The horizontal_distance_cost_weight is set to <= 0, and so can't be used to normalise the area",
                    file=stderr)
            self.normalised_area = float("nan")

        if travel_cost_function:
            # Allow the user to define their own cost function
            self.travel_cost_func = MethodType(travel_cost_function, self)
            # Note the MethodType is required to properly graft this function to this instance of the gLEC class
        else:
            # but if they don't, use the default
            self.travel_cost_func = self.strong_elevation_change_cost

        if neighbouring_triangles_function:
            # Allow the user to define their own function to find neighbouring triangles
            self.triangle_neighbours_func = MethodType(neighbouring_triangles_function, self)
        else:
            # but if they don't, use the default
            self.triangle_neighbours_func = self.get_adjoining_triangles

        if neighbouring_points_function:
            # Allow the user to define their own function to find neighbouring points
            self.point_neighbours_func = MethodType(neighbouring_points_function, self)
        else:
            # but if they don't, use the default
            self.point_neighbours_func = self.neighbouring_points_above_sealevel

        if not neighbours_cache_size:
            # If no cache size was supplied, make it as big as the mesh
            neighbours_cache_size = self.mesh.point_data['Z'].shape[0]

        if not other_cache_size:
            other_cache_size = neighbours_cache_size

        # Apply a LRU cache to all the hot functions
        self.triangle_neighbours_func = lru_cache(maxsize=neighbours_cache_size)(self.triangle_neighbours_func)
        self.point_neighbours_func = lru_cache(maxsize=neighbours_cache_size)(self.point_neighbours_func)
        self.travel_cost_func = lru_cache(maxsize=other_cache_size)(self.travel_cost_func)
        self.dist_func = lru_cache(maxsize=other_cache_size)(self.distance)


    def distance(self, current, _next):
        # from https://stackoverflow.com/a/1401828
        if current == _next:
            return 0.
        return np.linalg.norm(self.mesh.points[current]-self.mesh.points[_next])


    # This is the default travel cost function, which gets assigned in the constructor
    def strong_elevation_change_cost(self, current, _next):
        # Elevation changes contribute mostly to the calculated cost, with a much smaller
        # fraction being the horizontal distance travelled.
        if current == _next:
            return 0
        return int(
                # Elevation change
                abs(self.mesh.point_data['Z'][current] - self.mesh.point_data['Z'][_next]) \
                        # plus weighted horizontal distance 
                        + self.dist_func(current, _next) * self.horizontal_distance_cost_weight
               )


    def get_adjoining_triangles(self, current):
        # Get all the triangles that have the current point in them.
        return self.mesh.cells_dict['triangle'][np.where(self.mesh.cells_dict['triangle']==current)[0]]


    def neighbouring_points_above_sealevel(self, current):
        # Get all the other points from the cells that have the current point in them, and remove duplicates
        points = np.unique(self.triangle_neighbours_func(current))
        # remove the current point from these results:
        points = points[points != current]
        
        # Get the elevation of those connected points
        elevations = self.mesh.point_data['Z'][points]
        # Return a list of connected points, as long as they are above sea-level
        return points[elevations >= 0]
        
            
    def cost_search(self, start):
        # Some code and inspiration from http://theory.stanford.edu/~amitp/GameProgramming/ImplementationNotes.html

        frontier = PriorityQueue()  # The priority queue means that we can find the least cost path to continue
        frontier.put(start, 0)      # from, along any path, meaning the resulting paths should always be the least-cost
                                    # path to get to that point.
        
        # Setup the data structures. Keys are point indexes. 
        came_from = {}
        cost_so_far = {}
        dist_so_far = {}

        # Init the data structures
        came_from[start] = None
        cost_so_far[start] = 0
        dist_so_far[start] = 0
        
        while not frontier.empty():
            current = frontier.get()
            for _next in self.point_neighbours_func(current):
                # Calculate the cost of going to this new point.
                new_cost = cost_so_far[current] + self.travel_cost_func(current, _next)
                # Calculate the eulerian distance to this new point.
                new_dist = dist_so_far[current] + self.dist_func(current, _next)

                # The max_fuel check tells the algorithm to stop once a path has used up all its fuel.
                if (_next not in cost_so_far or new_cost < cost_so_far[_next]) and new_cost <= self.max_fuel:
                    cost_so_far[_next] = new_cost
                    dist_so_far[_next] = new_dist
                    priority = new_cost
                    frontier.put(_next, priority)
                    came_from[_next] = current
        return came_from, cost_so_far, dist_so_far


    def get_total_distance_for_all_paths_to_point(self, start):
        came_from, cost_so_far, dist_so_far = self.cost_search(start)
        
        # Find the edge nodes, and add up their costs to get the total
        total_dist = 0
        for k in came_from.keys():             # For all the points we've visited,
            if k not in came_from.values():    # Find all the points that haven't been 'came_from' (meaning they're the end of a path)
                total_dist += dist_so_far[k]
                
        return total_dist


    def get_area_covered_by_all_paths(self, start):
        came_from, cost_so_far, dist_so_far = self.cost_search(start)

        # Get a list of all the points we visted
        all_visted_points = came_from.keys()

        # We want a list of fully defined triangles - that is, where we visited all 3
        # vertexes of the triangle.
        neightris = []
        for p in all_visted_points:
            # For our current point, find all the triangles it is part of
            neightris.extend(self.triangle_neighbours_func(p))
        neightris = np.unique(np.array(neightris), axis=0)

        # For each triangle the point is in, see if the other vertexs were visited in all_visited_points
        good_tris = []
        for tri in neightris:
            if all(vertex in all_visted_points for vertex in tri):
                # If all points in the tri have been visited, then it's a 'good tri'
                good_tris.append(tri)
        good_tris = np.array(good_tris)

        def PolyArea(x,y):
            # From https://stackoverflow.com/a/30408825
            return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

        # For each good triangle, calculate the area, and add it to the total
        area = 0.
        for t in good_tris:
            points = self.mesh.points[t,:]
            area += PolyArea(points[:,0], points[:,1])

        return area


    def get_normalised_area_covered_by_all_paths(self, start, normalised_area = None):
        if not normalised_area:
            normalised_area = self.normalised_area
        return self.get_area_covered_by_all_paths(start) / normalised_area

