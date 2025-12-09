# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 10:38:10 2025

@author: JDION
"""

from functools import total_ordering

@total_ordering   
class Node:
    
    def __init__(self, town, cost, parent, road_to_parent):
        self.town = town
        self.cost = cost
        self.parent = parent
        self.road_to_parent = road_to_parent

    def __lt__(self, other):
        return self.cost < other.cost

    def __eq__(self, other):
        return self.cost == other.cost  
    
    
# Parcours à coût uniforme
def ucs(start_town, end_town):
    initial_node = Node(start_town, 0, None, None)
    frontier = PriorityQueue()
    frontier.put(initial_node)
    explored = list()
    while True:
        if frontier.empty():
            return None
        node = frontier.get()
        if node.town == end_town:
            return node
        explored.append(node.town)
        for neighbour_town, road_to_neighbour_town in node.town.neighbours.items():
            child_node = Node(neighbour_town, node.cost + road_to_neighbour_town.distance, node, road_to_neighbour_town)
            if child_node.town not in explored:
                not_in_frontier = True
                for frontier_node in frontier.queue:
                    if frontier_node.town == child_node.town:
                        not_in_frontier = False
                        break
                if not_in_frontier:
                    frontier.put(child_node)
                elif frontier_node.cost > child_node.cost: # on insère le noeud ayant un coût moins élevé dans la frontière (plutôt que de remplacer)
                    frontier.put(child_node)
    return None


# Parcours en profondeur itératif 
def dfs_iter(start_town, end_town):
    for depth in range(100):
        result = dls(start_town, end_town, depth)
        if result is not None:
            print(depth)
            return result

def dls(start_town, end_town, limit):
    initial_node = Node(start_town, 0, None, None)
    return recursive_dls(initial_node, end_town, limit)   

def recursive_dls(node, end_town, limit):
    if node.town == end_town:
        return node
    elif limit != 0:
        for neighbour_town, road_to_neighbour_town in node.town.neighbours.items():
            child_node = Node(neighbour_town, node.cost + road_to_neighbour_town.distance, node, road_to_neighbour_town)
            result = recursive_dls(child_node, end_town, limit - 1)
            if result is not None:
                return result
    return None


# Parcours en profondeur (risque de boucle !)
def dfs(start_town, end_town):
    initial_node = Node(start_town, 0, None, None)
    return recursive_dfs(initial_node, end_town)

def recursive_dfs(node, end_town):
    if node.town == end_town:
        return node
    for neighbour_town, road_to_neighbour_town in node.town.neighbours.items():
        child_node = Node(neighbour_town, node.cost + road_to_neighbour_town.distance, node, road_to_neighbour_town)
        result = recursive_dfs(child_node, end_town)
        if result is not None:
            return result 
    return None
        

# Parcours en largeur
def bfs(start_town, end_town):
    initial_node = Node(start_town, 0, None, None)
    if initial_node.town == end_town:
            return initial_node
    frontier = Queue() 
    frontier.put(initial_node)
    explored = list() 
    while True:
        if frontier.empty():
            return None
        node = frontier.get()
        explored.append(node.town)
        for neighbour_town, road_to_neighbour_town in node.town.neighbours.items():
            child_node = Node(neighbour_town, node.cost + road_to_neighbour_town.distance, node, road_to_neighbour_town)
            if child_node.town not in explored:
                not_in_frontier = True 
                for frontier_node in frontier.queue:
                    if frontier_node.town == child_node.town:
                        not_in_frontier = False
                        break
                if not_in_frontier:
                    if child_node.town == end_town:
                        return child_node
                    frontier.put(child_node)
    return None