# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 10:38:10 2025

@author: JDION
"""

class Node:
    
    def __init__(self, town, cost, parent, road_to_parent):
        self.town = town
        self.cost = cost
        self.parent = parent
        self.road_to_parent = road_to_parent
        

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