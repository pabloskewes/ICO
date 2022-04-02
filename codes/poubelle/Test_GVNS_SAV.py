#!/usr/bin/env python
# coding: utf-8

# # 0. Data Processing

import os
import numpy as np
import copy
import time
import matplotlib.pyplot as plt
# import pandas as pd


import random
from random import *

dir_name = os.path.dirname(os.path.realpath('__file__'))


# Basic Function

def vrp_graph(tour, x, y):
    plt.figure(figsize=(16, 16), dpi=160)
    plt.plot(x[0], y[0], 'r^')
    plt.scatter(x[1:], y[1:], s=5, c='k', marker=',')
    for i in range(1, cust_size + 1):
        plt.annotate(i, (x[i] + 0.2, y[i] + 0.2), size=8)
    for i in range(len(tour)):
        plt.plot(x[tour[i]], y[tour[i]])

    plt.show()

def total_distance(tours, distance):
    total_distance = 0
    for tour in tours:
        tour_distance = 0
        for i in range(len(tour) - 1):
            tour_distance += distance[tour[i]][tour[i + 1]]
        total_distance += tour_distance
    return total_distance


def time_checker(tour, travel_time, service_time, ready_time, due_time):
    time = 0
    counter = 0
    for i in range(1, len(tour)):
        time = max(time, ready_time[tour[i - 1]]) + service_time[tour[i - 1]] + travel_time[tour[i - 1]][tour[i]]
        if time <= due_time[tour[i]]:
            counter += 1
        else:
            break
    if counter == len(tour) - 1:
        return True
    else:
        return False


def begin_time(tour, travel_time, service_time, ready_time):
    begin_service_time = [0]
    time = 0
    for i in range(1, len(tour)):
        time = max(time, ready_time[tour[i - 1]]) + service_time[tour[i - 1]] + travel_time[tour[i - 1]][tour[i]]
        begin = max(time, ready_time[tour[i]])
        begin_service_time.append(begin)

    return begin_service_time


def total_time(tours, travel_time, service_time, ready_time):
    total_time = 0
    for tour in tours:
        tour_time = begin_time(tour, travel_time, service_time, ready_time)[-1]
        total_time += tour_time
    return total_time

def check_sequence(sequence, tour):
    i = tour.index(sequence[0])
    if tour[i+1] == sequence[1]:
        return True
    else:
        return False


# # 2. Route Improvement

# 2.1. 2-opt

def two_opt_move(tour, distance, travel_time, service_time, ready_time, due_time):
    best_imp = 0
    node1 = -1
    node2 = -1

    if len(tour) >= 5:
        for i in range(len(tour) - 3):
            for j in range(i + 2, len(tour) - 1):
                new_tour = tour[0:i + 1] + tour[i + 1:j + 1][::-1] + tour[j + 1:]
                if time_checker(new_tour, travel_time, service_time, ready_time, due_time):
                    imp = distance[tour[i]][tour[j]] + distance[tour[i + 1]][tour[j + 1]] - distance[tour[i]][
                        tour[i + 1]] - distance[tour[j]][tour[j + 1]]
                    if imp < best_imp:
                        node1 = i
                        node2 = j
                        best_imp = imp

    return node1, node2, best_imp


def two_opt_search(sub_tour, distance, travel_time, service_time, ready_time, due_time):
    Best_Imp = 0
    Tour = []
    Position1 = []
    Position2 = []

    for i in range(len(sub_tour)):
        [Node1, Node2, Imp] = two_opt_move(sub_tour[i], distance, travel_time, service_time, ready_time, due_time)

        if Node1 != -1:
            Best_Imp += Imp
            Tour.append(i)
            Position1.append(Node1)
            Position2.append(Node2)
    return Tour, Position1, Position2, Best_Imp


# 2.2. Or-opt Exchange

def or_opt_move(tour, distance, travel_time, service_time, ready_time, due_time, K):
    best_imp = 0
    node1 = -1
    node2 = -1
    node3 = -1

    if len(tour) >= K + 3:
        for i in range(len(tour) - K - 1):
            j = i + K
            for k in range(len(tour) - 1):
                if (k < i) or (j < k):
                    if k < i:
                        new_tour = tour[0:k + 1] + tour[i + 1:j + 1] + tour[k + 1:i + 1] + tour[j + 1:]
                    else:
                        new_tour = tour[0:i + 1] + tour[j + 1:k + 1] + tour[i + 1:j + 1] + tour[k + 1:]

                    if time_checker(new_tour, travel_time, service_time, ready_time, due_time):
                        Del_Cost = distance[tour[i]][tour[i + 1]] + distance[tour[j]][tour[j + 1]] + distance[tour[k]][
                            tour[k + 1]]
                        imp = distance[tour[i]][tour[j + 1]] + distance[tour[k]][tour[i + 1]] + distance[tour[j]][
                            tour[k + 1]] - Del_Cost

                        if imp < best_imp:
                            node1 = i
                            node2 = j
                            node3 = k
                            best_imp = imp

    return node1, node2, node3, best_imp


def or_opt_search(sub_tour, distance, travel_time, service_time, ready_time, due_time, K):
    Best_Imp = 0
    Tour = []
    Position1 = []
    Position2 = []
    Position3 = []

    for i in range(len(sub_tour)):
        [Node1, Node2, Node3, Imp] = or_opt_move(sub_tour[i], distance, travel_time, service_time, ready_time, due_time,
                                                 K)

        if Node1 != -1:
            Best_Imp += Imp
            Tour.append(i)
            Position1.append(Node1)
            Position2.append(Node2)
            Position3.append(Node3)
    return Tour, Position1, Position2, Position3, Best_Imp


# 3. Inter Relocation

def relocate(tour1, tour2, distance, travel_time, service_time, ready_time, due_time):
    best_imp = 0
    cust = -1
    position = -1

    for i in range(1, len(tour1) - 1):
        if demand[tour1[i]] + sum(demand[tour2]) <= capacity:

            for j in range(len(tour2) - 1):
                new_tour2 = tour2[:j + 1] + [tour1[i]] + tour2[j + 1:]

                time_check_relocate = time_checker(new_tour2, travel_time, service_time, ready_time, due_time)
                if time_check_relocate:
                    tour1_imp = distance[tour1[i - 1]][tour1[i]] + distance[tour1[i]][tour1[i + 1]] - \
                                distance[tour1[i - 1]][tour1[i + 1]]
                    tour2_inc = distance[tour2[j]][tour1[i]] + distance[tour1[i]][tour2[j + 1]] - distance[tour2[j]][
                        tour2[j + 1]]

                    if (tour2_inc - tour1_imp) < best_imp:
                        best_imp = tour2_inc - tour1_imp
                        cust = i
                        position = j

    return cust, position, best_imp


def relocate_search(sub_tour, distance, travel_time, service_time, ready_time, due_time):
    Best_Imp = 0
    T1 = -1
    T2 = -1
    Cust = -1
    Position = -1

    for t1 in range(len(sub_tour)):
        for t2 in range(len(sub_tour)):
            if t1 != t2:
                [cust, position, imp] = relocate(sub_tour[t1], sub_tour[t2], distance, travel_time, service_time,
                                                 ready_time, due_time)

                if imp < Best_Imp:
                    # print(imp)
                    T1 = t1
                    T2 = t2
                    Cust = cust
                    Position = position
                    Best_Imp = imp

    return T1, T2, Cust, Position, Best_Imp


# 4. Inter Exchange

def exchange(tour1, tour2, distance, travel_time, service_time, ready_time, due_time):
    best_imp = 0
    position1 = -1
    position2 = -1

    for i in range(1, len(tour1) - 1):
        for j in range(1, len(tour2) - 1):
            tour1_new_demand = demand[tour2[j]] + sum(demand[tour1]) - demand[tour1[i]]
            tour2_new_demand = demand[tour1[i]] + sum(demand[tour2]) - demand[tour2[j]]

            if (tour1_new_demand <= capacity) and (tour2_new_demand <= capacity):

                new_tour1 = tour1[:i] + [tour2[j]] + tour1[i + 1:]
                new_tour2 = tour2[:j] + [tour1[i]] + tour2[j + 1:]

                time_check_exchange1 = time_checker(new_tour1, travel_time, service_time, ready_time, due_time)
                time_check_exchange2 = time_checker(new_tour2, travel_time, service_time, ready_time, due_time)

                if time_check_exchange1 and time_check_exchange2:
                    ex_cost1 = distance[tour1[i - 1]][tour2[j]] + distance[tour2[j]][tour1[i + 1]] - \
                               distance[tour1[i - 1]][tour1[i]] - distance[tour1[i]][tour1[i + 1]]
                    ex_cost2 = distance[tour2[j - 1]][tour1[i]] + distance[tour1[i]][tour2[j + 1]] - \
                               distance[tour2[j - 1]][tour2[j]] - distance[tour2[j]][tour2[j + 1]]

                    if (ex_cost1 + ex_cost2) < best_imp:
                        best_imp = ex_cost1 + ex_cost2
                        position1 = i
                        position2 = j

    return position1, position2, best_imp


def exchange_search(sub_tour, distance, travel_time, service_time, ready_time, due_time):
    Best_Imp = 0
    T1 = -1
    T2 = -1
    Position1 = -1
    Position2 = -1

    for t1 in range(len(sub_tour) - 1):
        for t2 in range(t1 + 1, len(sub_tour)):
            [position1, position2, imp] = exchange(sub_tour[t1], sub_tour[t2], distance, travel_time, service_time,
                                                   ready_time, due_time)

            if imp < Best_Imp:
                T1 = t1
                T2 = t2
                Position1 = position1
                Position2 = position2
                Best_Imp = imp

    return T1, T2, Position1, Position2, Best_Imp


# 5. CROSS

def CROSS(tour1, tour2, distance, travel_time, service_time, ready_time, due_time):
    best_imp = 0
    node11 = -1
    node12 = -1
    node21 = -1
    node22 = -1

    for i in range(1, len(tour1) - 2):
        for k in range(i + 1, len(tour1) - 1):
            for j in range(1, len(tour2) - 2):
                for l in range(j + 1, len(tour2) - 1):
                    tour1_new_demand = sum(demand[tour2[j:l + 1]]) + sum(demand[tour1]) - sum(demand[tour1[i:k + 1]])
                    tour2_new_demand = sum(demand[tour1[i:k + 1]]) + sum(demand[tour2]) - sum(demand[tour2[j:l + 1]])

                    if (tour1_new_demand <= capacity) and (tour2_new_demand <= capacity):

                        new_tour1 = tour1[:i] + tour2[j:l + 1] + tour1[k + 1:]
                        new_tour2 = tour2[:j] + tour1[i:k + 1] + tour2[l + 1:]

                        time_check_CROSS1 = time_checker(new_tour1, travel_time, service_time, ready_time, due_time)
                        time_check_CROSS2 = time_checker(new_tour2, travel_time, service_time, ready_time, due_time)

                        if time_check_CROSS1 and time_check_CROSS2:
                            CROSS1 = round(distance[tour1[i - 1]][tour2[j]] + distance[tour2[l]][tour1[k + 1]] +
                                           distance[tour2[j - 1]][tour1[i]] + distance[tour1[k]][tour2[l + 1]], 10)
                            CROSS2 = round(distance[tour1[i - 1]][tour1[i]] + distance[tour1[k]][tour1[k + 1]] +
                                           distance[tour2[j - 1]][tour2[j]] + distance[tour2[l]][tour2[l + 1]], 10)
                            CROSS_cost = CROSS1 - CROSS2

                            if CROSS_cost < best_imp:
                                best_imp = CROSS_cost
                                node11 = i
                                node12 = k
                                node21 = j
                                node22 = l

    return node11, node12, node21, node22, best_imp


def CROSS_search(sub_tour, distance, travel_time, service_time, ready_time, due_time):
    Best_Imp = 0
    T1 = -1
    T2 = -1
    Node11 = -1
    Node12 = -1
    Node21 = -1
    Node22 = -1

    for t1 in range(len(sub_tour) - 1):
        for t2 in range(t1 + 1, len(sub_tour)):
            [node11, node12, node21, node22, imp] = CROSS(sub_tour[t1], sub_tour[t2], distance, travel_time,
                                                          service_time, ready_time, due_time)

            if imp < Best_Imp:
                T1 = t1
                T2 = t2
                Node11 = node11
                Node12 = node12
                Node21 = node21
                Node22 = node22
                Best_Imp = imp

    return T1, T2, Node11, Node12, Node21, Node22, Best_Imp


# 6. ICROSS

def ICROSS(tour1, tour2, distance, travel_time, service_time, ready_time, due_time):
    best_imp = 0
    node11 = -1
    node12 = -1
    node21 = -1
    node22 = -1

    for i in range(1, len(tour1) - 2):
        for k in range(i + 1, len(tour1) - 1):
            for j in range(1, len(tour2) - 2):
                for l in range(j + 1, len(tour2) - 1):
                    tour1_new_demand = sum(demand[tour2[j:l + 1]]) + sum(demand[tour1]) - sum(demand[tour1[i:k + 1]])
                    tour2_new_demand = sum(demand[tour1[i:k + 1]]) + sum(demand[tour2]) - sum(demand[tour2[j:l + 1]])

                    if (tour1_new_demand <= capacity) and (tour2_new_demand <= capacity):

                        new_tour1 = tour1[:i] + tour2[j:l + 1][::-1] + tour1[k + 1:]
                        new_tour2 = tour2[:j] + tour1[i:k + 1][::-1] + tour2[l + 1:]

                        time_check_ICROSS1 = time_checker(new_tour1, travel_time, service_time, ready_time, due_time)
                        time_check_ICROSS2 = time_checker(new_tour2, travel_time, service_time, ready_time, due_time)

                        if time_check_ICROSS1 and time_check_ICROSS2:
                            ICROSS1 = round(distance[tour1[i - 1]][tour2[l]] + distance[tour2[j]][tour1[k + 1]] +
                                            distance[tour2[j - 1]][tour1[k]] + distance[tour1[i]][tour2[l + 1]], 10)
                            ICROSS2 = round(distance[tour1[i - 1]][tour1[i]] + distance[tour1[k]][tour1[k + 1]] +
                                            distance[tour2[j - 1]][tour2[j]] + distance[tour2[l]][tour2[l + 1]], 10)
                            ICROSS3 = round(
                                total_distance([tour1[i:k + 1][::-1]], distance) - total_distance([tour1[i:k + 1]],
                                                                                                  distance), 10)
                            ICROSS4 = round(
                                total_distance([tour2[j:l + 1][::-1]], distance) - total_distance([tour2[j:l + 1]],
                                                                                                  distance), 10)
                            ICROSS_cost = ICROSS1 - ICROSS2 + ICROSS3 + ICROSS4

                            if ICROSS_cost < best_imp:
                                best_imp = ICROSS_cost
                                node11 = i
                                node12 = k
                                node21 = j
                                node22 = l

    return node11, node12, node21, node22, best_imp


def ICROSS_search(sub_tour, distance, travel_time, service_time, ready_time, due_time):
    Best_Imp = 0
    T1 = -1
    T2 = -1
    Node11 = -1
    Node12 = -1
    Node21 = -1
    Node22 = -1

    for t1 in range(len(sub_tour) - 1):
        for t2 in range(t1 + 1, len(sub_tour)):
            [node11, node12, node21, node22, imp] = ICROSS(sub_tour[t1], sub_tour[t2], distance, travel_time,
                                                           service_time, ready_time, due_time)

            if imp < Best_Imp:
                T1 = t1
                T2 = t2
                Node11 = node11
                Node12 = node12
                Node21 = node21
                Node22 = node22
                Best_Imp = imp

    return T1, T2, Node11, Node12, Node21, Node22, Best_Imp


# 7. GENI

def GENI(tour1, tour2, distance, travel_time, service_time, ready_time, due_time):
    best_imp = 0
    node1 = -1
    node21 = -1
    node22 = -1

    if len(tour2) >= 4:
        for i in range(1, len(tour1) - 1):
            for j in range(0, len(tour2) - 3):
                for k in range(j + 2, len(tour2) - 1):
                    tour2_new_demand = demand[tour1[i]] + sum(demand[tour2])

                    if tour2_new_demand <= capacity:
                        new_tour1 = tour1[:i] + tour1[i + 1:]
                        new_tour2 = tour2[:j + 1] + [tour1[i]] + [tour2[k]] + tour2[j + 1:k] + tour2[k + 1:]

                        time_check_GENI1 = time_checker(new_tour1, travel_time, service_time, ready_time, due_time)
                        time_check_GENI2 = time_checker(new_tour2, travel_time, service_time, ready_time, due_time)

                        if time_check_GENI1 and time_check_GENI2:
                            GENI1 = round(distance[tour1[i - 1]][tour1[i + 1]] - distance[tour1[i - 1]][tour1[i]] -
                                          distance[tour1[i]][tour1[i + 1]], 10)
                            GENI2 = round(
                                distance[tour2[j]][tour1[i]] + distance[tour1[i]][tour2[k]] + distance[tour2[k]][
                                    tour2[j + 1]] + distance[tour2[k - 1]][tour2[k + 1]] - distance[tour2[j]][
                                    tour2[j + 1]] - distance[tour2[k - 1]][tour2[k]] - distance[tour2[k]][tour2[k + 1]],
                                10)
                            GENI_cost = GENI1 + GENI2

                            if GENI_cost < best_imp:
                                best_imp = GENI_cost
                                node1 = i
                                node21 = j
                                node22 = k

    return node1, node21, node22, best_imp


def GENI_search(sub_tour, distance, travel_time, service_time, ready_time, due_time):
    Best_Imp = 0
    T1 = -1
    T2 = -1
    Node1 = -1
    Node21 = -1
    Node22 = -1

    for t1 in range(len(sub_tour)):
        for t2 in range(len(sub_tour)):
            if t1 != t2:
                [node11, node21, node22, imp] = GENI(sub_tour[t1], sub_tour[t2], distance, travel_time, service_time,
                                                     ready_time, due_time)

                if imp < Best_Imp:
                    T1 = t1
                    T2 = t2
                    Node1 = node11
                    Node21 = node21
                    Node22 = node22
                    Best_Imp = imp

    return T1, T2, Node1, Node21, Node22, Best_Imp


# 8. Ejection Chain


# 9. 2-opt*

def two_optstar(tour1, tour2, distance, travel_time, service_time, ready_time, due_time):
    best_imp = 0
    position1 = -1
    position2 = -1

    for i in range(0, len(tour1) - 1):
        for j in range(0, len(tour2) - 1):
            new_tour1 = tour1[:i + 1] + tour2[j + 1:]
            new_tour2 = tour2[:j + 1] + tour1[i + 1:]
            tour1_new_demand = sum(demand[new_tour1])
            tour2_new_demand = sum(demand[new_tour2])

            if (tour1_new_demand <= capacity) and (tour2_new_demand <= capacity):

                time_check_2opts1 = time_checker(new_tour1, travel_time, service_time, ready_time, due_time)
                time_check_2opts2 = time_checker(new_tour2, travel_time, service_time, ready_time, due_time)

                if time_check_2opts1 and time_check_2opts2:
                    twoopts_cost = round(
                        distance[tour1[i]][tour2[j + 1]] + distance[tour2[j]][tour1[i + 1]] - distance[tour1[i]][
                            tour1[i + 1]] - distance[tour2[j]][tour2[j + 1]], 10)

                    if twoopts_cost < best_imp:
                        best_imp = twoopts_cost
                        position1 = i
                        position2 = j

    return position1, position2, best_imp


def two_optstar_search(sub_tour, distance, travel_time, service_time, ready_time, due_time):
    Best_Imp = 0
    T1 = -1
    T2 = -1
    Position1 = -1
    Position2 = -1

    for t1 in range(len(sub_tour) - 1):
        for t2 in range(t1 + 1, len(sub_tour)):
            [position1, position2, imp] = two_optstar(sub_tour[t1], sub_tour[t2], distance, travel_time, service_time,
                                                      ready_time, due_time)

            if imp < Best_Imp:
                T1 = t1
                T2 = t2
                Position1 = position1
                Position2 = position2
                Best_Imp = imp

    return T1, T2, Position1, Position2, Best_Imp


# 10. λ-interchange

def interchange(tour1, tour2, distance, travel_time, service_time, ready_time, due_time, lam):
    best_imp = 0
    node11 = -1
    node12 = -1
    node21 = -1
    node22 = -1

    for i in range(1, len(tour1) - lam):
        for k in range(i - 1, i + lam):
            for j in range(1, len(tour2) - lam):
                for l in range(j - 1, j + lam):
                    tour1_new_demand = sum(demand[tour2[j:l + 1]]) + sum(demand[tour1]) - sum(demand[tour1[i:k + 1]])
                    tour2_new_demand = sum(demand[tour1[i:k + 1]]) + sum(demand[tour2]) - sum(demand[tour2[j:l + 1]])

                    if (tour1_new_demand <= capacity) and (tour2_new_demand <= capacity):

                        new_tour1 = tour1[:i] + tour2[j:l + 1] + tour1[k + 1:]
                        new_tour2 = tour2[:j] + tour1[i:k + 1] + tour2[l + 1:]

                        time_check_interchange1 = time_checker(new_tour1, travel_time, service_time, ready_time,
                                                               due_time)
                        time_check_interchange2 = time_checker(new_tour2, travel_time, service_time, ready_time,
                                                               due_time)

                        if time_check_interchange1 and time_check_interchange2:
                            interchange1 = round(
                                total_distance([new_tour1], distance) + total_distance([new_tour2], distance), 5)
                            interchange2 = round(total_distance([tour1], distance) + total_distance([tour2], distance),
                                                 5)
                            interchange_cost = interchange1 - interchange2

                            if interchange_cost < best_imp:
                                best_imp = interchange_cost
                                node11 = i
                                node12 = k
                                node21 = j
                                node22 = l

    return node11, node12, node21, node22, best_imp


def interchange_search(sub_tour, distance, travel_time, service_time, ready_time, due_time, lamda):
    Best_Imp = 0
    T1 = -1
    T2 = -1
    Node11 = -1
    Node12 = -1
    Node21 = -1
    Node22 = -1

    for t1 in range(len(sub_tour) - 1):
        for t2 in range(t1 + 1, len(sub_tour)):
            [node11, node12, node21, node22, imp] = interchange(sub_tour[t1], sub_tour[t2], distance, travel_time,
                                                                service_time, ready_time, due_time, lamda)

            if imp < Best_Imp:
                T1 = t1
                T2 = t2
                Node11 = node11
                Node12 = node12
                Node21 = node21
                Node22 = node22
                Best_Imp = imp

    return T1, T2, Node11, Node12, Node21, Node22, Best_Imp


# # 3. Metaheuristic: VNS
def VNS(Sub_tour, distance, travel_time, service_time, ready_time, due_time):
    Nb_NoImp = 0
    iteration = 0
    Improvement = float("inf")

    Neighbor_order = [4,5,6,7,8,9,10,0,1,2,3]
    #Neighbor_order=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10]
    # random.shuffle(Neighbor_order)

    #print(Neighbor_order)
    Neighbor_Str = 0
    stop = False

    while stop != True:
        iteration += 1
        #print('Neighbor',Neighbor_order[Neighbor_Str])

        if Neighbor_order[Neighbor_Str] == 0:  # 2-opt
            [Tour, Position1, Position2, Improvement] = two_opt_search(Sub_tour, distance, travel_time, service_time,
                                                                       ready_time, due_time)
        elif Neighbor_order[Neighbor_Str] == 1:  # Or-opt-1
            [Tour, Position1, Position2, Position3, Improvement] = or_opt_search(Sub_tour, distance, travel_time,
                                                                                 service_time, ready_time, due_time, 1)
        elif Neighbor_order[Neighbor_Str] == 2:  # Or-opt-2
            [Tour, Position1, Position2, Position3, Improvement] = or_opt_search(Sub_tour, distance, travel_time,
                                                                                 service_time, ready_time, due_time, 2)
        elif Neighbor_order[Neighbor_Str] == 3:  # Or-opt-3
            [Tour, Position1, Position2, Position3, Improvement] = or_opt_search(Sub_tour, distance, travel_time,
                                                                                 service_time, ready_time, due_time, 3)
        elif Neighbor_order[Neighbor_Str] == 4:  # 2-opt*
            [Tour1, Tour2, Position1, Position2, Improvement] = two_optstar_search(Sub_tour, distance, travel_time,
                                                                                   service_time, ready_time, due_time)
        elif Neighbor_order[Neighbor_Str] == 5:  # Relocation
            [Tour1, Tour2, Customer, Insert_Position, Improvement] = relocate_search(Sub_tour, distance, travel_time,
                                                                                     service_time, ready_time, due_time)
        elif Neighbor_order[Neighbor_Str] == 6:  # Exchange
            [Tour1, Tour2, Position1, Position2, Improvement] = exchange_search(Sub_tour, distance, travel_time,
                                                                                service_time, ready_time, due_time)
        elif Neighbor_order[Neighbor_Str] == 7:  # CROSS
            [Tour1, Tour2, Node11, Node12, Node21, Node22, Improvement] = CROSS_search(Sub_tour, distance, travel_time,
                                                                                       service_time, ready_time,
                                                                                       due_time)
        elif Neighbor_order[Neighbor_Str] == 8:  # ICROSS
            [Tour1, Tour2, Node11, Node12, Node21, Node22, Improvement] = ICROSS_search(Sub_tour, distance, travel_time,
                                                                                        service_time, ready_time,
                                                                                        due_time)
        elif Neighbor_order[Neighbor_Str] == 9:  # GENI
            [Tour1, Tour2, Node1, Node21, Node22, Improvement] = GENI_search(Sub_tour, distance, travel_time,
                                                                             service_time, ready_time, due_time)
        elif Neighbor_order[Neighbor_Str] == 10:  # λ-interchange
            [Tour1, Tour2, Node11, Node12, Node21, Node22, Improvement] = interchange_search(Sub_tour, distance,
                                                                                             travel_time, service_time,
                                                                                             ready_time, due_time, 2)

        Best_Improvement = round(Improvement, 5)

        if Best_Improvement < 0:
            # print(Neighbor_order[Neighbor_Str], Best_Improvement)
            # perform neighborhood operator
            if Neighbor_order[Neighbor_Str] == 0:  # 2-opt
                for t in range(len(Tour)):
                    tour = Sub_tour[Tour[t]]
                    New_tour = tour[0:Position1[t] + 1] + tour[Position1[t] + 1:Position2[t] + 1][::-1] + tour[
                                                                                                          Position2[
                                                                                                              t] + 1:]
                    Sub_tour[Tour[t]] = New_tour

            elif Neighbor_order[Neighbor_Str] == 1 or Neighbor_order[Neighbor_Str] == 2 or Neighbor_order[
                Neighbor_Str] == 3:  # Or-opt
                for t in range(len(Tour)):
                    tour = Sub_tour[Tour[t]]
                    if Position3[t] < Position1[t]:
                        New_tour = tour[:Position3[t] + 1] + tour[Position1[t] + 1:Position2[t] + 1] + tour[
                                                                                                       Position3[t] + 1:
                                                                                                       Position1[
                                                                                                           t] + 1] + tour[
                                                                                                                     Position2[
                                                                                                                         t] + 1:]
                    else:
                        New_tour = tour[:Position1[t] + 1] + tour[Position2[t] + 1:Position3[t] + 1] + tour[
                                                                                                       Position1[t] + 1:
                                                                                                       Position2[
                                                                                                           t] + 1] + tour[
                                                                                                                     Position3[
                                                                                                                         t] + 1:]
                    Sub_tour[Tour[t]] = New_tour

            elif Neighbor_order[Neighbor_Str] == 4:  # 2-opt*
                New_tour1 = Sub_tour[Tour1][:Position1 + 1] + Sub_tour[Tour2][Position2 + 1:]
                New_tour2 = Sub_tour[Tour2][:Position2 + 1] + Sub_tour[Tour1][Position1 + 1:]
                Sub_tour[Tour1] = New_tour1
                Sub_tour[Tour2] = New_tour2

            elif Neighbor_order[Neighbor_Str] == 5:  # Relocation
                Sub_tour[Tour2].insert(Insert_Position + 1, Sub_tour[Tour1][Customer])
                del Sub_tour[Tour1][Customer]

            elif Neighbor_order[Neighbor_Str] == 6:  # Exchange
                New_tour1 = Sub_tour[Tour1][:Position1] + [Sub_tour[Tour2][Position2]] + Sub_tour[Tour1][Position1 + 1:]
                New_tour2 = Sub_tour[Tour2][:Position2] + [Sub_tour[Tour1][Position1]] + Sub_tour[Tour2][Position2 + 1:]
                Sub_tour[Tour1] = New_tour1
                Sub_tour[Tour2] = New_tour2

            elif Neighbor_order[Neighbor_Str] == 7:  # CROSS
                New_tour1 = Sub_tour[Tour1][:Node11] + Sub_tour[Tour2][Node21:Node22 + 1] + Sub_tour[Tour1][Node12 + 1:]
                New_tour2 = Sub_tour[Tour2][:Node21] + Sub_tour[Tour1][Node11:Node12 + 1] + Sub_tour[Tour2][Node22 + 1:]
                Sub_tour[Tour1] = New_tour1
                Sub_tour[Tour2] = New_tour2

            elif Neighbor_order[Neighbor_Str] == 8:  # ICROSS
                New_tour1 = Sub_tour[Tour1][:Node11] + Sub_tour[Tour2][Node21:Node22 + 1][::-1] + Sub_tour[Tour1][
                                                                                                  Node12 + 1:]
                New_tour2 = Sub_tour[Tour2][:Node21] + Sub_tour[Tour1][Node11:Node12 + 1][::-1] + Sub_tour[Tour2][
                                                                                                  Node22 + 1:]
                Sub_tour[Tour1] = New_tour1
                Sub_tour[Tour2] = New_tour2

            elif Neighbor_order[Neighbor_Str] == 9:  # GENI
                New_tour1 = Sub_tour[Tour1][:Node1] + Sub_tour[Tour1][Node1 + 1:]
                New_tour2 = Sub_tour[Tour2][:Node21 + 1] + [Sub_tour[Tour1][Node1]] + [Sub_tour[Tour2][Node22]] + \
                            Sub_tour[Tour2][Node21 + 1:Node22] + Sub_tour[Tour2][Node22 + 1:]
                Sub_tour[Tour1] = New_tour1
                Sub_tour[Tour2] = New_tour2

            elif Neighbor_order[Neighbor_Str] == 10:  # λ-interchange
                New_tour1 = Sub_tour[Tour1][:Node11] + Sub_tour[Tour2][Node21:Node22 + 1] + Sub_tour[Tour1][Node12 + 1:]
                New_tour2 = Sub_tour[Tour2][:Node21] + Sub_tour[Tour1][Node11:Node12 + 1] + Sub_tour[Tour2][Node22 + 1:]
                Sub_tour[Tour1] = New_tour1
                Sub_tour[Tour2] = New_tour2

            Neighbor_Str = 0
            Nb_NoImp = 0

        else:
            Nb_NoImp += 1

            if Nb_NoImp > len(Neighbor_order):
                stop = True
            else:
                if Neighbor_Str >= len(Neighbor_order) - 1:
                    Neighbor_Str = 0
                    stop = False
                else:
                    Neighbor_Str += 1


def shaking(Input_tour, travel_time, service_time, ready_time, due_time, demand, Neighbor_Str):
    n = len(Input_tour) - 1
    Sub_tour = copy.deepcopy(Input_tour)
    shaking_start = time.time()
    #rnd = np.random
    #rnd.seed(0)

    #print('Shaking',Neighbor_Str)
    if Neighbor_Str == 0:  # 2-opt
        while True:
            while True:
                Tour = randint(0, n)
                if len(Sub_tour[Tour]) >= 5:
                    break

            Position1 = randint(0, len(Sub_tour[Tour]) - 4)
            Position2 = randint(Position1 + 2, len(Sub_tour[Tour]) - 2)

            New_tour = Sub_tour[Tour][0:Position1 + 1] + Sub_tour[Tour][Position1 + 1:Position2 + 1][::-1] + Sub_tour[
                                                                                                                 Tour][
                                                                                                             Position2 + 1:]
            time_check = time_checker(New_tour, travel_time, service_time, ready_time, due_time)

            shaking_end = time.time()

            if time_check:
                Sub_tour[Tour] = New_tour
                break
            elif shaking_end - shaking_start > 0.5:
                break


    elif Neighbor_Str == 1 or Neighbor_Str == 2 or Neighbor_Str == 3:  # Or-opt
        K = Neighbor_Str
        while True:
            while True:
                Tour = randint(0, n)
                if len(Sub_tour[Tour]) >= K + 3:
                    break

            Position1 = randint(0, len(Sub_tour[Tour]) - K - 2)
            Position2 = Position1 + K
            Position3 = randint(0, len(Sub_tour[Tour]) - 2)

            while Position1 <= Position3 and Position3 <= Position2:
                Position3 = randint(0, len(Sub_tour[Tour]) - 2)

            if Position3 < Position1:
                New_tour = Sub_tour[Tour][:Position3 + 1] + Sub_tour[Tour][Position1 + 1:Position2 + 1] + Sub_tour[
                                                                                                              Tour][
                                                                                                          Position3 + 1:Position1 + 1] + \
                           Sub_tour[Tour][Position2 + 1:]
            else:
                New_tour = Sub_tour[Tour][:Position1 + 1] + Sub_tour[Tour][Position2 + 1:Position3 + 1] + Sub_tour[
                                                                                                              Tour][
                                                                                                          Position1 + 1:Position2 + 1] + \
                           Sub_tour[Tour][Position3 + 1:]

            time_check = time_checker(New_tour, travel_time, service_time, ready_time, due_time)

            shaking_end = time.time()

            if time_check:
                Sub_tour[Tour] = New_tour
                break
            elif shaking_end - shaking_start > 0.5:
                break

    elif Neighbor_Str == 4 and n > 0:  # 2-optstar
        while True:
            Tour1 = randint(0, n - 1)
            Tour2 = randint(Tour1 + 1, n)
            Position1 = randint(1, len(Sub_tour[Tour1]) - 2)
            Position2 = randint(1, len(Sub_tour[Tour2]) - 2)

            New_tour1 = Sub_tour[Tour1][:Position1 + 1] + Sub_tour[Tour2][Position2 + 1:]
            New_tour2 = Sub_tour[Tour2][:Position2 + 1] + Sub_tour[Tour1][Position1 + 1:]

            time_check1 = time_checker(New_tour1, travel_time, service_time, ready_time, due_time)
            time_check2 = time_checker(New_tour2, travel_time, service_time, ready_time, due_time)
            new_tour1_demand = sum(demand[New_tour1])
            new_tour2_demand = sum(demand[New_tour2])

            shaking_end = time.time()

            if time_check1 and time_check2 and new_tour1_demand <= capacity and new_tour2_demand <= capacity:
                Sub_tour[Tour1] = New_tour1
                Sub_tour[Tour2] = New_tour2
                break
            elif shaking_end - shaking_start > 0.5:
                break


    elif Neighbor_Str == 5 and n > 0:  # Relocation

        while True:
            Tour1 = randint(0, n)
            Tour2 = randint(0, n)
            while Tour1 == Tour2:
                Tour2 = randint(0, n)

            Customer = randint(1, len(Sub_tour[Tour1]) - 2)
            Insert_Position = randint(0, len(Sub_tour[Tour2]) - 2)

            newtour = Sub_tour[Tour2][:Insert_Position + 1] + [Sub_tour[Tour1][Customer]] + Sub_tour[Tour2][
                                                                                            Insert_Position + 1:]
            time_check = time_checker(newtour, travel_time, service_time, ready_time, due_time)
            tour2_demand = demand[Sub_tour[Tour1][Customer]] + sum(demand[Sub_tour[Tour2]])

            shaking_end = time.time()

            if time_check and tour2_demand <= capacity:
                Sub_tour[Tour2].insert(Insert_Position + 1, Sub_tour[Tour1][Customer])
                del Sub_tour[Tour1][Customer]
                break
            elif shaking_end - shaking_start > 0.5:
                break


    elif Neighbor_Str == 6 and n > 0:  # Exchange

        while True:
            Tour1 = randint(0, n - 1)
            Tour2 = randint(Tour1 + 1, n)
            Position1 = randint(1, len(Sub_tour[Tour1]) - 2)
            Position2 = randint(1, len(Sub_tour[Tour2]) - 2)

            New_tour1 = Sub_tour[Tour1][:Position1] + [Sub_tour[Tour2][Position2]] + Sub_tour[Tour1][Position1 + 1:]
            New_tour2 = Sub_tour[Tour2][:Position2] + [Sub_tour[Tour1][Position1]] + Sub_tour[Tour2][Position2 + 1:]

            time_check1 = time_checker(New_tour1, travel_time, service_time, ready_time, due_time)
            time_check2 = time_checker(New_tour2, travel_time, service_time, ready_time, due_time)
            new_tour1_demand = sum(demand[New_tour1])
            new_tour2_demand = sum(demand[New_tour2])

            shaking_end = time.time()

            if time_check1 and time_check2 and new_tour1_demand <= capacity and new_tour2_demand <= capacity:
                Sub_tour[Tour1] = New_tour1
                Sub_tour[Tour2] = New_tour2
                break
            elif shaking_end - shaking_start > 0.5:
                break

    elif Neighbor_Str == 7 and n > 0:  # CROSS

        while True:
            while True:
                Tour1 = randint(0, n - 1)
                Tour2 = randint(Tour1 + 1, n)
                if len(Sub_tour[Tour1]) >= 4 and len(Sub_tour[Tour2]) >= 4:
                    break

            Node11 = randint(1, len(Sub_tour[Tour1]) - 3)
            Node12 = randint(Node11, len(Sub_tour[Tour1]) - 2)
            Node21 = randint(1, len(Sub_tour[Tour2]) - 3)
            Node22 = randint(Node21, len(Sub_tour[Tour2]) - 2)

            New_tour1 = Sub_tour[Tour1][:Node11] + Sub_tour[Tour2][Node21:Node22 + 1] + Sub_tour[Tour1][Node12 + 1:]
            New_tour2 = Sub_tour[Tour2][:Node21] + Sub_tour[Tour1][Node11:Node12 + 1] + Sub_tour[Tour2][Node22 + 1:]

            time_check1 = time_checker(New_tour1, travel_time, service_time, ready_time, due_time)
            time_check2 = time_checker(New_tour2, travel_time, service_time, ready_time, due_time)
            new_tour1_demand = sum(demand[New_tour1])
            new_tour2_demand = sum(demand[New_tour2])

            shaking_end = time.time()

            if time_check1 and time_check2 and new_tour1_demand <= capacity and new_tour2_demand <= capacity:
                Sub_tour[Tour1] = New_tour1
                Sub_tour[Tour2] = New_tour2
                break
            elif shaking_end - shaking_start > 0.5:
                break

    elif Neighbor_Str == 8 and n > 0:  # ICROSS

        while True:
            while True:
                Tour1 = randint(0, n - 1)
                Tour2 = randint(Tour1 + 1, n)
                if len(Sub_tour[Tour1]) >= 4 and len(Sub_tour[Tour2]) >= 4:
                    break

            Node11 = randint(1, len(Sub_tour[Tour1]) - 3)
            Node12 = randint(Node11, len(Sub_tour[Tour1]) - 2)
            Node21 = randint(1, len(Sub_tour[Tour2]) - 3)
            Node22 = randint(Node21, len(Sub_tour[Tour2]) - 2)

            New_tour1 = Sub_tour[Tour1][:Node11] + Sub_tour[Tour2][Node21:Node22 + 1][::-1] + Sub_tour[Tour1][
                                                                                              Node12 + 1:]
            New_tour2 = Sub_tour[Tour2][:Node21] + Sub_tour[Tour1][Node11:Node12 + 1][::-1] + Sub_tour[Tour2][
                                                                                              Node22 + 1:]

            time_check1 = time_checker(New_tour1, travel_time, service_time, ready_time, due_time)
            time_check2 = time_checker(New_tour2, travel_time, service_time, ready_time, due_time)
            new_tour1_demand = sum(demand[New_tour1])
            new_tour2_demand = sum(demand[New_tour2])

            shaking_end = time.time()

            if time_check1 and time_check2 and new_tour1_demand <= capacity and new_tour2_demand <= capacity:
                Sub_tour[Tour1] = New_tour1
                Sub_tour[Tour2] = New_tour2
                break
            elif shaking_end - shaking_start > 0.5:
                break


    elif Neighbor_Str == 9 and n > 0:  # GENI

        while True:
            while True:
                Tour1 = randint(0, n)
                Tour2 = randint(0, n)
                if len(Sub_tour[Tour2]) >= 4 and Tour1 != Tour2:
                    break

            Node1 = randint(1, len(Sub_tour[Tour1]) - 2)
            Node21 = randint(0, len(Sub_tour[Tour2]) - 4)
            Node22 = randint(Node21 + 2, len(Sub_tour[Tour2]) - 2)

            New_tour1 = Sub_tour[Tour1][:Node1] + Sub_tour[Tour1][Node1 + 1:]
            New_tour2 = Sub_tour[Tour2][:Node21 + 1] + [Sub_tour[Tour1][Node1]] + [Sub_tour[Tour2][Node22]] + Sub_tour[
                                                                                                                  Tour2][
                                                                                                              Node21 + 1:Node22] + \
                        Sub_tour[Tour2][Node22 + 1:]

            time_check1 = time_checker(New_tour1, travel_time, service_time, ready_time, due_time)
            time_check2 = time_checker(New_tour2, travel_time, service_time, ready_time, due_time)
            new_tour1_demand = sum(demand[New_tour1])
            new_tour2_demand = sum(demand[New_tour2])

            shaking_end = time.time()

            if time_check1 and time_check2 and new_tour1_demand <= capacity and new_tour2_demand <= capacity:
                Sub_tour[Tour1] = New_tour1
                Sub_tour[Tour2] = New_tour2
                break
            elif shaking_end - shaking_start > 0.5:
                break

    elif Neighbor_Str == 10 and n > 0:  #λ-interchange

        while True:
            while True:
                Tour1 = randint(0, n - 1)
                Tour2 = randint(Tour1 + 1, n)
                if len(Sub_tour[Tour1]) >= 4 and len(Sub_tour[Tour2]) >= 4:
                    break

            Node11 = randint(1, len(Sub_tour[Tour1]) - 3)
            Node12 = randint(Node11, len(Sub_tour[Tour1]) - 2)
            Node21 = randint(1, len(Sub_tour[Tour2]) - 3)
            Node22 = randint(Node21, len(Sub_tour[Tour2]) - 2)

            New_tour1 = Sub_tour[Tour1][:Node11] + Sub_tour[Tour2][Node21:Node22 + 1] + Sub_tour[Tour1][Node12 + 1:]
            New_tour2 = Sub_tour[Tour2][:Node21] + Sub_tour[Tour1][Node11:Node12 + 1] + Sub_tour[Tour2][Node22 + 1:]

            time_check1 = time_checker(New_tour1, travel_time, service_time, ready_time, due_time)
            time_check2 = time_checker(New_tour2, travel_time, service_time, ready_time, due_time)
            new_tour1_demand = sum(demand[New_tour1])
            new_tour2_demand = sum(demand[New_tour2])

            shaking_end = time.time()

            if time_check1 and time_check2 and new_tour1_demand <= capacity and new_tour2_demand <= capacity:
                Sub_tour[Tour1] = New_tour1
                Sub_tour[Tour2] = New_tour2
                break
            elif shaking_end - shaking_start > 0.5:
                break

    return Sub_tour



# folder = ["R101", "R102", "R103", "R104", "R105", "R106", "R107", "R108", "R109", "R110", "R111", "R112",
#           "R201", "R202", "R203", "R204", "R205", "R206", "R207", "R208", "R209", "R210", "R211",
#           "RC101", "RC102", "RC103", "RC104", "RC105", "RC106", "RC107", "RC108",
#           "RC201", "RC202", "RC203", "RC204", "RC205", "RC206", "RC207", "RC208",
#           "C201", "C202", "C203", "C204", "C205", "C206", "C207", "C208",
#           "C101", "C102", "C103", "C104", "C105", "C106", "C107", "C108", "C109"]

file_names = ["R101", "R106", "C101", "C106", "RC101", "RC106", "R201", "R206", "C201", "C206", "RC201", "RC206"]
cust_size = 25


# CW's parameters
muy_sv = 1
#
## MAINCODE
#for file in file_names:
#    file_name = os.path.join(dir_name, 'Sample Dataset\\', file + '.csv')
#    data = np.genfromtxt(file_name, delimiter=',')
#    df = data[1:cust_size + 2]
#
#    customer = df[:, 0]
#    xcoor = df[:, 1]
#    ycoor = df[:, 2]
#    demand = df[:, 3]
#    readytime = df[:, 4]
#    duetime = df[:, 5]
#    servicetime = df[:, 6]
#    capacity = df[0, -1]
#
#    # Calculate Distance Matrix
#
#    dist_matrix = np.zeros((cust_size + 1, cust_size + 1))
#    for i in range(cust_size + 1):
#        for j in range(cust_size + 1):
#            dist_matrix[i][j] = np.sqrt((xcoor[i] - xcoor[j]) ** 2 + (ycoor[i] - ycoor[j]) ** 2)
#
#    print('Start', file, cust_size)
#
#    # Route Construction
#    Sub_tour = []
#    for i in range(cust_size + 1):
#        sub = [0, i, 0]
#        Sub_tour.append(sub)
#    sub_size = len(Sub_tour) - 1
#
#    saving_list = []
#    for i in range(1, sub_size + 1):
#        for j in range(1, sub_size + 1):
#            if i != j:
#                saving = round(dist_matrix[i, 0] + dist_matrix[0, j] - muy_sv * dist_matrix[i, j], 2)
#                saving_list.append([i, j, saving])
#
#    np.asarray(saving_list)
#    saving_list_sorted = sorted(saving_list, key=lambda x: x[2], reverse=True)
#
#    while len(saving_list_sorted) != 0:
#        ind = saving_list_sorted[0][:2]
#        a = [i for i in range(len(Sub_tour)) if ind[0] in Sub_tour[i]]
#        b = [i for i in range(len(Sub_tour)) if ind[1] in Sub_tour[i]]
#
#        if a != b:
#            new_sub11 = Sub_tour[a[0]][::-1][:-1] + Sub_tour[b[0]][1:]
#            new_sub12 = Sub_tour[a[0]][::-1][:-1] + Sub_tour[b[0]][::-1][1:]
#            new_sub13 = Sub_tour[a[0]][:-1] + Sub_tour[b[0]][1:]
#            new_sub14 = Sub_tour[a[0]][:-1] + Sub_tour[b[0]][::-1][1:]
#            # print(new_sub11,new_sub12)
#
#            if check_sequence(ind, new_sub11):
#                new_sub1 = new_sub11
#            elif check_sequence(ind, new_sub12):
#                new_sub1 = new_sub12
#            elif check_sequence(ind, new_sub13):
#                new_sub1 = new_sub13
#            else:
#                new_sub1 = new_sub14
#
#            new_sub2 = new_sub1[::-1]
#
#            time_check1 = time_checker(new_sub1, dist_matrix, servicetime, readytime, duetime)
#            time_check2 = time_checker(new_sub2, dist_matrix, servicetime, readytime, duetime)
#
#            if time_check1:
#                new_sub = new_sub1
#            elif time_check2:
#                new_sub = new_sub2
#
#            merge_demand = sum(demand[Sub_tour[a[0]]]) + sum(demand[Sub_tour[b[0]]])
#
#            del saving_list_sorted[0]
#            k = [i for i in range(len(saving_list_sorted)) if
#                 saving_list_sorted[i][0] == ind[1] or saving_list_sorted[i][1] == ind[0]]
#            del saving_list_sorted[k[0]]
#
#            # print(saving_list_sorted)
#
#            if (merge_demand <= capacity and (time_check1 or time_check2)):
#                for ele in sorted([a[0], b[0]], reverse=True):
#                    del Sub_tour[ele]
#
#                unlink_nodes = new_sub[2:-2]
#
#                for node in unlink_nodes:
#                    index = [i for i in range(len(saving_list_sorted)) if
#                             saving_list_sorted[i][0] == node or saving_list_sorted[i][1] == node]
#                    for j in sorted(index, reverse=True):
#                        del saving_list_sorted[j]
#
#                Sub_tour.append(new_sub)
#        else:
#            del saving_list_sorted[0]
#            k = [i for i in range(len(saving_list_sorted)) if
#                 saving_list_sorted[i][0] == ind[1] or saving_list_sorted[i][1] == ind[0]]
#            del saving_list_sorted[k[0]]
#
#    Ini_tour = Sub_tour[1:]
#    vrp_graph(Ini_tour, xcoor, ycoor)
#
#    print('0', total_distance(Ini_tour, dist_matrix), len(Ini_tour))
#
#    # VNS
#    rnd = np.random
#    rnd.seed(0)
#
#    DIST = []
#    NO_VEHICLE = []
#    RUN_TIME = []
#
#    # MAINCODE
#    for counter in range(10):
#        Sub_tour_VNS = copy.deepcopy(Ini_tour)
#
#        Shaking_Neighbor = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10]
#
#        time_start = time.time()
#
#        NO_IMP = 0
#        n = 0
#        STOP = False
#        while STOP == False:
#            Sub_tour_shaking = shaking(Sub_tour_VNS, dist_matrix, servicetime, readytime, duetime, demand,
#                                       Shaking_Neighbor[n])
#
#            #print(Sub_tour_shaking)
#            #print(Sub_tour_VNS)
#            VNS(Sub_tour_shaking, dist_matrix, dist_matrix, servicetime, readytime, duetime)
#            #print(Sub_tour_shaking)
#            #print(Sub_tour_VNS)
#
#            if total_distance(Sub_tour_shaking, dist_matrix) < total_distance(Sub_tour_VNS, dist_matrix):
#                #a=total_distance(Sub_tour_shaking, dist_matrix)
#                #print(a)
#                Sub_tour_VNS = copy.deepcopy(Sub_tour_shaking)
#                n = 0
#                NO_IMP = 0
#            else:
#                NO_IMP += 1
#
#                if NO_IMP > len(Shaking_Neighbor):
#                    STOP = True
#                else:
#                    if n >= len(Shaking_Neighbor) - 1:
#                        n = 0
#                        STOP = False
#                    else:
#                        n += 1
#                        STOP = False
#            Sub_tour_VNS = [Sub_tour_VNS[i] for i in range(len(Sub_tour_VNS)) if
#                            len(Sub_tour_VNS[i]) > 2]  # Remove empty tour
#
#        time_end = time.time()
#
#        dist = total_distance(Sub_tour_VNS, dist_matrix)
#        no_veh = len(Sub_tour_VNS)
#        time_exe = time_end - time_start
#
#        print(counter + 1, dist, no_veh, time_exe)
#
#        DIST.append(dist)
#        NO_VEHICLE.append(no_veh)
#        RUN_TIME.append(time_exe)
#        print(Sub_tour_VNS)
#
#
#    print('\n', min(NO_VEHICLE), np.mean(NO_VEHICLE), np.std(NO_VEHICLE))
#    print(min(DIST), np.mean(DIST), np.std(DIST), np.mean(RUN_TIME))
#    print("====================")
#        #print(Sub_tour_VNS)
#
#    # VRP_graph(Sub_tour_VNS, xcoor, ycoor)
#    # print(total_distance(Sub_tour_VNS, dist_matrix))