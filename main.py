from planner_utils import *
from graph_utils import *
from gnn import *
from komo_motion import *

import os
import sys
import random
import robotic as ry
import time
import contextlib

def optimize(obj_index):
    komo = ry.KOMO(C, 3, 30, 1, False)
    define_optimization(C,obj_index,komo)
    ret = ry.NLP_Solver(komo.nlp(), verbose=0).solve()
    q = komo.getPath()
    return q,ret,komo

in_channels = 1  
hidden_channels = 32 
out_channels = 1
C = ry.Config()
C2 = ry.Config()
model = GNNModel(in_channels, hidden_channels, out_channels)

model_path = 'iros_model1.pth'  # Update with your file path
model.load_state_dict(torch.load(model_path))
target_graph, pos = process_all_g_files('target')
correct_graph_edge_indices(target_graph)
target_pyg = [convert_to_pyg_data(graph) for graph, positions in zip(target_graph, pos)][0]
#test_data = target_pyg
target_folder = 'target'

plan = create_plan(target_folder,model,option=1)
building_order = plan[0]
print(building_order)

print("Building Order:", building_order)
object_number = len(building_order)
init_komo(object_number,C,pos)

num_objects = len(building_order)
objective_realized =False

last_state = None
completed_checkpoints = []
states = []
checkpoint_counter = 0
plan_lst = []

while objective_realized != True:
    if checkpoint_counter == 0:
        obj_index = building_order[checkpoint_counter]
        q,ret,komo = optimize(obj_index)
    if ret.feasible:
        if checkpoint_counter !=0:
            C.setFrameState(last_state)
        checkpoint_counter += 1
        graph = plan[1]
        plan = create_plan(target_folder,model,graph=graph,option=2)
        building_order = plan[0]
        print(building_order)
        obj_index = building_order[0]
        q, ret,komo = optimize(obj_index)
        waypoints = komo.getPath_qAll()
        states.append(komo.getFrameState(len(waypoints)-1))
        last_state = komo.getFrameState(len(waypoints)-1)
        #print(last_state)
        if checkpoint_counter == num_objects -1:
            init_komo(object_number,C2,pos)
            for s in states:
                C2.setFrameState(s)
                time.sleep(.5)
                C2.view(True)
            print("done")
            objective_realized =True
            break
    else:
        print("Plan infeasible. Replanning...")
        #online_train(model, 0.001, new_graph_data, batch_size=1)
        if checkpoint_counter == 0:
            building_order = create_plan(target_folder,model,option=1)[0]
        else:
            building_order = create_plan(target_folder,model,graph=graph,option=2)[0]


