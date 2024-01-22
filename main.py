from planner_utils import *
from graph_utils import *
from gnn import *
from komo_motion import *

import os
import random
import robotic as ry
import time

in_channels = 1  
hidden_channels = 32 
out_channels = 1
C = ry.Config()
model = GNNModel(in_channels, hidden_channels, out_channels)

model_path = 'iros_model1.pth'  # Update with your file path
model.load_state_dict(torch.load(model_path))
target_graph, pos = process_all_g_files('target')
correct_graph_edge_indices(target_graph)
target_pyg = [convert_to_pyg_data(graph) for graph, positions in zip(target_graph, pos)][0]
test_data = target_pyg
target_folder = 'target'

plan = create_plan(target_folder,model,option=1)
building_order = plan[0]

print("Building Order:", building_order)
object_number = len(building_order)
init_komo(object_number,C,pos)

num_objects = len(building_order)
objective_realized =False

completed_checkpoints = []
checkpoint_counter = 0

while objective_realized != True:
    if checkpoint_counter == 0:
        komo = ry.KOMO(C, 3, 30, 1, False)
        obj_index = building_order[checkpoint_counter]
        define_optimization(C,obj_index,komo)
        ret = ry.NLP_Solver(komo.nlp(), verbose=0).solve()
        q = komo.getPath()
    if ret.feasible:
        checkpoint_counter += 1
        graph = plan[1]
        plan = create_plan(target_folder,model,graph=graph,option=2)
        building_order = plan[0]
        if checkpoint_counter == num_objects -1:
            print("Planning Completed")
            print("Setting joint states...")
            for t in range(q.shape[0]):
                C.setJointState(q[t])
                time.sleep(.1)
            print("Completed")
            objective_realized =True
            break
    else:
        #online_train(model, 0.001, new_graph_data, batch_size=1)
        if checkpoint_counter == 0:
            building_order = create_plan(target_folder,model,option=1)[0]
        else:
            building_order = create_plan(target_folder,model,graph=graph,option=2)[0]
    komo = ry.KOMO(C, 3, 30, 1, False)
    obj_index = building_order[0]
    define_optimization(C,obj_index,komo)
    ret = ry.NLP_Solver(komo.nlp(), verbose=0).solve()
    q = komo.getPath()

