from planner_utils import *
from graph_utils import *
from gnn import *

import os
import random
import robotic as ry
import time

in_channels = 1  
hidden_channels = 32 
out_channels = 1

model = GNNModel(in_channels, hidden_channels, out_channels)

model_path = 'iros_model1.pth'  # Update with your file path
model.load_state_dict(torch.load(model_path))
target_graph, pos = process_all_g_files('target')
correct_graph_edge_indices(target_graph)
target_pyg = [convert_to_pyg_data(graph) for graph, positions in zip(target_graph, pos)][0]
test_data = target_pyg
target_folder = 'target'
removal_order = run_inference_and_reduce_graph(target_folder,model)
building_order = removal_order[::-1]

print("Building Order:", building_order)

C = ry.Config()
C.addFile("robot_free.g")

def start_points(n, y_value, start_x):
    points = [(round(start_x + i*1.2,2), y_value,0.4) for i in range(n)]
    return np.array(points)

start_list = start_points(len(building_order), -1.3, -3)
pos_dict = {}
for node_index, position in pos[0].items():
    pos_dict[node_index] = [start_list[node_index], position]

for obj in pos_dict:
    name = "object"+str(obj)
    target = "target"+str(obj)
    pos = pos_dict[obj][0]
    target_pos = pos_dict[obj][1]
    C.addFrame(name).setShape(ry.ST.ssBox, [0.8, 0.8, 0.8, .01]).setColor([0.5,0.5,0.5]).setPosition(pos)
    C.addFrame(target).setShape(ry.ST.marker, [.1]) .setPosition(target_pos)

num_objects = len(building_order)
komo = ry.KOMO(C, 3*num_objects, 30, 1, False)

komo.addControlObjective([], 1, 1e0)

for i,j in enumerate(building_order):
    obj_name = "object"+str(j)
    target_name = "target"+str(j)
    komo.addObjective([float(3*i+1)], ry.FS.positionDiff, ['r_endeffector', obj_name], ry.OT.eq, [1e3],[0,0,0])
    komo.addObjective([float(3*i+1)], ry.FS.jointState, [], ry.OT.eq, [1e1], [], order=1)
    komo.addModeSwitch([float(3*i+2),float(3*i+3)], ry.SY.stable, ['r_endeffector', obj_name])
    komo.addObjective([float(3*i+3)], ry.FS.positionDiff, [obj_name, target_name], ry.OT.eq, [1e2],[0,0,0])
    komo.addObjective([float(3*i+3)], ry.FS.vectorZ, [obj_name], ry.OT.eq, [1e2], [0., 0., 0.])
    komo.addObjective([float(3*i+3)], ry.FS.vectorX, [obj_name], ry.OT.eq, [1e2], [0., 0., 0.])
    komo.addObjective([float(3*i+3)], ry.FS.vectorY, [obj_name], ry.OT.eq, [1e2], [0., 0., 0.])
    komo.addObjective([float(3*i+3)], ry.FS.jointState, [], ry.OT.eq, [1e1], [], order=1)
    komo.addModeSwitch([float(3*i+3),-1], ry.SY.stable, [target_name,obj_name])
komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq,[1e1])

ret = ry.NLP_Solver(komo.nlp(), verbose=0 ).solve()
print(ret)
q = komo.getPath()
print('size of path:', q.shape)

for t in range(q.shape[0]):
    C.setJointState(q[t])
    time.sleep(.1)

komo.view_play(True, .1)
