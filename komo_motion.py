import robotic as ry
import numpy as np
def start_points(n, y_value, start_x):
    points = [(round(start_x + i*1.2,2), y_value,0.4) for i in range(n)]
    return np.array(points)

def init_komo(num_objects,C,pos):
    C.addFile("robot_free.g")
    pos_dict = {}
    start_list = start_points(num_objects, -1.3, -3)
    for node_index, position in pos[0].items():
        pos_dict[node_index] = [start_list[node_index], position]
    for obj in pos_dict:
        name = "object"+str(obj)
        target = "target"+str(obj)
        pos = pos_dict[obj][0]
        target_pos = pos_dict[obj][1]
        C.addFrame(name).setShape(ry.ST.ssBox, [0.8, 0.8, 0.8, .01]).setColor([0.5,0.5,0.5]).setPosition(pos)
        C.addFrame(target).setShape(ry.ST.marker, [.1]) .setPosition(target_pos)

def define_optimization(C,obj_index,komo):
    komo.addControlObjective([], 1, 1e0)
    obj_name = "object"+str(obj_index)
    target_name = "target"+str(obj_index)
    komo.addObjective([float(1)], ry.FS.positionDiff, ['r_endeffector', obj_name], ry.OT.eq, [1e3],[0,0,0])
    komo.addObjective([float(1)], ry.FS.jointState, [], ry.OT.eq, [1e1], [], order=1)
    komo.addModeSwitch([float(2),float(3)], ry.SY.stable, ['r_endeffector', obj_name])
    komo.addObjective([float(3)], ry.FS.positionDiff, [obj_name, target_name], ry.OT.eq, [1e2],[0,0,0])
    # komo.addObjective([float(3)], ry.FS.vectorZ, [obj_name], ry.OT.eq, [1e2], [0., 0., 0.])
    # komo.addObjective([float(3)], ry.FS.vectorX, [obj_name], ry.OT.eq, [1e2], [0., 0., 0.])
    # komo.addObjective([float(3)], ry.FS.vectorY, [obj_name], ry.OT.eq, [1e2], [0., 0., 0.])
    komo.addObjective([float(3)], ry.FS.jointState, [], ry.OT.eq, [1e1], [], order=1)
    komo.addModeSwitch([float(3),-1], ry.SY.stable, [target_name,obj_name])
    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq,[1e1])
