# ros message for one 6d grasp planning result

string label    # Object label
float32 score   # Grasp score
float32 width   # Grasp width
float32 depth   # Grasp depth
float32[] rotation_matrix  # Rotation matrix of predict grasp in camera frame, order in row [T_00, T_01, T_02, T_10, T_11, ...], length=9x1
float32[] translation  # Translation of predict grasp in camera frame, order in row [T_x, T_y, T_z], length=3x1