# QuadRectify 结果: 在 DetItem 基础上增加四角/中心/方向

string label            # 物体 label
float32 confidence      # 检测置信度
int16[4] bbox           # 边界框 [x1, y1, x2, y2]
sensor_msgs/Image mask   # 填满四边形后的 mask (mono8)

# 四角 (图像坐标, 顺序 TL, TR, BR, BL), 8 个数: x,y,x,y,x,y,x,y
float32[8] corners
# 中心 (cx, cy)
float32[2] center
# 长边单位方向 (dx, dy)
float32[2] direction
# 是否成功做了四边拟合 (false 表示退化为原始 mask)
bool rectified
