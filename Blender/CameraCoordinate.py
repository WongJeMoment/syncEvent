import bpy
import mathutils
import csv

# 设置参数
object_name = "Cube"         # 你的物体名称
camera_name = "Camera"       # 相机名称
output_path = "C:/Users/63508/Desktop/pose_output.csv"


obj = bpy.data.objects[object_name]
cam = bpy.data.objects[camera_name]

# 打开CSV写入文件
with open(output_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Frame", "Tx", "Ty", "Tz", "Rx", "Ry", "Rz"])  # 头部

    # 遍历每一帧
    for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1):
        bpy.context.scene.frame_set(frame)

        # 获取矩阵
        obj_matrix_world = obj.matrix_world
        cam_matrix_world = cam.matrix_world

        # 相对变换（物体在相机坐标系下）
        obj_in_cam = cam_matrix_world.inverted() @ obj_matrix_world

        # 提取位移和欧拉角
        trans = obj_in_cam.to_translation()
        rot = obj_in_cam.to_euler('XYZ')  # 可选 'XYZ', 'ZYX', etc.

        writer.writerow([
            frame,
            round(trans.x, 6), round(trans.y, 6), round(trans.z, 6),
            round(rot.x, 6), round(rot.y, 6), round(rot.z, 6)
        ])

print("导出完毕 ✅，保存在：", output_path)