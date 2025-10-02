#!/usr/bin/env python3

import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import PointCloud2
import sqlite3
import os
import struct
from pathlib import Path

def read_point_cloud2(msg):
    """Читает PointCloud2 и возвращает список точек со всеми полями"""
    points = []
    
    # Получаем информацию о полях
    field_names = [field.name for field in msg.fields]
    point_step = msg.point_step
    data = msg.data
    
    for i in range(0, len(data), point_step):
        point_data = data[i:i+point_step]
        point = {}
        
        for field in msg.fields:
            offset = field.offset
            dtype = None
            
            if field.datatype == 7:  # FLOAT32
                value = struct.unpack('f', point_data[offset:offset+4])[0]
            elif field.datatype == 8:  # FLOAT64
                value = struct.unpack('d', point_data[offset:offset+8])[0]
            elif field.datatype == 2:  # UINT8
                value = struct.unpack('B', point_data[offset:offset+1])[0]
            elif field.datatype == 3:  # INT8
                value = struct.unpack('b', point_data[offset:offset+1])[0]
            elif field.datatype == 4:  # UINT16
                value = struct.unpack('H', point_data[offset:offset+2])[0]
            elif field.datatype == 5:  # INT16
                value = struct.unpack('h', point_data[offset:offset+2])[0]
            elif field.datatype == 6:  # UINT32
                value = struct.unpack('I', point_data[offset:offset+4])[0]
            elif field.datatype == 1:  # INT32
                value = struct.unpack('i', point_data[offset:offset+4])[0]
            else:
                value = 0
            
            point[field.name] = value
        
        points.append(point)
    
    return points

def write_pcd_file(points, filename, fields=None):
    """Записывает точки в PCD файл"""
    if not fields:
        fields = ['x', 'y', 'z']  # базовые поля
    
    with open(filename, 'w') as f:
        # Заголовок PCD
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write(f"FIELDS {' '.join(fields)}\n")
        f.write(f"SIZE {' '.join(['4']*len(fields))}\n")
        f.write(f"TYPE {' '.join(['F']*len(fields))}\n")
        f.write(f"COUNT {' '.join(['1']*len(fields))}\n")
        f.write(f"WIDTH {len(points)}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {len(points)}\n")
        f.write("DATA ascii\n")
        
        # Данные точек
        for point in points:
            line = ' '.join([str(point.get(field, 0)) for field in fields])
            f.write(line + '\n')

def convert_bag_to_pcd_advanced(bag_file, output_dir, topic_name):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(bag_file)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT messages.timestamp, topics.name, messages.data 
        FROM messages 
        JOIN topics ON messages.topic_id = topics.id 
        WHERE topics.name = ?
        ORDER BY messages.timestamp
    """, (topic_name,))
    
    msg_type = get_message('sensor_msgs/msg/PointCloud2')
    
    count = 0
    for timestamp, topic, data in cursor.fetchall():
        msg = deserialize_message(data, msg_type)
        points = read_point_cloud2(msg)
        
        if len(points) > 0:
            # Определяем доступные поля
            available_fields = list(points[0].keys())
            
            output_file = os.path.join(output_dir, f"cloud_{count:06d}.pcd")
            write_pcd_file(points, output_file, available_fields)
            print(f"Сохранено: {output_file} с {len(points)} точками")
            
            count += 1
    
    conn.close()
    print(f"Всего преобразовано {count} облаков")

if __name__ == "__main__":
    bag_file = "path/file_name.db3"
    output_dir = "pcd_output"
    topic_name = "/sensing/lidar/concatenated/pointcloud"  # укажите ваш топик

    convert_bag_to_pcd_advanced(bag_file, output_dir, topic_name)