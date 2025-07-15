#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2022 Aeva, Inc
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Script to run registration algorithms on a sequence.

The dataset structure is as follows:

    REPOSITORY_ROOT/dataset/
    ├── sequence_01/
    │   ├── point_clouds/
    │   │   ├── 00001.bin  # N * (3 + 1) float32 bytes containing XYZ points
    │   │   ├── 00002.bin  # and Doppler velocities.
    │   │   └── ...
    │   ├── calibration.json
    │   └── ref_poses.txt  # N reference poses with timestamps in TUM format.
    ├── sequence_02/
    │   └── ...
    └── ...

Example usage:

    # Run Doppler ICP the sample sequence.
    $ python run.py -o /tmp/sample_output

    # Run point-to-plane ICP on a sequence in another directory (frame 100-150).
    $ python run.py --sequence /tmp/carla-town05 -o /tmp/sample_output \
        -s 100 -e 150 -m point_to_plane
"""

import rclpy
from rclpy.node import Node
import argparse
import os
import sys
import traceback
from os.path import basename, dirname, isdir, join, realpath
import numpy as np
import open3d as o3d


import registration
import utils

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import TransformStamped, PoseStamped
from tf2_ros import TransformBroadcaster
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

import tf_transformations as tf

class DopplerICPRealtime(Node):
    def __init__(self, args):
        super().__init__('realtime_doppler_icp_node')
        self.args = args
        self.params = vars(args)
        self.icp_method = registration.doppler_icp if args.method == 'doppler' else registration.point_to_plane_icp

        self.prev_pcd = None
        self.prev_pose = np.eye(4)
        self.results = {
            'poses': [np.eye(4)],
            'timestamps': [],
            'convergence': [],
            'iterations': []
        }

        self.params['T_V_to_S'] = np.eye(4)
        self.params['period'] = 0.1  # 실시간 주기 (수정 가능)

        self.odom_pub = self.create_publisher(Odometry, '/odom', 1)
        self.path_pub = self.create_publisher(Path, '/trajectory', 10)
        self.tf_broadcaster = TransformBroadcaster(self)


        qos_profile = QoSProfile(
                        history=QoSHistoryPolicy.KEEP_LAST,
                        depth=1,
                        reliability=QoSReliabilityPolicy.BEST_EFFORT,
                        durability=QoSDurabilityPolicy.VOLATILE
                        )
        self.create_subscription(PointCloud2, "/aeva/pointcloud", self.callback, qos_profile)

        self.map_pcd = o3d.geometry.PointCloud()
        self.map_pub = self.create_publisher(PointCloud2, '/map', 1)
        self.path_msg = Path()
        self.path_msg.header.frame_id = "odom"

    def pointcloud2_to_o3d(self, msg):
        raw_points = np.array(list(point_cloud2.read_points(msg, field_names=("x", "y", "z", "velocity"), skip_nans=True)), dtype=[
        ('x', np.float32), ('y', np.float32), ('z', np.float32), ('velocity', np.float32)
        ])

        if raw_points.shape[0] == 0:
            self.get_logger().warn("PointCloud is empty.")
            return None

        xyz_points = np.zeros((raw_points.shape[0], 3), dtype=np.float32)
        xyz_points[:, 0] = raw_points['x']     # structured array -> (N, 3) Numpy array로 변경 
        xyz_points[:, 1] = raw_points['y']
        xyz_points[:, 2] = raw_points['z']

        doppler_values = raw_points['velocity'] # doppler value 추가 

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_points) 
        pcd.dopplers = o3d.utility.DoubleVector(doppler_values.tolist())
        
        return pcd

    def callback(self, msg):
        target = self.pointcloud2_to_o3d(msg)
        if self.prev_pcd is None:
            self.prev_pcd = target
            stamp = msg.header.stamp
            time_in_sec = stamp.sec + stamp.nanosec * 1e-9
            self.results['timestamps'].append(time_in_sec)
            return
        
        
        init_transform = self.results['poses'][-1] if self.args.seed else np.eye(4)
        
        print("cccc")   
        try:
            result = self.icp_method(self.prev_pcd, target, self.params, init_transform)
            print("aaaa")

        except Exception as e:
            rclpy.logwarn("ICP Failed: %s" % e)
            return
        
        T_rel = np.linalg.inv(result.transformation)
        curr_pose = self.results['poses'][-1] @ T_rel
        
        # 🔧 여기에서 바로 정규 직교화
        R, t = curr_pose[:3, :3], curr_pose[:3, 3]
        u, _, vh = np.linalg.svd(R)
        R_orthogonal = u @ vh
        curr_pose[:3, :3] = R_orthogonal
        curr_pose[:3, 3] = t  # 위치는 그대로 유지

        self.results['poses'].append(curr_pose)

        #target point cloud 다운샘플링 후 transform
        target_down = target.voxel_down_sample(voxel_size=0.01)
        target_in_map = o3d.geometry.PointCloud(target)
        target_in_map.transform(curr_pose)
        
        #누적
        self.map_pcd += target_in_map

        if len(self.map_pcd.points) > 500000:
            self.map_pcd = self.map_pcd.voxel_down_sample(voxel_size=0.1)
            
        self.publish_map(self.map_pcd, msg.header)

        self.prev_pcd = target

        self.publish_odometry(curr_pose)

    def publish_map(self, pcd, header):
        points = np.asarray(pcd.points)
        map_msg = point_cloud2.create_cloud_xyz32(header, points)
        map_msg.header.frame_id = "map"
        self.map_pub.publish(map_msg)

    def publish_odometry(self, pose):

        trans = pose[:3, 3]
        quat = tf.quaternion_from_matrix(pose)
        
        now = self.get_clock().now().to_msg()

        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = "odom"
        odom.child_frame_id = "aeva_frame"

        odom.pose.pose.position.x = trans[0]
        odom.pose.pose.position.y = trans[1]
        odom.pose.pose.position.z = trans[2]
        odom.pose.pose.orientation.x = quat[0]
        odom.pose.pose.orientation.y = quat[1]
        odom.pose.pose.orientation.z = quat[2]
        odom.pose.pose.orientation.w = quat[3]

        self.odom_pub.publish(odom)
        
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = "odom"
        t.child_frame_id = "aeva_frame"
        t.transform.translation.x = trans[0]
        t.transform.translation.y = trans[1]
        t.transform.translation.z = trans[2]
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        self.tf_broadcaster.sendTransform(t)

        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = now
        pose_stamped.header.frame_id = "odom"
        pose_stamped.pose = odom.pose.pose

        
        self.path_msg.header.stamp = now
        self.path_msg.poses.append(pose_stamped)
        self.path_pub.publish(self.path_msg)

        




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', '-s', type=int, default=0,
                        help='Start frame index (inclusive)')
    parser.add_argument('--end', '-e', type=int, default=-1,
                        help='End frame index (inclusive)')
    parser.add_argument('--gui', action='store_true',
                        help='Shows the Open3D GUI after each registration')

    parser.add_argument('--method', '-m', default='doppler',
                        help='Registration method to use',
                        choices=['doppler', 'point_to_plane'])
    parser.add_argument('--seed', action='store_true',
                        help='Seed ICP using the previous pose estimate')
    parser.add_argument('--convergence_thresh', type=float, default=1e-5,
                        help='Convergence threshold for the registration'
                             ' algorithm. Higher the value, faster the'
                             ' convergence and lower the pose accuracy.')
    parser.add_argument('--max_iters', type=int, default=100,
                        help='Max iterations for the registration algorithm')
    parser.add_argument('--max_corr_distance', type=float, default=0.3,
                        help='Maximum correspondence points-pair distance (m)')
    parser.add_argument('--downsample_factor', type=int, default=2,
                        help='Factor to uniformly downsample the points by')
    parser.add_argument('--normals_radius', type=float, default=10.0,
                        help='Search radius (m) used in normal estimation')
    parser.add_argument('--normals_max_nn', type=int, default=30,
                        help='Max neighbors used in normal estimation search')

    parser.add_argument('--lambda_doppler', type=float, default=0.01,
                        help='Factor that weighs the Doppler residual term in'
                             ' the overall DICP objective. Setting a value of'
                             ' 0 is equivalent to point-to-plane ICP.')
    parser.add_argument('--reject_outliers', action='store_true',
                        help='Enable dynamic point outlier rejection')
    parser.add_argument('--outlier_thresh', type=float, default=2.0,
                        help='Error threshold (m/s) to reject dynamic outliers')
    parser.add_argument('--rejection_min_iters', type=int, default=2,
                        help='Number of iterations of ICP after which dynamic'
                             ' point outlier rejection is enabled')
    parser.add_argument('--geometric_min_iters', type=int, default=0,
                        help='Number of iterations of ICP after which robust'
                             ' loss for the geometric term is enabled')
    parser.add_argument('--doppler_min_iters', type=int, default=2,
                        help='Number of iterations of ICP after which robust'
                             ' loss for the Doppler term is enabled')
    parser.add_argument('--geometric_k', type=float, default=0.5,
                        help='Scale factor for the geometric robust loss')
    parser.add_argument('--doppler_k', type=float, default=0.2,
                        help='Scale factor for the Doppler robust loss')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rclpy.init()
    node = DopplerICPRealtime(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
