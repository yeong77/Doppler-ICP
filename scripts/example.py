#!/usr/bin/env python3
import rospy
import numpy as np
import open3d as o3d
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from nav_msgs.msg import Odometry
import tf
import registration
import utils
import argparse

class DopplerICPRealtime:
    def __init__(self, args):
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

        self.odom_pub = rospy.Publisher('/odom', Odometry, queue_size=10)
        self.tf_broadcaster = tf.TransformBroadcaster()

        rospy.Subscriber("/points_raw", PointCloud2, self.callback)

    def pointcloud2_to_o3d(self, msg):
        points = np.array([p[:3] for p in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd

    def callback(self, msg):
        target = self.pointcloud2_to_o3d(msg)

        if self.prev_pcd is None:
            self.prev_pcd = target
            self.results['timestamps'].append(msg.header.stamp.to_sec())
            return

        init_transform = self.results['poses'][-1] if self.args.seed else np.eye(4)

        try:
            result = self.icp_method(self.prev_pcd, target, self.params, init_transform)
        except Exception as e:
            rospy.logwarn("ICP Failed: %s" % e)
            return

        T_rel = np.linalg.inv(result.transformation)
        curr_pose = self.results['poses'][-1] @ T_rel

        self.results['poses'].append(curr_pose)
        self.results['timestamps'].append(msg.header.stamp.to_sec())
        self.results['convergence'].append(result.converged)
        self.results['iterations'].append(result.num_iterations)

        self.prev_pcd = target

        self.publish_odometry(curr_pose)

    def publish_odometry(self, pose):
        trans = pose[:3, 3]
        quat = tf.transformations.quaternion_from_matrix(pose)

        odom = Odometry()
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"

        odom.pose.pose.position.x = trans[0]
        odom.pose.pose.position.y = trans[1]
        odom.pose.pose.position.z = trans[2]
        odom.pose.pose.orientation.x = quat[0]
        odom.pose.pose.orientation.y = quat[1]
        odom.pose.pose.orientation.z = quat[2]
        odom.pose.pose.orientation.w = quat[3]

        self.odom_pub.publish(odom)
        self.tf_broadcaster.sendTransform(trans, quat, rospy.Time.now(), "base_link", "odom")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', '-m', default='doppler', choices=['doppler', 'point_to_plane'])
    parser.add_argument('--seed', action='store_true')
    parser.add_argument('--convergence_thresh', type=float, default=1e-5)
    parser.add_argument('--max_iters', type=int, default=100)
    parser.add_argument('--max_corr_distance', type=float, default=0.3)
    parser.add_argument('--downsample_factor', type=int, default=2)
    parser.add_argument('--normals_radius', type=float, default=10.0)
    parser.add_argument('--normals_max_nn', type=int, default=30)
    parser.add_argument('--lambda_doppler', type=float, default=0.01)
    parser.add_argument('--reject_outliers', action='store_true')
    parser.add_argument('--outlier_thresh', type=float, default=2.0)
    parser.add_argument('--rejection_min_iters', type=int, default=2)
    parser.add_argument('--geometric_min_iters', type=int, default=0)
    parser.add_argument('--doppler_min_iters', type=int, default=2)
    parser.add_argument('--geometric_k', type=float, default=0.5)
    parser.add_argument('--doppler_k', type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    rospy.init_node("realtime_doppler_icp_node")
    args = parse_args()
    DopplerICPRealtime(args)
    rospy.loginfo("Realtime Doppler ICP started")
    rospy.spin()
