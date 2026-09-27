#!/usr/bin/env python3
"""Starter node for the UEH CRC 2026 simulation round - lane keeping version."""

import math
import signal
import time
import numpy as np

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, LaserScan

try:
    from cv_bridge import CvBridge
    HAVE_CV = True
except ImportError:
    HAVE_CV = False

try:
    import cv2
except ImportError:
    cv2 = None


class Starter(Node):

    def __init__(self):
        super().__init__('crc_starter')

        self.declare_parameter('max_speed', 0.20)
        self.declare_parameter('max_turn', 1.0)
        self.declare_parameter('stop_distance', 0.35)
        self.declare_parameter('rate', 20.0)

        self.max_speed = self.get_parameter('max_speed').value
        self.max_turn = self.get_parameter('max_turn').value
        self.stop_distance = self.get_parameter('stop_distance').value
        rate = self.get_parameter('rate').value

        self.image = None
        self.scan = None
        self.x = self.y = self.yaw = 0.0
        self._last_log = {}

        self.bridge = CvBridge() if HAVE_CV else None
        if not HAVE_CV:
            self.get_logger().warn(
                'cv_bridge not found, self.image will stay None. '
                'apt install ros-humble-cv-bridge python3-opencv')

        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Image, '/camera/image_raw',
                                  self.on_image, qos_profile_sensor_data)
        self.create_subscription(LaserScan, '/scan',
                                  self.on_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.on_odom, 10)
        # CÁC BIẾN TRẠNG THÁI CHO LOGIC BIỂN BÁO
        self.state = 'NORMAL'
        self.wait_start_time = 0.0
        self.ignore_stop_until = 0.0 

        self.create_timer(1.0 / rate, self.tick)
        self.get_logger().info(
            'ready | max_speed=%.2f m/s | %.0f Hz' % (self.max_speed, rate))

    def on_image(self, msg):
        if self.bridge is None:
            return
        try:
            self.image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().warn('image conversion failed: %s' % e)

    def on_scan(self, msg):
        self.scan = msg

    def on_odom(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.x, self.y = p.x, p.y
        self.yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def range_at(self, angle_deg, width_deg=10.0):
        if self.scan is None or not self.scan.ranges:
            return float('inf')
        n = len(self.scan.ranges)
        best = float('inf')
        half = int(round(width_deg / 2.0))
        centre = int(round(angle_deg)) % 360
        for d in range(-half, half + 1):
            r = self.scan.ranges[int((centre + d) % 360 * n / 360)]
            if math.isfinite(r) and r > self.scan.range_min:
                best = min(best, r)
        return best

    def drive(self, v, w):
        msg = Twist()
        msg.linear.x = float(max(-self.max_speed, min(self.max_speed, v)))
        msg.angular.z = float(max(-self.max_turn, min(self.max_turn, w)))
        self.pub_cmd.publish(msg)

    def stop(self):
        self.pub_cmd.publish(Twist())

    def log_every(self, seconds, text):
        now = time.time()
        if now - self._last_log.get(text[:20], 0.0) >= seconds:
            self._last_log[text[:20]] = now
            self.get_logger().info(text)

    def tick(self):
        try:
            self.control()
        except Exception as e:
            self.get_logger().error('control() raised: %s' % e)
            self.stop()

    def control(self):
        if self.image is None or self.scan is None:
            self.stop()
            return

        if cv2 is None:
            self.log_every(5.0, 'cv2 not available')
            self.stop()
            return

        now = time.time()
        self.log_every(1.0, f"=== TRẠNG THÁI HIỆN TẠI: {self.state} ===")
        h, w = self.image.shape[:2]

        # Chuyển ảnh sang HSV
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        front_dist = self.range_at(0, width_deg=10.0)
        self.log_every(1.0, f"=== STATE: {self.state} | Lidar trước: {front_dist:.2f}m ===")
        
        mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([179, 255, 255]))
        red_mask = mask1 + mask2

        is_red_sign = False   
        is_red_light = False  

        contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in contours_red:
            x, y, w_box, h_box = cv2.boundingRect(c)
            box_area = w_box * h_box
            
            if box_area < 100:
                continue

            if y < int(h * 0.4) and 80 <= box_area < 300:
                is_red_light = True
                self.log_every(0.5, f"🔴 CHỐT ĐÈN ĐỎ | Area={box_area}")
                
            elif box_area >= 250:
                aspect_ratio = float(w_box) / h_box
                
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                
                is_stop_shape = (7 <= len(approx) <= 10) or (box_area > 1000)
                is_square_like = 0.8 <= aspect_ratio <= 1.2
                
                if is_stop_shape and is_square_like:
                    if front_dist <= 1.2 or box_area > 1000:
                        is_red_sign = True
                        self.log_every(0.5, f"🛑 CHỐT BIỂN STOP | Cạnh: {len(approx)} | Aspect: {aspect_ratio:.2f} | Area: {box_area}")
                        break
        green_light_mask = cv2.inRange(hsv, np.array([35, 10, 200]), np.array([90, 255, 255])) 

        is_green_light = False

        contours_green_light, _ = cv2.findContours(green_light_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours_green_light:
            x, y, w_box, h_box = cv2.boundingRect(c)
            box_area = w_box * h_box
            if 250 <= box_area < 600 and y < int(h * 0.6):
                is_green_light = True
                self.log_every(0.5, f"🟩 ĐÃ BẮT ĐƯỢC ĐÈN XANH | Diện tích: {box_area}")
                break

        purple_mask = cv2.inRange(hsv, np.array([125, 50, 50]), np.array([160, 255, 255]))
        is_purple_visible = cv2.countNonZero(purple_mask) > 400

        green_mask = cv2.inRange(hsv, np.array([40, 100, 100]), np.array([85, 255, 255]))
        is_green_far = False
        is_green_close = False
        contours_green, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours_green:
            x, y, w_box, h_box = cv2.boundingRect(c)
            box_area = w_box * h_box
            if box_area > 800: # Thấy biển từ xa
                is_green_far = True
                # Nếu Lidar báo cách vật <= 1.2m HOẶC biển phình to trong camera -> Đã tới gần!
                if front_dist <= 1.2 or box_area > 1500:
                    is_green_close = True
                break

        if self.state == 'WAITING_RED':
            # Lấy cờ đánh dấu từ bước trước để biết đang chờ Biển hay chờ Đèn
            if getattr(self, 'stopped_by_sign', False):
                # Chờ Biển STOP: Dùng logic 2 giây cũ của bạn
                if now - self.wait_start_time < 2.0:
                    self.stop()
                    return
                else:
                    self.state = 'NORMAL'
                    self.ignore_stop_until = now + 4.0
                    self.get_logger().info("Đã chờ xong biển STOP 2s. Đi tiếp!")
            else:
                # Chờ Đèn Đỏ/Vàng: Phanh cứng cho đến khi is_green_light = True
                if is_green_light:
                    self.state = 'NORMAL'
                    self.ignore_stop_until = now + 4.0
                    self.get_logger().info("🟢 ĐÈN XANH BẬT! Tiến lên!")
                else:
                    self.stop()
                    return

        if self.state == 'WAITING_PURPLE':
            if is_purple_visible:
                self.stop()
                return
            else:
                self.state = 'NORMAL'
                self.ignore_stop_until = now + 4.0 
                self.get_logger().info("Vật màu Tím đã đi qua. Tiếp tục hành trình!")
        
        if self.state == 'NORMAL' and now > self.ignore_stop_until:
            if is_green_far:
                self.state = 'APPROACHING_GREEN'
                self.get_logger().info("Thấy biển Xanh từ xa, từ từ tiến lại gần...")
            elif is_purple_visible:
                self.state = 'FINDING_LINE_PURPLE'
            elif is_red_sign or is_red_light:
                self.state = 'FINDING_LINE_RED'
                self.stopped_by_sign = is_red_sign 
                self.get_logger().info("Bất thình lình Đỏ/Biển! Bắt đầu rình vạch ngang để dừng.")
                
        elif self.state == 'APPROACHING_GREEN':
            # Khi đang tiến lại gần, đợi đến khi cách đúng 1m (is_green_close = True) mới rẽ
            if is_green_close:
                self.state = 'SHIFTING_LEFT'
                self.maneuver_start_time = now
                self.get_logger().info("🟩 Cách biển 1m! Bắt đầu bẻ lái lách trái.")
            elif not is_green_far:
                self.state = 'NORMAL' # Nếu tự nhiên mất dấu biển thì quay lại chạy bình thường
                
        if self.state in ['FINDING_LINE_RED', 'FINDING_LINE_PURPLE', 'APPROACHING_GREEN']:
            if self.state == 'FINDING_LINE_PURPLE' and not is_purple_visible:
                self.state = 'NORMAL'
            else:
                gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                _, thresh_line = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
                
                if self.state == 'FINDING_LINE_RED':
                    if getattr(self, 'stopped_by_sign', False):
                        # 1. Nếu là BIỂN STOP: Quét sát gầm xe để dẫm đúng vạch mới dừng (~0.5m)
                        stop_roi = thresh_line[h-40:h, int(w*0.3):int(w*0.7)]
                    else:
                        # 2. Nếu là ĐÈN ĐỎ/VÀNG: Ngẩng lên quét xa để phanh sớm (~1m - 1.2m)
                        stop_roi = thresh_line[h-180:h-120, int(w*0.3):int(w*0.7)]
                else:
                    # Các vạch bình thường khác: Quét sát gầm
                    stop_roi = thresh_line[h-40:h, int(w*0.3):int(w*0.7)]
                
                white_pixels = cv2.countNonZero(stop_roi)
                roi_area = stop_roi.shape[0] * stop_roi.shape[1]
                
                # Nếu lượng điểm trắng chiếm hơn 15% vùng nhìn -> Đạp vạch!
                if now > getattr(self, 'ignore_line_until', 0.0) and white_pixels > roi_area * 0.15:
                    # CHỐT DỪNG Ở ĐÂY CHO ĐÈN ĐỎ & BIỂN ĐỎ
                    self.stop()
                    if self.state == 'FINDING_LINE_RED':
                        self.state = 'WAITING_RED'
                        self.wait_start_time = now
                        self.get_logger().info("Đã đạp vạch STOP. PHANH LẠI!")
                    else:
                        self.state = 'WAITING_PURPLE'
                        self.get_logger().info("Đã dừng trước chướng ngại vật màu tím.")
                    return

        if self.state == 'SHIFTING_LEFT':
            elapsed = now - self.maneuver_start_time
            
            if elapsed < 1.9: 
                self.drive(2.0, 0.4) 
                return 
                
            else:
                self.state = 'HW_LANE_LEFT'
                self.get_logger().info("Đã lách sang trái! Bắt đầu bám lề trái.")
                
        elif self.state == 'HW_LANE_LEFT':
            pass
        bottom = self.image[int(h * 2 / 3):h, :]
        gray_lane = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)
        _, mask_lane = cv2.threshold(gray_lane, 180, 255, cv2.THRESH_BINARY)

        left_third = mask_lane[:, 0:int(w / 3)]
        right_half = mask_lane[:, int(w * 0.5):w]

        def get_lane_center(roi):
            contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_contours = []
            for c in contours:
                if cv2.contourArea(c) > 15:
                    x, y, w_box, h_box = cv2.boundingRect(c)
                    if w_box > h_box * 2.5: continue
                    valid_contours.append(c)
            if not valid_contours: return None
            closest = max(valid_contours, key=lambda c: np.max(c[:, :, 1]))
            return int(np.max(closest[:, :, 0]))

        def get_largest_lane_center(roi):
            contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_contours = []
            for c in contours:
                if cv2.contourArea(c) > 15:
                    x, y, w_box, h_box = cv2.boundingRect(c)
                    if w_box > h_box * 2.5: continue
                    valid_contours.append(c)
            if not valid_contours: return None
            largest = max(valid_contours, key=cv2.contourArea) # Chọn diện tích to nhất
            return int(np.max(largest[:, :, 0]))

        cx_left = get_largest_lane_center(left_third)
        
        # Làn phải giữ nguyên logic cũ
        cx_right = get_lane_center(right_half)
        if cx_right is not None: 
            cx_right += int(w * 0.5)

        target_found = False
        cx_target = 0
        pixel_offset = 0

        # NẾU ĐANG VƯỢT THÌ BÁM TRÁI, CÒN LẠI BÁM PHẢI TẤT CẢ
        if self.state == 'HW_LANE_LEFT':
            if cx_left is not None:
                cx_target = cx_left
                target_found = True
                pixel_offset = -200
                lane_name = "LỀ TRÁI"
        else:
            if cx_right is not None:
                cx_target = cx_right
                target_found = True
                pixel_offset = 275
                lane_name = "LỀ PHẢI"

        if target_found:
            target_center = cx_target - pixel_offset
            error = target_center - (w // 2)

            kp = 0.005  
            kd = 0.015  
            
            last_err = getattr(self, 'last_error', 0.0)
            derivative = error - last_err
            self.last_error = error

            with open('/tmp/lane_error_log.csv', 'a') as f:
                f.write(f"{now},{self.state},{error}\n")
            
            wz = -(kp * error + kd * derivative)
            wz = max(-1.0, min(1.0, wz))

            base_speed = 0.8 
            if self.state in ['FINDING_LINE_RED', 'FINDING_LINE_PURPLE', 'FINDING_FIRST_HW_LINE']:
                if front_dist < 3.0:
                    base_speed = 0.08  
                else:
                    base_speed = 0.35  
                    
            v_speed = base_speed - (abs(error) * 0.001)
            v_speed = max(0.06, v_speed) 
            
            self.last_wz = wz 
            self.drive(v_speed, wz)
            
            if self.state in ['TRANSITION_LEFT', 'TRANSITION_RIGHT']:
                self.log_every(0.5, f"ĐANG ÉP OFFSET {pixel_offset} | Neo vào: {lane_name} | Lái: {wz:.2f}")
            else:
                self.log_every(0.5, f"BÁM {lane_name} | Tâm: {cx_target} | Sai số: {error:.1f} | Lái: {wz:.2f}")
            
        else:
            self.last_error = 0.0
            v_speed = 0.08
            wz_fallback = getattr(self, 'last_wz', 0.0) * 0.9 
            self.drive(v_speed, wz_fallback)
            self.log_every(0.5, "MẤT VẠCH! Rà vô lăng...")
def catch_sigterm():
    stopping = {'now': False}
    signal.signal(signal.SIGTERM, lambda *_: stopping.update(now=True))
    return stopping


def spin(node, stopping):
    while rclpy.ok() and not stopping['now']:
        try:
            rclpy.spin_once(node, timeout_sec=0.1)
        except Exception:
            if stopping['now'] or not rclpy.ok():
                break
            raise


def main(args=None):
    rclpy.init(args=args)
    stopping = catch_sigterm()
    node = Starter()
    try:
        spin(node, stopping)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
