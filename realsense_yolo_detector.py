import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from cv_bridge import CvBridge
import time

class RealSenseYOLONode(Node):
    def __init__(self):
        super().__init__('realsense_yolo_node')
        
        # RealSenseの設定
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # デバイスの確認と設定
        try:
            # 利用可能なデバイスの検索
            ctx = rs.context()
            devices = ctx.query_devices()
            if len(devices) == 0:
                self.get_logger().error('No RealSense devices found!')
                return
                
            # 最初のデバイスのシリアル番号を取得
            device = devices[0]
            self.get_logger().info(f'Found device: {device.get_info(rs.camera_info.name)}')
            
            # ストリームの設定
            self.config.enable_device(device.get_info(rs.camera_info.serial_number))
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            
            # 高度な設定
            self.align = rs.align(rs.stream.color)
            
            # パイプラインの開始
            pipeline_profile = self.pipeline.start(self.config)
            
            # ストリーミング開始後のウォームアップ待機
            self.get_logger().info('Waiting for device to warm-up...')
            time.sleep(2)
            
            # デバイスの設定
            device = pipeline_profile.get_device()
            depth_sensor = device.first_depth_sensor()
            depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
            
            # Advanced modeの設定（利用可能な場合）
            if device.supports(rs.camera_info.product_line):
                advanced_mode = rs.rs400_advanced_mode(device)
                if not advanced_mode.is_enabled():
                    advanced_mode.toggle_advanced_mode(True)
                    time.sleep(1)  # 設定の適用を待機
                
        except Exception as e:
            self.get_logger().error(f'Error initializing RealSense: {str(e)}')
            return
            
        # YOLOモデルの読み込み
        try:
            self.model = YOLO('yolov8n.pt')
        except Exception as e:
            self.get_logger().error(f'Error loading YOLO model: {str(e)}')
            return
            
        # 結果表示用のウィンドウ名
        self.window_name = 'RealSense YOLO Detection'
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        
        # タイマーの設定（20Hz - より安定した動作のため）
        self.timer = self.create_timer(1/20, self.timer_callback)
        self.get_logger().info('RealSense YOLO Node has been started')
        
        # フレーム取得失敗回数のカウンター
        self.frame_failure_count = 0
        self.MAX_FAILURES = 5
        
    def timer_callback(self):
        try:
            # フレームの取得（タイムアウト設定を3秒に延長）
            frames = self.pipeline.wait_for_frames(timeout_ms=3000)
            if not frames:
                self.frame_failure_count += 1
                if self.frame_failure_count >= self.MAX_FAILURES:
                    self.get_logger().error('Multiple frame acquisition failures. Restarting pipeline...')
                    self.restart_pipeline()
                return
                
            self.frame_failure_count = 0  # 成功したらカウンターをリセット
            
            # フレームの位置合わせ
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                self.get_logger().warn('Invalid frames received')
                return
                
            # カラー画像をnumpy配列に変換
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # YOLO検出の実行
            results = self.model(color_image)
            
            # 検出結果の描画
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = self.model.names[cls]
                    
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    depth_value = depth_frame.get_distance(center_x, center_y)
                    
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f'{class_name} {conf:.2f} {depth_value:.2f}m'
                    cv2.putText(color_image, label, (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 結果の表示
            cv2.imshow(self.window_name, color_image)
            key = cv2.waitKey(1)
            
            if key & 0xFF == ord('q'):
                self.destroy_node()
                rclpy.shutdown()
                
        except Exception as e:
            self.get_logger().error(f'Error in timer callback: {str(e)}')
            self.frame_failure_count += 1
            if self.frame_failure_count >= self.MAX_FAILURES:
                self.restart_pipeline()
                
    def restart_pipeline(self):
        """パイプラインを再起動する"""
        try:
            self.pipeline.stop()
            time.sleep(1)
            self.pipeline.start(self.config)
            time.sleep(2)  # ウォームアップ待機
            self.frame_failure_count = 0
            self.get_logger().info('Pipeline successfully restarted')
        except Exception as e:
            self.get_logger().error(f'Failed to restart pipeline: {str(e)}')
            
    def __del__(self):
        # リソースの解放
        try:
            self.pipeline.stop()
            cv2.destroyAllWindows()
        except:
            pass

def main(args=None):
    rclpy.init(args=args)
    node = RealSenseYOLONode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node:
            node.__del__()
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()