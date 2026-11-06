"""
Enhanced Collision Detection Demos - v2
Realistic physics, impact visualization, and persistent crash state
"""

import cv2
import numpy as np
import time
from src.spatial_engine import SpatialEngine
from src.visualizer import ParkingVisualizer


class Vehicle:
    """Realistic vehicle with physics"""
    def __init__(self, vehicle_id: int, x: float, y: float, vx: float = 0.0, vy: float = 0.0):
        self.id = vehicle_id
        self.x = x
        self.y = y
        self.vx = vx  # velocity x
        self.vy = vy  # velocity y
        self.crashed = False
        self.crash_time = 0
        self.speed = 0

    def update(self, dt: float, friction: float = 0.95):
        """Update position with physics"""
        if not self.crashed:
            # Apply velocity
            self.x += self.vx * dt
            self.y += self.vy * dt

            # Apply friction
            self.vx *= friction
            self.vy *= friction

            # Calculate speed
            self.speed = np.sqrt(self.vx**2 + self.vy**2)

    def stop(self):
        """Emergency stop"""
        self.vx *= 0.8  # Rapid deceleration
        self.vy *= 0.8

    def crash(self):
        """Set crashed state"""
        self.crashed = True
        self.crash_time = time.time()
        self.vx = 0
        self.vy = 0

    def get_bbox(self):
        """Get bounding box"""
        w, h = 80, 110
        return (
            int(self.x - w//2), int(self.y - h//2),
            int(self.x + w//2), int(self.y + h//2)
        )


def run_near_miss_demo_v2():
    """
    ENHANCED NEAR-MISS DEMO
    - Realistic vehicle physics
    - Emergency braking (not just stopping)
    - Velocity vectors displayed
    - Collision avoidance animation
    """
    print("\n" + "="*70)
    print("DEMO v2: NEAR-MISS WITH REALISTIC PHYSICS")
    print("="*70)

    engine = SpatialEngine()
    visualizer = ParkingVisualizer(engine.slots)

    width, height = 1280, 720
    fps = 30.0
    dt = 1.0 / fps

    # Vehicles with realistic physics
    car_101 = Vehicle(101, 190.0, 190.0, vx=0, vy=0)  # Starting parked
    car_105 = Vehicle(105, 60.0, 360.0, vx=4.5, vy=0)  # Moving through lane

    cv2.namedWindow("DEMO v2: NEAR-MISS (Realistic Physics)", cv2.WINDOW_AUTOSIZE)

    frame_idx = 0
    emergency_brake_triggered = False

    while True:
        frame_idx += 1
        bg = np.full((height, width, 3), (45, 45, 45), dtype=np.uint8)

        # Lane markings
        for x in range(0, width, 60):
            cv2.line(bg, (x, 360), (x + 30, 360), (0, 255, 255), 2)

        # Update physics
        car_105.update(dt)

        # Car 101 starts reversing at frame 15
        if frame_idx >= 15 and frame_idx < 40 and not car_101.crashed:
            car_101.vy = 3.5  # Reverse velocity

        # Calculate distance
        dist = np.sqrt((car_105.x - car_101.x)**2 + (car_105.y - car_101.y)**2)

        # Emergency braking at 130px distance
        if dist < 130 and not emergency_brake_triggered and not car_101.crashed:
            emergency_brake_triggered = True
            print(f"[ALERT] Frame {frame_idx}: NEAR-MISS DETECTED! Distance: {dist:.1f}px")
            print(f"        Car #101 emergency braking at {car_101.y:.1f}")

        # Apply braking
        if emergency_brake_triggered:
            car_101.stop()  # Rapid deceleration
            car_105.vx *= 0.98  # Slight braking

        # Create tracked vehicles for engine
        tracked = []
        for car in [car_101, car_105]:
            bbox = car.get_bbox()
            tracked.append({
                "id": car.id,
                "bbox": bbox,
                "conf": 0.95,
                "cls_name": "car"
            })

        # Draw vehicles
        for car, tracked_info in zip([car_101, car_105], tracked):
            x1, y1, x2, y2 = tracked_info["bbox"]

            # Vehicle color (red if emergency braking)
            color = (0, 0, 255) if emergency_brake_triggered else (80, 80, 200)
            cv2.rectangle(bg, (x1, y1), (x2, y2), color, -1)

            # Vehicle ID
            cv2.putText(bg, f"Car #{car.id}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Speed indicator
            speed_text = f"{car.speed:.1f} px/s"
            cv2.putText(bg, speed_text, (x1, y2+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if car.speed < 2 else (255, 0, 0), 1)

            # Velocity vector
            if car.speed > 0.5:
                end_x = int(car.x + car.vx * 5)
                end_y = int(car.y + car.vy * 5)
                cv2.arrowedLine(bg, (int(car.x), int(car.y)), (end_x, end_y),
                               (0, 255, 255), 2, tipLength=0.3)

        # Distance indicator
        cv2.putText(bg, f"Distance: {dist:.1f}px", (50, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0) if dist < 150 else (255, 255, 255), 2)

        # Status bar
        status = "NEAR-MISS ALERT!" if dist < 130 else "APPROACHING..."
        status_color = (0, 0, 255) if dist < 130 else (255, 165, 0)

        cv2.rectangle(bg, (0, height - 60), (width, height), (25, 25, 25), -1)
        cv2.putText(bg, f"NEAR-MISS SCENARIO | Status: {status} | Frame: {frame_idx}",
                   (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        cv2.imshow("DEMO v2: NEAR-MISS (Realistic Physics)", bg)

        if cv2.waitKey(int(1000/fps)) & 0xFF in [ord('q'), 27]:
            break

    cv2.destroyAllWindows()
    print("\n✅ Near-Miss demo complete!\n")


def run_collision_demo_v2():
    """
    ENHANCED COLLISION DEMO
    - Realistic vehicle physics
    - Actual impact/crash
    - Vehicles stay crashed (persistent state)
    - Damage visualization
    - Impact force calculation
    """
    print("\n" + "="*70)
    print("DEMO v2: ACTUAL COLLISION WITH PERSISTENT CRASH STATE")
    print("="*70)

    engine = SpatialEngine()
    visualizer = ParkingVisualizer(engine.slots)

    width, height = 1280, 720
    fps = 30.0
    dt = 1.0 / fps

    # Vehicles - no braking!
    car_101 = Vehicle(101, 190.0, 190.0, vx=0, vy=0)
    car_105 = Vehicle(105, 60.0, 360.0, vx=4.0, vy=0)

    cv2.namedWindow("DEMO v2: COLLISION (Persistent Crash)", cv2.WINDOW_AUTOSIZE)

    frame_idx = 0
    collision_occurred = False
    collision_frame = 0
    impact_force = 0

    while True:
        frame_idx += 1
        bg = np.full((height, width, 3), (45, 45, 45), dtype=np.uint8)

        # Lane markings
        for x in range(0, width, 60):
            cv2.line(bg, (x, 360), (x + 30, 360), (0, 255, 255), 2)

        # Update physics
        if not collision_occurred:
            car_105.update(dt)

            # Car 101 reverses into lane (no braking!)
            if frame_idx >= 15 and car_101.y < 360:
                car_101.vy = 4.0  # Faster reversal
                car_101.update(dt)

            # Check for collision
            dist = np.sqrt((car_105.x - car_101.x)**2 + (car_105.y - car_101.y)**2)

            if dist < 70 and not collision_occurred:
                collision_occurred = True
                collision_frame = frame_idx
                impact_force = abs(car_105.vx) + abs(car_101.vy)

                car_101.crash()
                car_105.crash()

                print(f"[IMPACT] Frame {frame_idx}: COLLISION DETECTED!")
                print(f"         Impact Force: {impact_force:.2f} units")
                print(f"         Car #101 velocity: {car_101.vy:.2f}")
                print(f"         Car #105 velocity: {car_105.vx:.2f}")
                print(f"         Distance at impact: {dist:.1f}px")

        else:
            # Vehicles stay crashed (don't move)
            pass

        # Draw vehicles
        for car in [car_101, car_105]:
            bbox = car.get_bbox()
            x1, y1, x2, y2 = bbox

            # Color based on state
            if car.crashed:
                color = (0, 0, 255)  # Red for crashed
            else:
                color = (80, 80, 200)  # Blue for normal

            cv2.rectangle(bg, (x1, y1), (x2, y2), color, -1)

            # Damage cracks if crashed
            if car.crashed:
                for _ in range(3):
                    crack_x = np.random.randint(x1, x2)
                    crack_y = np.random.randint(y1, y2)
                    cv2.line(bg, (crack_x, crack_y),
                            (crack_x + np.random.randint(-20, 20),
                             crack_y + np.random.randint(-20, 20)),
                            (0, 165, 255), 2)

            # Vehicle ID
            cv2.putText(bg, f"Car #{car.id}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Status
            status = "CRASHED" if car.crashed else f"{car.speed:.1f} px/s"
            status_color = (0, 0, 255) if car.crashed else (0, 255, 0)
            cv2.putText(bg, status, (x1, y2+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

        # Impact visualization - expanding circles
        if collision_occurred:
            time_since_impact = frame_idx - collision_frame
            pulse = int((time_since_impact % 10) * 4)

            impact_x = int((car_101.x + car_105.x) / 2)
            impact_y = int((car_101.y + car_105.y) / 2)

            # Impact rings
            cv2.circle(bg, (impact_x, impact_y), 30 + pulse, (0, 0, 255), 3)
            cv2.circle(bg, (impact_x, impact_y), 60 + pulse, (0, 165, 255), 2)

            # Impact text
            cv2.putText(bg, f"IMPACT FORCE: {impact_force:.1f}", (impact_x-80, impact_y-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Status bar
        if collision_occurred:
            status = "🚨 COLLISION! VEHICLES IMMOBILIZED 🚨"
            status_color = (0, 0, 255)
        else:
            status = "⚠️ VEHICLES APPROACHING ⚠️"
            status_color = (0, 165, 255)

        cv2.rectangle(bg, (0, height - 60), (width, height), (25, 25, 25), -1)
        cv2.putText(bg, f"COLLISION SCENARIO | {status} | Frame: {frame_idx}",
                   (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

        cv2.imshow("DEMO v2: COLLISION (Persistent Crash)", bg)

        if cv2.waitKey(int(1000/fps)) & 0xFF in [ord('q'), 27]:
            break

    cv2.destroyAllWindows()
    print("\n✅ Collision demo complete!\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced collision/near-miss demos v2")
    parser.add_argument("--mode", type=str, default="near_miss",
                       choices=["near_miss", "collision", "both"],
                       help="Demo mode")

    args = parser.parse_args()

    if args.mode in ["near_miss", "both"]:
        run_near_miss_demo_v2()

    if args.mode in ["collision", "both"]:
        run_collision_demo_v2()

    print("="*70)
    print("✅ All demos complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
