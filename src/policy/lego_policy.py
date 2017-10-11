from mqtt_master.master import Master
from policy.detector import Detector
from tracking.tracker import Tracker
from policy.avoid_object import AvoidObject
import time
import numpy as np
import cv2

class AssessResult:

    OBJECT_TARGET_BRICK = "brick"
    OBJECT_TARGET_BOX = "box"
    OBJECT_OBSTACLE = "obstacle"
    OBJECT_NONE = "none"

    def __init__(self, object, point=None, max_y=None):
        self.object = object
        self.point = point
        self.max_y = max_y

class Policy:

    STRAIGHT_THRESH_BRICK = 400
    STRAIGHT_THRESH_BOX = 330

    IGNORE_OBSTACLE_THRESH = 150

    BASELINE = 30
    FACTOR = 0.3

    THRES_R = lambda y: 320 + Policy.BASELINE + Policy.FACTOR * (480 - y)
    THRES_L = lambda y: 320 - Policy.BASELINE - Policy.FACTOR * (480 - y)

    MOVEMENT_STATE_LEFT = "move_left"
    MOVEMENT_STATE_RIGHT = "move_right"
    MOVEMENT_STATE_STRAIGHT = "move_straight"
    MOVEMENT_STATE_APPROACH = "move_approach"

    POLICY_STATE_SCANNING = "scanning"
    POLICY_STATE_ON_TARGET = "on_target"
    POLICY_STATE_APPROACHING = "approaching"
    POLICY_STATE_AVOID = "avoid"
    POLICY_STATE_PICK_UP_DURING_AVOIDANCE = "pick_up_during_avoidance"

    GRIPPER_STATE_EMPTY = "gripper_empty"
    GRIPPER_STATE_FULL = "gripper_full"

    AVOID_STRAIGHT_FOR = 10

    def __init__(self, mqtt_master, model, camera_front, camera_down):
        self.master = mqtt_master

        self.model = model

        self.cam_handler_front = camera_front
        self.cam_handler_down = camera_down

        self.detector = Detector(self.model)

        self.tracker = Tracker(self.model)
        self.tracker.reset()

        self.avoider = AvoidObject()
        self.avoid_start_time = None
        self.already_avoided = 0

        self._policy()

    def _policy(self):
        policy_state = Policy.POLICY_STATE_SCANNING
        gripper_state = Policy.GRIPPER_STATE_EMPTY
        cluster_id = -1
        while True:
            if policy_state == Policy.POLICY_STATE_SCANNING:
                print("scanning")
                if gripper_state == Policy.GRIPPER_STATE_EMPTY:
                    policy_state = self._scan_policy(assess_fn=self._assess_brick)

                else:
                    policy_state = self._scan_policy(assess_fn=lambda: self._assess_box(cluster_id + 1))
            elif policy_state == Policy.POLICY_STATE_ON_TARGET:
                if gripper_state == Policy.GRIPPER_STATE_EMPTY:
                    policy_state = self._go_to_target_policy(assess_fn=self._assess_brick,
                                                             straight_thresh=Policy.STRAIGHT_THRESH_BRICK)

                else:
                    policy_state = self._go_to_target_policy(assess_fn=lambda: self._assess_box(cluster_id + 1),
                                                             straight_thresh=Policy.STRAIGHT_THRESH_BOX)

            elif policy_state == Policy.POLICY_STATE_APPROACHING:
                if gripper_state == Policy.GRIPPER_STATE_EMPTY:
                    assert gripper_state == Policy.GRIPPER_STATE_EMPTY, "Wrong state - gripper not empty for pick up"
                    policy_state, cluster_id = self._pick_up_brick()
                    gripper_state = Policy.GRIPPER_STATE_FULL
                else:
                    assert gripper_state == Policy.GRIPPER_STATE_FULL, "Wrong state - gripper not full for release"
                    policy_state = self._put_brick_into_box()
                    gripper_state = Policy.GRIPPER_STATE_EMPTY

            elif policy_state == Policy.POLICY_STATE_AVOID:
                policy_state = self._avoid_policy(pick_up_during_avoidance=(gripper_state == Policy.GRIPPER_STATE_EMPTY))

            elif policy_state == Policy.POLICY_STATE_PICK_UP_DURING_AVOIDANCE:
                assert gripper_state == Policy.GRIPPER_STATE_EMPTY, "Wrong state - gripper not empty for pick up"
                policy_state, cluster_id = self._pick_up_brick()
                policy_state = Policy.POLICY_STATE_AVOID
                gripper_state = Policy.GRIPPER_STATE_FULL

            else:
                assert False, "Unknown State"

    def _put_brick_into_box(self):
        self.master.blocking_call(Master.to_slave, 'stop_motion()')
        self.master.blocking_call(Master.to_slave, 'raise_lift()')
        self.master.blocking_call(Master.to_slave, 'approach_brick()')
        self.master.blocking_call(Master.to_slave, 'open_gripper()')
        self.master.blocking_call(Master.to_slave, 'move_straight(-1000, 1000)')
        self.master.blocking_call(Master.to_slave, 'turn_right(180)')
        self.master.blocking_call(Master.to_slave, 'raise_lift_driving()')
        return Policy.POLICY_STATE_SCANNING

    def _pick_up_brick(self):
        self.master.blocking_call(Master.to_slave, 'stop_motion()')
        self.master.blocking_call(Master.to_slave, 'lower_lift()')
        cluster_id = self._approach_brick()
        self.master.blocking_call(Master.to_slave, 'close_gripper()')
        self.master.blocking_call(Master.to_slave, 'raise_lift_driving()')
        return Policy.POLICY_STATE_SCANNING, cluster_id

    def _approach_brick(self):
        self.master.blocking_call(Master.to_slave, "move_forever()")
        while True:
            cluster_id = self.detector.detect_brick_gripper(self.cam_handler_down.get_next_frame())
            if cluster_id is not None:
                print("Cluster", cluster_id)
                break
        self.master.blocking_call(Master.to_slave, "stop_motion()")
        return cluster_id

    def _assess_brick(self):

        img = self.cam_handler_front.get_next_frame()
        lego_map, obstacle_map = self.model.extract_hmap(img)
        obstacle = self.avoider.getObstacle(obstacle_map)
        cv2.imshow('Legomap',lego_map)
        print("Obstacle brick: ",obstacle)
        if obstacle is not None:
            return AssessResult(object=AssessResult.OBJECT_OBSTACLE, point=obstacle[0], max_y=obstacle[1])
        else:
            target = self.tracker.track(lego_map)
            if target is not None:
                return AssessResult(object=AssessResult.OBJECT_TARGET_BRICK, point=target)
            else:
                return AssessResult(object=AssessResult.OBJECT_NONE)
	
    def _assess_box(self, cid):
        assert cid >= 0, "Invalid Cluster ID"

        acc_point = np.zeros(2)
        not_none = 0
        iters = 3

        img = self.cam_handler_front.get_next_frame()
        _, obstacle_map = self.model.extract_hmap(img, True)

        obstacle = self.avoider.getObstacle(obstacle_map)
        target = Detector.detect_box(img, cid)
        if target is not None:
            not_none += 1
            acc_point += target

        print("Obstacle box: ",obstacle)

        for _ in range(iters):
            print("Request_frame")
            img = self.cam_handler_front.get_next_frame()
            target = Detector.detect_box(img, cid)
            '''if the chilitag is far away we can avoid obstacles'''
            if target is not None:
                if obstacle is not None and target[1] < Policy.IGNORE_OBSTACLE_THRESH:
                    return AssessResult(object=AssessResult.OBJECT_OBSTACLE, point=obstacle[0], max_y=obstacle[1])
                not_none += 1
                acc_point += target
		
        if not_none == 0:
            return AssessResult(object=AssessResult.OBJECT_NONE)
        else:

            return AssessResult(object=AssessResult.OBJECT_TARGET_BOX, point=acc_point / not_none)

    def _start_approach(self, x, y, straight_thresh):
        return y > straight_thresh and Policy.THRES_L(y) < x < Policy.THRES_R(y)

    def _avoid_policy(self, pick_up_during_avoidance):
        self.master.blocking_call(Master.to_slave, 'turn_left_forever()')
        while True:
            print("turning")
            img = self.cam_handler_front.get_next_frame()
            _, obstacle_map = self.model.extract_hmap(img)
            obstacle = self.avoider.getObstacle(obstacle_map)
            if obstacle is None:
                break
        print("Moving past the obstacle")
        self.avoid_start_time = time.time()
        self.master.blocking_call(Master.to_slave, 'move_forever()')

        while time.time() - self.avoid_start_time < Policy.AVOID_STRAIGHT_FOR - self.already_avoided:
           if pick_up_during_avoidance:
                print("approach while avoid")	
                self.tracker.reset()
                res = self._assess_brick()
                if res.object == AssessResult.OBJECT_TARGET_BRICK and self._start_approach(res.point[0], res.point[1], Policy.STRAIGHT_THRESH_BRICK):
                    self.already_avoided = time.time() - self.avoid_start_time
                    return Policy.POLICY_STATE_PICK_UP_DURING_AVOIDANCE
        print("Done moving")		
        self.already_avoided = 0
        return Policy.POLICY_STATE_SCANNING


    def _go_to_target_policy(self, assess_fn, straight_thresh):
        state = ''
        while True:
            res = assess_fn()
            if res.object == AssessResult.OBJECT_TARGET_BRICK or res.object == AssessResult.OBJECT_TARGET_BOX:
                x = res.point[0]
                y = res.point[1]
                if self._start_approach(x, y, straight_thresh):
                    return Policy.POLICY_STATE_APPROACHING
                elif x > Policy.THRES_R(y):
                    if state != Policy.MOVEMENT_STATE_RIGHT:
                        state = Policy.MOVEMENT_STATE_RIGHT
                        print('changed to right')
                        self.master.blocking_call(Master.to_slave, 'turn_right_forever()')
                elif x < Policy.THRES_L(y):
                    if state != Policy.MOVEMENT_STATE_LEFT:
                        print('changed to left')
                        state = Policy.MOVEMENT_STATE_LEFT
                        self.master.blocking_call(Master.to_slave, 'turn_left_forever()')
                else:
                    if state != Policy.MOVEMENT_STATE_STRAIGHT:
                        print('changed to straight')
                        state = Policy.MOVEMENT_STATE_STRAIGHT
                        self.master.blocking_call(Master.to_slave, 'move_forever()')
            elif res.object == AssessResult.OBJECT_OBSTACLE:
                print("Obstacle detected")
                return Policy.POLICY_STATE_AVOID
            else:
                return Policy.POLICY_STATE_SCANNING

    def _scan_policy(self, assess_fn):
        turning = False
        while True:
            print("scan policy")
            res = assess_fn()
            if not turning:
                self.master.blocking_call(Master.to_slave, 'turn_right_forever()')    
                turning = True
            if res.object == AssessResult.OBJECT_TARGET_BRICK:
                start = time.time()
                run_time = start
                old_point = res.point
                print('Overshooting')
                while run_time-start<3:
                    run_time = time.time()

                self.tracker.reset()
                res = assess_fn()

                print(old_point)
                print(res.point)
                if res.point is None or old_point[1]>res.point[1]:
                    start = time.time()
                    run_time = start
                    self.master.blocking_call(Master.to_slave, 'turn_left_forever()')
                    while run_time-start<3:
                        res = assess_fn()
                        run_time = time.time()
                return Policy.POLICY_STATE_ON_TARGET
            elif res.object == AssessResult.OBJECT_TARGET_BOX:
                return Policy.POLICY_STATE_ON_TARGET






















