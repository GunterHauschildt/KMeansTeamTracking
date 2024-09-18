import os
import numpy as np
from numpy.typing import NDArray
import cv2 as cv
import glob
import torch
import json
import argparse
from ultralytics import YOLO
import norfair as nf
from keypoints import KeyPoints
from team_classifier import TeamClassifier
from utils.object_tracker import ObjectTracker, MotionEstimator
from utils.video_capture import VideoStream
from dataclasses import dataclass


def main():
    has_cuda = torch.cuda.is_available()
    device = "cpu" if not has_cuda else "cuda:0"

    parser = argparse.ArgumentParser()
    parser.add_argument('--video-source', type=str, default=None)
    parser.add_argument('--video-write', type=bool, default=False)
    parser.add_argument('--team-colors', type=str,
                        default='{"Home": "lightgray", "Visitors": "darkblue"}')
    parser.add_argument('--tracker-distance-threshold-perc', type=float, default=.1)
    parser.add_argument('--tracker-timeout-s', type=float, default=2)
    parser.add_argument('--tracker-r', type=float, default=16.)
    parser.add_argument('--tracker-q', type=float, default=.05)
    parser.add_argument('--yolo-model', type=str, default='yolov8l-pose.pt')
    parser.add_argument('--load-yolo', type=bool, default=False)
    parser.add_argument('--write-yolo', type=bool, default=False)
    args = parser.parse_args()
    team_colors = json.loads(args.team_colors)

    # if debugging on a non-gpu machine its quicker just to do the
    # yolo predictions once - they won't change
    args.load_yolo = True
    args.write_yolo = False
    player_detection = None
    if args.write_yolo or not args.load_yolo:
        player_detection = YOLO(args.yolo_model)

    # set the video stream i/o and a helper window for display
    video_stream = VideoStream(args.video_source)
    cv.namedWindow("frame", cv.WINDOW_NORMAL)
    cv.resizeWindow("frame",
                    round(video_stream.size_xy()[0] * .25),
                    round(video_stream.size_xy()[1] * .25)
                    )

    file_name = os.path.basename(args.video_source)
    video_writer = None
    if args.video_write:
        name = os.path.splitext(file_name)[0]
        ext = os.path.splitext(file_name)[1]
        video_writer = cv.VideoWriter(name + "_teams" + ext,
                                      cv.VideoWriter.fourcc(*'mp4v'),
                                      video_stream.fps(),
                                      video_stream.size_xy()
                                      )

    # set the team_classifier. its team_colors parameter is just used to name
    # the team according the closest colors
    team_classifier = TeamClassifier(team_colors)

    # motion estimator to deal with camera movement,
    motion_estimator = MotionEstimator(video_stream.size_xy(), np.uint8)

    # create the object trackers - one per team
    object_tracker = {}
    for team_name in team_colors:
        object_tracker[team_name] = ObjectTracker(args, video_stream)

    #########################
    # ok enough initializing lets get on with it

    while (frame := video_stream.next_frame()) is not None:

        # frames i/o
        frame_num = frame[1]
        frame = frame[0]
        frame_draw = frame.copy()
        # for camera motion estimation
        # detected_objects_mask = np.ones(frame.shape[:2], frame.dtype)
        motion_estimator.reset()

        # data i/o as needed if debugging
        folder = None
        if args.write_yolo or args.load_yolo:
            folder = os.path.splitext(file_name)[0]
            folder = os.path.join(folder, str(frame_num))
        if args.write_yolo:
            os.makedirs(folder, exist_ok=False)

        if args.write_yolo or not args.load_yolo:
            players = player_detection.predict(frame, conf=.25, iou=.80, imgsz=(768, 1280), device=device)
        elif args.load_yolo:
            players = len(glob.glob(os.path.join(folder, "box_*.npy")))
            players = [[p for p in range(players)]]

        # get the boxes, keypoints and keypoint confidences
        # save if required

        @dataclass
        class PlayerInfo:
            color_lab: tuple[float, float, float]
            box: NDArray
            body_box: NDArray
            xys: NDArray
            confs: NDArray

        all_players_info = []
        for p, player in enumerate(players[0]):
            if not args.load_yolo:
                box = player.boxes[0].xyxy.detach().cpu().numpy()[0].astype(np.int32)
                xys = player.keypoints.xy.detach().cpu().numpy()
                xys = np.squeeze(xys, axis=0)
                confs = player.keypoints.conf.detach().cpu().numpy()
                confs = np.squeeze(confs, axis=0)
            else:
                box = np.load(os.path.join(folder, f"box_{p}.npy"))
                xys = np.load(os.path.join(folder, f"xys_{p}.npy"))
                confs = np.load(os.path.join(folder, f"confs_{p}.npy"))

            if args.write_yolo:
                np.save(os.path.join(folder, f"box_{p}"), box)
                np.save(os.path.join(folder, f"xys_{p}"), xys)
                np.save(os.path.join(folder, f"confs_{p}"), confs)
                print(f"Wrote box, xys, confs for {frame_num} {p}, continuing ...!")
                continue

            # for camera motion detection, move the objects from the mask.
            # ideally we'll be left with only the stationary objects that
            # calling goodFeaturesToTrack and getHomography can use
            # (and Norfair does this for  us)
            motion_estimator.update_mask((box[0], box[1]), (box[2], box[3]))

            # find the player keypoints and his body as defined by those keypoints
            player_body_box, player_body_roi = KeyPoints(frame, "", box, xys, confs).body()

            if player_body_box is None or player_body_roi is None:
                continue

            # find the players (uniform) color as defined by the body roi
            this_player_colors = TeamClassifier.get_colors(player_body_roi, max_clusters=2)
            if this_player_colors is not None:
                all_players_info.append(
                    PlayerInfo(this_player_colors[0].lab, box, player_body_box, xys, confs)
                )
                TeamClassifier.draw(frame_draw, player_body_box, this_player_colors)

        # motion estimation
        motion_transform = motion_estimator.update(frame)

        # predict the teams by their colors
        teams = team_classifier.predict([player_info.color_lab for player_info in all_players_info])
        if teams is None:
            continue

        # ... and track the players, per team. Each detection gets only one team's players.
        for team_name, players in teams.items():
            detections = []
            for player in players:
                detections.append(nf.Detection(
                    points=all_players_info[player].body_box,
                    data=all_players_info[player])
                )

            team = object_tracker[team_name].tracker().update(detections=detections,
                                                              coord_transformations=motion_transform)

            # and draw, cause well, that's all this demo does
            for team_member in team:
                color = team_classifier.uniform_color_bgr(team_name)
                frame_draw = KeyPoints(frame, team_member.id,
                                       team_member.last_detection.data.box,
                                       team_member.last_detection.data.xys,
                                       team_member.last_detection.data.confs
                                       ).draw(frame_draw, color)

        # and we're done. draw. waitKey(1) might be better depending on your gpu/pc
        if video_writer is not None:
            video_writer.write(frame_draw)
        cv.imshow("frame", frame_draw)
        cv.waitKey(video_stream.mspf())


if __name__ == "__main__":
    main()
