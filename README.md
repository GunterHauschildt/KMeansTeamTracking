# Team Tracking

https://github.com/user-attachments/assets/b09e3491-6ec4-4114-b702-4741bd63776f

Tracking unpredictable objects such as a person playing a team sport can be tricky.
If the raw bounding box from an object detection algorithm such as Yolo is used, velocity vectors are created 
1) when a person changes 'shape' when limbs are extended (kicking balls for example)
2) when a person changes 'shape' when his body is partially hidden by (say) his opponent on another team

Instead, what this algorithm tries is:
1) Using pose keypoints points (right out of the box from YoloV8) to detect not only the player but his/her uniform
2) Using KMeans to determine the key color(s) of that uniform
3) Using KMeans to determine the team that that uniform belongs to
4) Object tracking by team - so nearby players (say a forward and the opposition's defense) are handled by a different tracker.



Code to come! Its still WIP
