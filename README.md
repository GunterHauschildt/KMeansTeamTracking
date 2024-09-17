# Team Tracking
Tracking unpredictable objects such as a person playing a team sport can be tricky.

https://github.com/user-attachments/assets/b09e3491-6ec4-4114-b702-4741bd63776f

https://github.com/user-attachments/assets/a702d437-f1a8-4f22-b72a-8c4b8b694e35

Tracking unpredictable objects such as a person playing a team sport can be tricky.
If the raw bounding box from an object detection algorithm such as Yolo is used, false velocity vectors are created: 
1) when a person changes 'shape' when limbs are extended (kicking balls for example).
2) when a person changes 'shape' when his body is partially hidden by (say) his opponent on another team.

Moreover, the attempts of a player to "loose" his nearby defender tend to also "loose" the object tracker's Kalman filter.

Instead, what this algorithm tries is:
1) Using pose keypoints points to detect not only the player but his/her uniform - as defined by the keypoints at the shoulder & waist.
2) Using KMeans to determine the key color(s) of that uniform by using the polygon as defined by the visible key points.
3) Using KMeans again to determine the team that that uniform is part of.
4) Object tracking by team - so nearby players (say a forward and the opposition's defense) are handled by a different tracker - and as such the tracker's job is made simpler.


Code to come! Its still WIP
