import cv2
import json
import os

img_path = '/home/wangzhe/ICRA2025/MY/data/222.jpg'
output_json = '/home/wangzhe/ICRA2025/MY/data/222.json'
output_key = os.path.basename(img_path)

keypoints = []
img = cv2.imread(img_path)
clone = img.copy()

def click_event(event, x, y, flags, param):
    global keypoints
    if event == cv2.EVENT_LBUTTONDOWN:
        keypoints.append([x, y])
        cv2.circle(img, (x, y), 4, (0, 255, 0), -1)
        cv2.imshow("Image", img)
        print(f"📌 Point {len(keypoints)}: ({x}, {y})")

cv2.imshow("Image", img)
cv2.setMouseCallback("Image", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存 JSON
if keypoints:
    if os.path.exists(output_json):
        with open(output_json, 'r') as f:
            all_data = json.load(f)
    else:
        all_data = {}

    all_data[output_key] = keypoints

    with open(output_json, 'w') as f:
        json.dump(all_data, f, indent=4)

    print(f"✅ Saved {len(keypoints)} keypoints to {output_json}")
else:
    print("❌ No keypoints were selected.")
