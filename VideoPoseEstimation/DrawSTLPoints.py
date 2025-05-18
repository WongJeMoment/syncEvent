import cv2
def draw_projected_keypoints(image, object_points, rvec, tvec, camera_matrix):
    """将 STL 的关键点投影、绘制并标出序号"""
    projected_pts, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, distCoeffs=None)
    projected_pts = projected_pts.reshape(-1, 2)
    for idx, pt in enumerate(projected_pts):
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # 红点
        cv2.putText(image, str(idx), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, lineType=cv2.LINE_AA)  # 绿色索引
    return image