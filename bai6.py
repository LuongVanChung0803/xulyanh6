import cv2
import numpy as np
from sklearn.cluster import KMeans
from fcmeans import FCM
import matplotlib.pyplot as plt

# Đọc ảnh vệ tinh
image = cv2.imread("D:/XLA/vt/input.jpg")

if image is None:
    print("Không thể đọc ảnh, hãy kiểm tra đường dẫn.")
else:
    # Chuyển ảnh sang RGB và lấy dữ liệu điểm ảnh
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixel_values = image_rgb.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # K-Means clustering
    k = 2  # Số cụm
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans_labels = kmeans.fit_predict(pixel_values)
    kmeans_centers = kmeans.cluster_centers_

    # Tạo ảnh từ kết quả K-Means
    kmeans_segmented_image = kmeans_centers[kmeans_labels.flatten()]
    kmeans_segmented_image = kmeans_segmented_image.reshape(image.shape)
    kmeans_segmented_image = np.uint8(kmeans_segmented_image)

    # FCM clustering
    fcm = FCM(n_clusters=k)
    fcm.fit(pixel_values)
    fcm_labels = fcm.predict(pixel_values)
    fcm_centers = fcm.centers

    # Tạo ảnh từ kết quả FCM
    fcm_segmented_image = fcm_centers[fcm_labels]
    fcm_segmented_image = fcm_segmented_image.reshape(image.shape)
    fcm_segmented_image = np.uint8(fcm_segmented_image)

    # Phát hiện cạnh với Canny
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)

    # Tìm các hình chữ nhật (ngôi nhà) trong ảnh
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Lọc các hình chữ nhật có diện tích lớn (ngôi nhà)
    house_contours = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:  # Hình chữ nhật hoặc gần hình chữ nhật
            if cv2.contourArea(contour) > 500:  # Diện tích ngưỡng
                house_contours.append(approx)

    # Vẽ các hình chữ nhật lên ảnh
    house_image = image.copy()
    cv2.drawContours(house_image, house_contours, -1, (0, 255, 0), 2)

    # Hiển thị ảnh K-Means, FCM, và ảnh có hình chữ nhật (nhà)
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    axes[0].imshow(kmeans_segmented_image)
    axes[0].set_title("K-Means Segmentation")
    axes[0].axis("off")

    axes[1].imshow(fcm_segmented_image)
    axes[1].set_title("FCM Segmentation")
    axes[1].axis("off")

    axes[2].imshow(cv2.cvtColor(house_image, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Detected Houses")
    axes[2].axis("off")

    # Lưu ảnh kết quả
    output_path = "D:/XLA/vt/kmeans_fcm_house_detection1.jpg"
    plt.savefig(output_path)
    print(f"Ảnh kết quả đã được lưu tại {output_path}")

    plt.show()
