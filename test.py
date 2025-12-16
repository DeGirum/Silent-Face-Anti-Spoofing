# -*- coding: utf-8 -*-
# @Time : 20-6-9 下午3:06
# @Author : zhuying
# @Company : Minivision
# @File : test.py
# @Software : PyCharm

import os
import cv2
import numpy as np
import argparse
import warnings
import time

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

warnings.filterwarnings("ignore")


def test_camera(model_dir, device_id, camera_id=0):
    """
    Test anti-spoofing on camera stream
    Args:
        model_dir: path to anti-spoof models
        device_id: GPU device id
        camera_id: camera device id (default 0)
    """
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()

    # Open camera
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Get face bounding box
        image_bbox = model_test.get_bbox(frame)

        if image_bbox[0] == -1:
            # No face detected
            cv2.putText(
                frame,
                "No face detected",
                (10, 30),
                cv2.FONT_HERSHEY_COMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
        else:
            # Perform prediction
            prediction = np.zeros((1, 3))
            test_speed = 0

            # Sum predictions from all models
            for model_name in os.listdir(model_dir):
                h_input, w_input, model_type, scale = parse_model_name(model_name)
                param = {
                    "org_img": frame,
                    "bbox": image_bbox,
                    "scale": scale,
                    "out_w": w_input,
                    "out_h": h_input,
                    "crop": True,
                }
                if scale is None:
                    param["crop"] = False
                img = image_cropper.crop(**param)
                start = time.time()
                prediction += model_test.predict(
                    img, os.path.join(model_dir, model_name)
                )
                test_speed += time.time() - start

            # Draw result
            label = np.argmax(prediction)
            value = prediction[0][label] / 2

            if label == 1:
                result_text = "RealFace: {:.2f}".format(value)
                color = (0, 255, 0)  # Green for real
            else:
                result_text = "FakeFace: {:.2f}".format(value)
                color = (0, 0, 255)  # Red for fake

            # Draw bounding box
            cv2.rectangle(
                frame,
                (image_bbox[0], image_bbox[1]),
                (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
                color,
                2,
            )

            # Draw prediction text
            cv2.putText(
                frame,
                result_text,
                (image_bbox[0], image_bbox[1] - 10),
                cv2.FONT_HERSHEY_COMPLEX,
                0.7,
                color,
                2,
            )

            # Draw FPS
            fps_text = "FPS: {:.2f}".format(1.0 / test_speed if test_speed > 0 else 0)
            cv2.putText(
                frame,
                fps_text,
                (10, 30),
                cv2.FONT_HERSHEY_COMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        # Display frame
        cv2.imshow("Anti-Spoofing Detection", frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    desc = "Anti-spoofing test on camera stream"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--device_id", type=int, default=0, help="which gpu id, [0/1/2/3]"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./resources/anti_spoof_models",
        help="model_lib used to test",
    )
    parser.add_argument(
        "--camera_id", type=int, default=0, help="camera device id (default 0)"
    )
    args = parser.parse_args()
    test_camera(args.model_dir, args.device_id, args.camera_id)
