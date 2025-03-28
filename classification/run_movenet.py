import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import torch

from model import BaseModel


def _get_clips(full_kp, clip_len, num_clips=1, p_interval=(1, 1)):
    M, T, V, C = full_kp.shape

    # print(f"--> {full_kp} {full_kp.shape}")
    clips = []

    for clip_idx in range(num_clips):
        pi = p_interval
        ratio = np.random.rand() * (pi[1] - pi[0]) + pi[0]
        num_frames = int(ratio * T)
        off = np.random.randint(T - num_frames + 1)

        if num_frames < clip_len:
            start = np.random.randint(0, num_frames)
            inds = (np.arange(start, start + clip_len) % num_frames) + off
            clip = full_kp[:, inds].copy()
        elif clip_len <= num_frames < 2 * clip_len:
            basic = np.arange(clip_len)
            inds = np.random.choice(clip_len + 1, num_frames - clip_len, replace=False)
            offset = np.zeros(clip_len + 1, dtype=np.int64)
            offset[inds] = 1
            inds = basic + np.cumsum(offset)[:-1] + off
            clip = full_kp[:, inds].copy()
        else:
            bids = np.array([i * num_frames // clip_len for i in range(clip_len + 1)])
            bsize = np.diff(bids)
            bst = bids[:clip_len]
            offset = np.random.randint(bsize)
            inds = bst + offset + off
            clip = full_kp[:, inds].copy()
        clips.append(clip)
    return np.concatenate(clips, 1)


if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
else:
    print("No GPU available. Training will run on CPU.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
input_size = 192
movenet = module.signatures["serving_default"]

mn_to_ntu = {0: 4, 5: 5, 9: 6, 7: 6, 8: 10, 9: 8, 10: 12, 11: 13, 12: 17, 13: 14, 14: 18, 15: 15, 16: 19}

ntu_to_mn = dict()
for key, value in mn_to_ntu.items():
    ntu_to_mn[value] = key


cap = cv2.VideoCapture("data/2.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

model = BaseModel(checkpoint="models/best_top1_acc_epoch_150.pth", num_classes=120, device=device)

num_clips = 1
clip_len = 25
clip = np.array([])
keypoints_len = 25


def detect_keypoints(image):
    global clip

    image = cv2.resize(image, (640, 640))
    img = image.copy()

    input_image = tf.expand_dims(img, axis=0)
    input_image = tf.image.resize_with_pad(input_image, input_size, input_size)
    input_image = tf.cast(input_image, dtype=tf.int32)

    results = movenet(input_image)

    keypoints_with_scores = results["output_0"].numpy().reshape((17, 3))
    print(keypoints_with_scores.shape)
    keypoints = np.zeros((keypoints_len, 3))
    for i in range(keypoints_with_scores.shape[0]):
        if i in mn_to_ntu:
            keypoints[mn_to_ntu[i]] = keypoints_with_scores[i]

    l = clip.shape[0]
    result = None
    if l > 0:
        clip = np.append(clip, [keypoints], axis=0)
    elif l == 0:
        clip = np.expand_dims(keypoints, axis=0)
    if l >= clip_len:
        clip = np.delete(clip, 0, axis=0)

        np_keypoints = np.expand_dims(clip, axis=0)
        np_keypoints = _get_clips(np_keypoints, clip_len=clip_len)
        M, T, V, C = np_keypoints.shape
        np_keypoints = np_keypoints.reshape((M, num_clips, T // num_clips, V, C)).transpose(1, 0, 2, 3, 4)
        np_keypoints = np.ascontiguousarray(np_keypoints)
        np_keypoints = np.expand_dims(np_keypoints, axis=0)

        np_keypoints = torch.tensor(np_keypoints, dtype=torch.float32)
        np_keypoints = np_keypoints.to(device)
        label = torch.tensor([8]).to(device)
        print(np_keypoints.shape)

        data = dict(keypoint=np_keypoints, label=label)
        ind, result = model.run(data)

    return result


cap = cv2.VideoCapture("data/3.mp4")
ret, frame = cap.read()
fourcc = cv2.VideoWriter_fourcc(*"MP4V")
out = cv2.VideoWriter("output/output_movenet4.mp4", fourcc, 25, (frame.shape[1], frame.shape[0]))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    class_name = detect_keypoints(frame)
    cv2.putText(frame, str(class_name), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    out.write(frame)

    cv2.imshow("Keypoint Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
