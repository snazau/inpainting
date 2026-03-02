import argparse
import os
import psutil

import cv2
import numpy as np
import torch
import torchvision

from masking.track_anything import TrackingAnything
from model.misc import get_device


def parse_augment():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--sam_model_type', type=str, default="vit_h")
    parser.add_argument('--port', type=int, default=8000, help="only useful when running gradio applications")
    parser.add_argument('--mask_save', default=False)
    args = parser.parse_args()

    if not args.device:
        args.device = str(get_device())

    return args


def get_frames_from_video(video_path, downscale=False):
    frames = list()
    try:
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        while cap.isOpened():
            ret, frame = cap.read()

            if ret and downscale:
                height, width = frame.shape[:2]
                new_size = (int(width // 2), int(height // 2))
                frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_CUBIC)

            if ret is True:
                current_memory_usage = psutil.virtual_memory().percent
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if current_memory_usage > 90:
                    print("Memory usage is too high (>90%). Please reduce the video resolution or frame rate.")
                    break
            else:
                break
    except (OSError, TypeError, ValueError, KeyError, SyntaxError) as e:
        print("read_frame_source:{} error. {}\n".format(video_path, str(e)))
        raise e

    return frames, fps


def collect_point_prompt(reference_frame):
    clicked_points = list()
    clicked_labels = list()

    reference_frame_copy = reference_frame.copy()
    reference_frame_copy = cv2.cvtColor(reference_frame_copy, cv2.COLOR_BGR2RGB)

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Record the coordinates
            clicked_points.append((x, y))
            clicked_labels.append(1)
            print(f"Clicked at coordinates: ({x}, {y})")

            # Optional: Draw a circle and text on the image where clicked
            font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(reference_frame_copy, '(+)', (x, y), font, 0.5, (0, 0, 255), 2)
            cv2.circle(reference_frame_copy, (x, y), 3, (0, 0, 255), -1)
            cv2.imshow("reference_frame", reference_frame_copy)
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Record the coordinates
            clicked_points.append((x, y))
            clicked_labels.append(0)
            print(f"Clicked at coordinates: ({x}, {y})")

            # Optional: Draw a circle and text on the image where clicked
            font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(reference_frame_copy, '(-)', (x, y), font, 0.5, (255, 0, 0), 2)
            cv2.circle(reference_frame_copy, (x, y), 3, (255, 0, 0), -1)
            cv2.imshow("reference_frame", reference_frame)

    cv2.namedWindow("reference_frame")
    cv2.setMouseCallback("reference_frame", click_event)
    print("Click on the reference_frame window. Press 'Esc' to exit.")

    while True:
        cv2.imshow("reference_frame", reference_frame_copy)

        # Wait for a key press for 1ms; key 27 is the Esc key
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    # Close all windows
    cv2.destroyAllWindows()
    print(f"All clicked points: {clicked_points}")
    print(f"All clicked labels: {clicked_labels}")

    clicked_points = np.array(clicked_points)
    clicked_labels = np.array(clicked_labels)

    return clicked_points, clicked_labels


def sam_refine(tracking_model, video_state, prompt_points, prompt_labels, multimask):
    # prompt for sam model
    reference_frame = video_state["origin_images"][video_state["select_frame_number"]]
    tracking_model.samcontroler.sam_controler.reset_image()
    tracking_model.samcontroler.sam_controler.set_image(reference_frame)

    mask, logit, painted_image = tracking_model.first_frame_click(
        image=reference_frame,
        points=prompt_points,
        labels=prompt_labels,
        multimask=multimask,
    )
    painted_image = np.asarray(painted_image)
    print(f"mask = {mask.shape} min = {mask.min()} mean = {mask.mean()} max = {mask.max()}")
    print(f"logit = {logit.shape} min = {logit.min()} mean = {logit.mean()} max = {logit.max()}")
    print(f"reference_frame = {reference_frame.shape} min = {reference_frame.min()} mean = {reference_frame.mean()} max = {reference_frame.max()}")
    print(f"painted_image = {painted_image.shape} min = {painted_image.min()} mean = {painted_image.mean()} max = {painted_image.max()}")

    return mask, logit, painted_image


def generate_video_from_frames(frames, output_path, fps=30):
    frames = torch.from_numpy(np.asarray(frames))
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    torchvision.io.write_video(output_path, frames, fps=fps, video_codec="libx264")
    return output_path


def vos_tracking_video(tracking_model, video_state):
    tracking_model.cutie.clear_memory()
    following_frames = video_state["origin_images"][video_state["select_frame_number"]:video_state["track_end_number"] + 1]

    template_mask = video_state["masks"][video_state["select_frame_number"]]
    if len(np.unique(template_mask)) == 1:
        raise Exception("Please add at least one mask to track by clicking the image")

    masks, logits, painted_images = tracking_model.generator(images=following_frames, template_mask=template_mask)
    tracking_model.cutie.clear_memory()

    return masks, logits, painted_images


def inpaint_video(
        tracking_model,
        video_state,
        resize_ratio,
        dilate_radius,
        raft_iter,
        subvideo_length,
        neighbor_length,
        ref_stride,
):
    frames = np.asarray(video_state["origin_images"])
    inpaint_masks = np.asarray(video_state["masks"])

    # inpaint for videos
    inpainted_frames = tracking_model.baseinpainter.inpaint(
        frames,
        inpaint_masks,
        ratio=resize_ratio,
        dilate_radius=dilate_radius,
        raft_iter=raft_iter,
        subvideo_length=subvideo_length,
        neighbor_length=neighbor_length,
        ref_stride=ref_stride,
    )  # numpy array, T, H, W, 3

    return inpainted_frames


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # common params
    args = parse_augment()

    downscale = False
    # downscale = True
    # video_path = "./data/test_sample/test-sample0.mp4"  # ice
    # video_path = "./data/test_sample/test-sample1.mp4"  # parkour
    # video_path = "./data/test_sample/test-sample2.mp4"  # bicycle
    # video_path = "./data/test_sample/test-sample3.mp4"  # spider
    video_path = "./data/test_sample/test-sample4.mp4"  # cars
    # video_path = "./data/test_sample/VID_20260227_173012.mp4"  # selfie
    # video_path = "./data/test_sample/VID_20260301_141420.mp4"  # shoes
    # video_path = "./data/test_sample/VID_20260301_141543.mp4"  # PC
    # video_path = "./data/test_sample/VID_20260301_161822.mp4"  # ruler
    # video_path = "./data/test_sample/arcane_clip_01.mp4"
    # video_path = "./data/test_sample/zootopia_clip_01.mp4"
    save_dir = "./data/results/"
    save_suffix = "_ds" if downscale else ""
    sam_checkpoint_dict = {"vit_h": "sam_vit_h_4b8939.pth"}
    checkpoint_folder = os.path.join(".", "weights")

    # ProPainter params
    resize_ratio = 1.0
    # resize_ratio = 0.5
    dilate_radius = 4
    raft_iter = 20
    subvideo_length = 80
    neighbor_length = 10
    ref_stride = 10
    if resize_ratio != 1.0:
        save_suffix += f"_s{resize_ratio}"

    # initialize sam, cutie, propainter models
    sam_checkpoint = os.path.join(checkpoint_folder, sam_checkpoint_dict[args.sam_model_type])
    cutie_checkpoint = os.path.join(checkpoint_folder, "cutie-base-mega.pth")
    propainter_checkpoint = os.path.join(checkpoint_folder, "ProPainter.pth")
    raft_checkpoint = os.path.join(checkpoint_folder, "raft-things.pth")
    flow_completion_checkpoint = os.path.join(checkpoint_folder, "recurrent_flow_completion.pth")

    model = TrackingAnything(
        sam_checkpoint,
        cutie_checkpoint,
        propainter_checkpoint,
        raft_checkpoint,
        flow_completion_checkpoint,
        args,
    )

    # read video
    frames, fps = get_frames_from_video(video_path, downscale)

    frame_num = len(frames)
    height, width = frames[0].shape[0], frames[0].shape[1]
    video_state = {
        "video_name": os.path.splitext(os.path.split(video_path)[-1])[0],

        "origin_images": frames,
        "painted_images": frames.copy(),
        "masks": [np.zeros((height, width), np.uint8)] * frame_num,
        "logits": [None] * frame_num,

        "select_frame_number": 0,
        "track_end_number": frame_num - 1,

        "fps": fps,
        "frame_num": frame_num,
        "height": height,
        "width": width,
    }

    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(video_state["origin_images"][0])

    # get point prompt
    prompt_points, prompt_labels = collect_point_prompt(
        video_state["origin_images"][video_state["select_frame_number"]]
    )

    # use SAM for reference frame using point-prompt
    mask, logit, painted_image = sam_refine(model, video_state, prompt_points, prompt_labels, multimask=True)
    video_state["masks"][video_state["select_frame_number"]] = mask
    video_state["logits"][video_state["select_frame_number"]] = logit
    video_state["painted_images"][video_state["select_frame_number"]] = painted_image

    # visualization
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    axs[0, 0].imshow(video_state["origin_images"][video_state["select_frame_number"]])
    axs[0, 0].set_title("reference_frame")

    axs[0, 1].imshow(painted_image)
    axs[0, 1].set_title("painted_image")

    axs[1, 0].imshow(mask)
    axs[1, 0].set_title("mask")

    # axs[1, 1].imshow(logit)
    # axs[1, 1].set_title("logit")

    plt.show()

    # mask tracking with Cutie
    masks, logits, painted_images = vos_tracking_video(model, video_state)
    video_state["masks"][video_state["select_frame_number"]:video_state["track_end_number"] + 1] = masks
    video_state["logits"][video_state["select_frame_number"]:video_state["track_end_number"] + 1] = logits
    video_state["painted_images"][video_state["select_frame_number"]:video_state["track_end_number"] + 1] = painted_images

    generate_video_from_frames(
        video_state["painted_images"],
        output_path=os.path.join(save_dir, "tracking", f'{video_state["video_name"]}{save_suffix}.mp4'),
        fps=video_state["fps"],
    )

    # inpainting with ProPainter
    inpainted_frames = inpaint_video(
        model,
        video_state,
        resize_ratio,
        dilate_radius,
        raft_iter,
        subvideo_length,
        neighbor_length,
        ref_stride,
    )

    video_output = generate_video_from_frames(
        inpainted_frames,
        output_path=os.path.join(save_dir, "inpainting", f'{video_state["video_name"]}{save_suffix}.mp4'),
        fps=fps
    )