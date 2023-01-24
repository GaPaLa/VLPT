# NOTE: this is _not_ the original code of IDM!
# As such, while it is close and seems to function well,
# its performance might be bit off from what is reported
# in the paper.

from argparse import ArgumentParser
import pickle
import cv2
import numpy as np
import json
import torch as th

from agent import ENV_KWARGS
from inverse_dynamics_model import IDMAgent


KEYBOARD_BUTTON_MAPPING = {
    "key.keyboard.escape" :"ESC",
    "key.keyboard.s" :"back",
    "key.keyboard.q" :"drop",
    "key.keyboard.w" :"forward",
    "key.keyboard.1" :"hotbar.1",
    "key.keyboard.2" :"hotbar.2",
    "key.keyboard.3" :"hotbar.3",
    "key.keyboard.4" :"hotbar.4",
    "key.keyboard.5" :"hotbar.5",
    "key.keyboard.6" :"hotbar.6",
    "key.keyboard.7" :"hotbar.7",
    "key.keyboard.8" :"hotbar.8",
    "key.keyboard.9" :"hotbar.9",
    "key.keyboard.e" :"inventory",
    "key.keyboard.space" :"jump",
    "key.keyboard.a" :"left",
    "key.keyboard.d" :"right",
    "key.keyboard.left.shift" :"sneak",
    "key.keyboard.left.control" :"sprint",
    "key.keyboard.f" :"swapHands",
}

# Template action
NOOP_ACTION = {
    "ESC": 0,
    "back": 0,
    "drop": 0,
    "forward": 0,
    "hotbar.1": 0,
    "hotbar.2": 0,
    "hotbar.3": 0,
    "hotbar.4": 0,
    "hotbar.5": 0,
    "hotbar.6": 0,
    "hotbar.7": 0,
    "hotbar.8": 0,
    "hotbar.9": 0,
    "inventory": 0,
    "jump": 0,
    "left": 0,
    "right": 0,
    "sneak": 0,
    "sprint": 0,
    "swapHands": 0,
    "camera": np.array([0, 0]),
    "attack": 0,
    "use": 0,
    "pickItem": 0,
}

MESSAGE = """
This script will take a video, predict actions for its frames and
and show them with a cv2 window.

Press any button the window to proceed to the next frame.
"""

# Matches a number in the MineRL Java code regarding sensitivity
# This is for mapping from recorded sensitivity to the one used in the model
CAMERA_SCALER = 360.0 / 2400.0


def json_action_to_env_action(json_action):
    """
    Converts a json action into a MineRL action.
    Returns (minerl_action, is_null_action)
    """
    # This might be slow...
    env_action = NOOP_ACTION.copy()
    # As a safeguard, make camera action again so we do not override anything
    env_action["camera"] = np.array([0, 0])

    is_null_action = True
    keyboard_keys = json_action["keyboard"]["keys"]
    for key in keyboard_keys:
        # You can have keys that we do not use, so just skip them
        # NOTE in original training code, ESC was removed and replaced with
        #      "inventory" action if GUI was open.
        #      Not doing it here, as BASALT uses ESC to quit the game.
        if key in KEYBOARD_BUTTON_MAPPING:
            env_action[KEYBOARD_BUTTON_MAPPING[key]] = 1
            is_null_action = False

    mouse = json_action["mouse"]
    camera_action = env_action["camera"]
    camera_action[0] = mouse["dy"] * CAMERA_SCALER
    camera_action[1] = mouse["dx"] * CAMERA_SCALER

    if mouse["dx"] != 0 or mouse["dy"] != 0:
        is_null_action = False
    else:
        if abs(camera_action[0]) > 180:
            camera_action[0] = 0
        if abs(camera_action[1]) > 180:
            camera_action[1] = 0

    mouse_buttons = mouse["buttons"]
    if 0 in mouse_buttons:
        env_action["attack"] = 1
        is_null_action = False
    if 1 in mouse_buttons:
        env_action["use"] = 1
        is_null_action = False
    if 2 in mouse_buttons:
        env_action["pickItem"] = 1
        is_null_action = False

    return env_action, is_null_action


def main(model, weights, video_path, json_path):
    print(MESSAGE)
    agent_parameters = pickle.load(open(model, "rb"))
    net_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = IDMAgent(idm_net_kwargs=net_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(weights)

    required_resolution = ENV_KWARGS["resolution"]
    cap = cv2.VideoCapture(video_path)

    json_index = 0
    with open(json_path) as json_file:
        json_lines = json_file.readlines()
        json_data = "[" + ",".join(json_lines) + "]"
        json_data = json.loads(json_data)

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # the frames  variable cnotains the batch of frames to estimate actions from.
    # it is always ([batch size]x64 + 64) frames.
    # every iteration, the next [batch size]x64
    frames_batch = th.zeros([128,128,128,3])
    predicted_actions = []
    recorded_actions = []
    # continuously get bunch of new frames and return actions until video is over
    current_frame = 0
    video_ended = False
    while not video_ended:
        th.cuda.empty_cache()
        print("=== Loading up frames ===")


        # if at first frame, get starting 64 frames in addition to batchx64 frames (to get to 128 frames)
        # if not at first frame, increment b so that we dont overwrite the past context needed
        frames_to_get = 64
        if current_frame==0:
            frames_to_get = 128
        # add the amount of needed frames.
        for t in range(frames_to_get):
            ret, frame = cap.read()
            if not ret:
                video_ended = True
                break
            assert frame.shape[0] == required_resolution[1] and frame.shape[1] == required_resolution[0], "Video must be of resolution {}".format(required_resolution)
            # BGR -> RGB
            frames_batch[t+(128-frames_to_get)] = frame[..., ::-1] #is always 128 frames. first 64 frames get removed and next 64 are added on every ieration
            env_action, _ = json_action_to_env_action(json_data[json_index])
            recorded_actions.append(env_action)
            json_index += 1

        print("=== Predicting actions ===")
        # predict actions from 64 starter context and 64xbatch_size
        predicted_actions = agent.predict_actions(frames_batch)       # @ THIS LINE IS DIFFERENT; in the paper they onyl take the middle predicted action as the estimated action from IDM so that it has more surrounding context. edit this appropriately.
        if current_frame == 0:
            predicted_actions.append(predicted_actions[0,0:96]) # if first frame, predict actions 0 through 32 despite not being centred. remove last 32 frames since this can be estimated windowed
            predicted_actions.append(predicted_actions[1:,32:96].reshape([batch_size*64]))
        elif video_ended:
            predicted_actions[batch_size-1,frame_2]=
        else:
            predicted_actions = predicted_actions[] # get centre values



        """we apply the IDM over a
        video using a sliding window with stride 64 frames and only use the pseudo-label prediction for
        frames 32 to 96 (the center 64 frames). By doing this, the IDM prediction at the boundary of the
        video clip is never used except for the first and last frames of a full video."""
        ### !thankfully IDM model is not  transformerXL - all internal hidden_states past to itself are None, so we can just convolve it without worrying about weird reucrrence order stuff being messed up

        # remove already  predicted frames except the last 64 frames required for context



        # display frames predicted
        display_frames = False
        for i in range(n_frames):
            frame = frames[i]
            recorded_action = recorded_actions[i]
            cv2.putText(
                frame,
                f"name: prediction (true)",
                (10, 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1
            )
            for y, (action_name, action_array) in enumerate(predicted_actions.items()):
                current_prediction = action_array[0, i]
                cv2.putText(
                    frame,
                    f"{action_name}: {current_prediction} ({recorded_action[action_name]})",
                    (10, 25 + y * 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (255, 255, 255),
                    1
                )
            # RGB -> BGR again...
            cv2.imshow("MineRL IDM model predictions", frame[..., ::-1])
            cv2.waitKey(0)

    cv2.destroyAllWindows()
    assert predicted_actions.shape[1] == current_frame+1


if __name__ == "__main__":
    parser = ArgumentParser("Run IDM on MineRL recordings.")

    parser.add_argument("--weights", type=str, required=True, help="Path to the '.weights' file to be loaded.")
    parser.add_argument("--model", type=str, required=True, help="Path to the '.model' file to be loaded.")
    parser.add_argument("--video-path", type=str, required=True, help="Path to a .mp4 file (Minecraft recording).")
    parser.add_argument("--jsonl-path", type=str, required=True, help="Path to a .jsonl file (Minecraft recording).")
    parser.add_argument("--n-frames", type=int, default=128, help="Number of frames to process at a time.")
    parser.add_argument("--n-batches", type=int, default=10, help="Number of batches (n-frames) to process for visualization.")

    args = parser.parse_args()

    main(args.model, args.weights, args.video_path, args.jsonl_path, args.n_batches, args.n_frames)
