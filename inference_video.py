import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings
import _thread
import skvideo.io
from model.VFSS_infer_b import Model
from queue import Queue
import time

warnings.filterwarnings("ignore")


def transferAudio(sourceVideo, targetVideo):
    import shutil
    import moviepy.editor
    tempAudioFileName = "./temp/audio.mkv"

    # split audio from original video file and store in "temp" directory
    if True:

        # clear old "temp" directory if it exits
        if os.path.isdir("temp"):
            # remove temp directory
            shutil.rmtree("temp")
        # create new "temp" directory
        os.makedirs("temp")
        # extract audio from video
        os.system('ffmpeg -y -i "{}" -c:a copy -vn {}'.format(sourceVideo, tempAudioFileName))

    targetNoAudio = os.path.splitext(targetVideo)[0] + "_noaudio" + os.path.splitext(targetVideo)[1]
    os.rename(targetVideo, targetNoAudio)
    # combine audio file and new video file
    os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(targetNoAudio, tempAudioFileName, targetVideo))

    if os.path.getsize(
            targetVideo) == 0:  # if ffmpeg failed to merge the video and audio together try converting the audio to aac
        tempAudioFileName = "./temp/audio.m4a"
        os.system('ffmpeg -y -i "{}" -c:a aac -b:a 160k -vn {}'.format(sourceVideo, tempAudioFileName))
        os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(targetNoAudio, tempAudioFileName, targetVideo))
        if (os.path.getsize(targetVideo) == 0):  # if aac is not supported by selected format
            os.rename(targetNoAudio, targetVideo)
            print("Audio transfer failed. Interpolated video will have no audio")
        else:
            print("Lossless audio transfer failed. Audio was transcoded to AAC (M4A) instead.")

            # remove audio-less video
            os.remove(targetNoAudio)
    else:
        os.remove(targetNoAudio)

    # remove temp directory
    shutil.rmtree("temp")


parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
parser.add_argument('--video', dest='video', type=str, default=None)
parser.add_argument('--output', dest='output', type=str, default=None)
parser.add_argument('--img', dest='img', type=str, default=None)
parser.add_argument('--montage', dest='montage', action='store_true', help='montage origin video')
parser.add_argument('--model', dest='modelDir', type=str, default='train_log',
                    help='directory with trained model files')
parser.add_argument('--scale', dest='scale', type=float, default=1.0, help='Try scale=0.5 for 4k video')
parser.add_argument('--fps', dest='fps', type=int, default=None)
parser.add_argument('--png', dest='png', action='store_true', help='whether to vid_out png format vid_outs')
parser.add_argument('--ext', dest='ext', type=str, default='mkv', help='vid_out video extension')
parser.add_argument('--exp', dest='exp', type=int, default=1)
parser.add_argument('--multi', dest='multi', type=int, default=2)
parser.add_argument('--flow_mode', dest='mode', type=str, default='MOF',
                    help='MOF use 4 frames when inference video, BOF use 3 frames')
parser.add_argument('--model_type', dest='model_type', type=str, default='anime',
                    help='anime/real')

args = parser.parse_args()
if args.exp != 1:
    args.multi = (2 ** args.exp)
assert (not args.video is None or not args.img is None)
assert args.scale in [0.25, 0.5, 1.0, 2.0, 4.0]
if not args.img is None:
    args.png = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

model = Model(args.mode)
if args.model_type == 'anime':
    model.load_model(os.path.join(args.modelDir, 'anime'), -1)
else:
    model.load_model(os.path.join(args.modelDir, 'real'), -1)
print("Loaded model")
model.eval()
model.device()

if not args.video is None:
    videoCapture = cv2.VideoCapture(args.video)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    tot_frame = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    videoCapture.release()
    if args.fps is None:
        fpsNotAssigned = True
        args.fps = fps * args.multi
    else:
        fpsNotAssigned = False
    videogen = skvideo.io.vreader(args.video)
    i0 = next(videogen)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video_path_wo_ext, ext = os.path.splitext(args.video)
    print('{}.{}, {} frames in total, {}FPS to {}FPS'.format(video_path_wo_ext, args.ext, tot_frame, fps, args.fps))
    if args.png == False and fpsNotAssigned == True:
        print("The audio will be merged after interpolation process")
    else:
        print("Will not merge audio because using png or fps flag!")
else:
    videogen = []
    for f in os.listdir(args.img):
        if 'png' in f:
            videogen.append(f)
    tot_frame = len(videogen)
    videogen.sort(key=lambda x: int(x[:-4]))
    i0 = cv2.imread(os.path.join(args.img, videogen[0]), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
    videogen = videogen[1:]
h, w, _ = i0.shape
vid_out_name = None
vid_out = None
if args.png:
    if not os.path.exists('vid_out'):
        os.mkdir('vid_out')
else:
    if args.output is not None:
        vid_out_name = args.output
    else:
        vid_out_name = '{}_{}X_{}fps.{}'.format(video_path_wo_ext, args.multi, int(np.round(args.fps)), args.ext)
    vid_out = cv2.VideoWriter(vid_out_name, fourcc, args.fps, (w, h))


def clear_write_buffer(user_args, write_buffer):
    cnt = 0
    while True:
        item = write_buffer.get()
        if item is None:
            break
        if user_args.png:
            cv2.imwrite('vid_out/{:0>7d}.png'.format(cnt), item[:, :, ::-1])
            cnt += 1
        else:
            vid_out.write(item[:, :, ::-1])


def build_read_buffer(user_args, read_buffer, videogen):
    try:
        for frame in videogen:
            if not user_args.img is None:
                frame = cv2.imread(os.path.join(user_args.img, frame))[:, :, ::-1].copy()
            if user_args.montage:
                frame = frame[:, left: left + w]
            read_buffer.put(frame)
    except:
        pass
    read_buffer.put(None)


def make_inference(I0, I1, I2, I3, n, scale):
    global model
    reusethings = model.reuse(I0, I1, I2, I3, scale)
    output = []
    step = 1 / (n + 1)
    target_ts = [i * step for i in range(1, n + 1)]
    for t in target_ts:
        output.append(model.inference(I0, I1, I2, I3, reusethings, t))
    return output


def pad_image(img):
    return F.pad(img, padding)


if args.montage:
    left = w // 4
    w = w // 2
tmp = max(64, int(64 / args.scale))
ph = ((h - 1) // tmp + 1) * tmp
pw = ((w - 1) // tmp + 1) * tmp
padding = (0, pw - w, 0, ph - h)
pbar = tqdm(total=tot_frame)
if args.montage:
    i0 = i0[:, left: left + w]
write_buffer = Queue(maxsize=500)
read_buffer = Queue(maxsize=500)
_thread.start_new_thread(build_read_buffer, (args, read_buffer, videogen))
_thread.start_new_thread(clear_write_buffer, (args, write_buffer))

# head of the video
I0 = torch.from_numpy(np.transpose(i0, (2, 0, 1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
I0 = F.interpolate(I0, (ph, pw), mode='bilinear', align_corners=False)
i1 = read_buffer.get()
assert i1 is not None
I1 = torch.from_numpy(np.transpose(i1, (2, 0, 1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
I1 = F.interpolate(I1, (ph, pw), mode='bilinear', align_corners=False)
i2 = read_buffer.get()
assert i2 is not None
I2 = torch.from_numpy(np.transpose(i2, (2, 0, 1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
I2 = F.interpolate(I2, (ph, pw), mode='bilinear', align_corners=False)
output = make_inference(I0, I0, I1, I1, args.multi - 1, args.scale)
if args.montage:
    write_buffer.put(np.concatenate((i0, i0), 1))
    for mid in output:
        mid = (mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)
        write_buffer.put(np.concatenate((i0, mid[:h, :w]), 1))
else:
    write_buffer.put(i0)
    for mid in output:
        mid = F.interpolate(mid, (h, w), mode='bilinear', align_corners=False)
        mid = (mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)
        write_buffer.put(mid)

while True:
    i3 = read_buffer.get()
    if i3 is None:
        # end of the video
        output = make_inference(I1, I1, I2, I2, args.multi - 1, args.scale)
    else:
        I3 = torch.from_numpy(np.transpose(i3, (2, 0, 1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
        I3 = F.interpolate(I3, (ph, pw), mode='bilinear', align_corners=False)
        output = make_inference(I0, I1, I2, I3, args.multi - 1, args.scale)

    if args.montage:
        write_buffer.put(np.concatenate((i1, i1), 1))
        for mid in output:
            mid = (mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)
            write_buffer.put(np.concatenate((i1, mid[:h, :w]), 1))
    else:
        write_buffer.put(i1)
        for mid in output:
            mid = F.interpolate(mid, (h, w), mode='bilinear', align_corners=False)
            mid = (mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)
            write_buffer.put(mid)
    pbar.update(1)
    i0, i1, i2 = i1, i2, i3
    I0, I1, I2 = I1, I2, I3

if args.montage:
    write_buffer.put(np.concatenate((i2, i2), 1))
else:
    write_buffer.put(i2)

while not write_buffer.empty():
    time.sleep(0.1)
pbar.close()
if not vid_out is None:
    vid_out.release()

# move audio to new video file if appropriate
if args.png == False and fpsNotAssigned == True and not args.video is None:
    try:
        transferAudio(args.video, vid_out_name)
    except:
        print("Audio transfer failed. Interpolated video will have no audio")
        targetNoAudio = os.path.splitext(vid_out_name)[0] + "_noaudio" + os.path.splitext(vid_out_name)[1]
        os.rename(targetNoAudio, vid_out_name)
