from __future__ import print_function
from argparse import ArgumentParser
import sys
sys.path.insert(0, 'src')
import os, random, subprocess, shutil
from utils import save_img, get_img, exists, list_files
from moviepy.video.io.VideoFileClip import VideoFileClip
import moviepy.video.io.ffmpeg_writer as ffmpeg_writer
import pdb

TMP_DIR = '.fns_frames_%s/' % random.randint(0,99999)
DEVICE = '/gpu:0'
BATCH_SIZE = 1

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        dest='checkpoint', help='checkpoint directory or .ckpt file',
                        metavar='CHECKPOINT', required=True)

    parser.add_argument('--in-path', type=str,
                        dest='in_path', help='in video path',
                        metavar='IN_PATH', required=True)
    
    parser.add_argument('--out-path', type=str,
                        dest='out', help='path to save processed video to',
                        metavar='OUT', required=True)
    
    parser.add_argument('--tmp-dir', type=str, dest='tmp_dir',
                        help='tmp dir for processing', metavar='TMP_DIR',
                        default=TMP_DIR)

    parser.add_argument('--device', type=str, dest='device',
                        help='device for eval. CPU discouraged. ex: \'/gpu:0\'',
                        metavar='DEVICE', default=DEVICE)

    parser.add_argument('--batch-size', type=int,
                        dest='batch_size',help='batch size for eval. default 1.',
                        metavar='BATCH_SIZE', default=BATCH_SIZE)

    parser.add_argument('--no-disk', type=bool, dest='no_disk',
                        help='Don\'t save intermediate files to disk. Default False',
                        metavar='NO_DISK', default=False)
    return parser

def ffwd_video(path_in, path_out, checkpoint_dir, device_t='/gpu:0', batch_size=4):
    video_clip = VideoFileClip(path_in, audio=False)
    video_writer = ffmpeg_writer.FFMPEG_VideoWriter(path_out, video_clip.size, video_clip.fps, codec="libx264",
                                                    preset="medium", bitrate="2000k",
                                                    audiofile=path_in, threads=None,
                                                    ffmpeg_params=None)

    g = tf.Graph()
    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True
    with g.as_default(), g.device(device_t), \
            tf.Session(config=soft_config) as sess:
        batch_shape = (batch_size, video_clip.size[1], video_clip.size[0], 3)
        img_placeholder = tf.placeholder(tf.float32, shape=batch_shape,
                                         name='img_placeholder')

        preds = transform.net(img_placeholder)
        saver = tf.train.Saver()
        if os.path.isdir(checkpoint_dir):
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise Exception("No checkpoint found...")
        else:
            saver.restore(sess, checkpoint_dir)

        X = np.zeros(batch_shape, dtype=np.float32)

        def style_and_write(count):
            for i in range(count, batch_size):
                X[i] = X[count - 1]  # Use last frame to fill X
            _preds = sess.run(preds, feed_dict={img_placeholder: X})
            for i in range(0, count):
                video_writer.write_frame(np.clip(_preds[i], 0, 255).astype(np.uint8))

        frame_count = 0  # The frame count that written to X
        for frame in video_clip.iter_frames():
            X[frame_count] = frame
            frame_count += 1
            if frame_count == batch_size:
                style_and_write(frame_count)
                frame_count = 0

        if frame_count != 0:
            style_and_write(frame_count)

        video_writer.close()


def check_opts(opts):
    exists(opts.checkpoint)
    exists(opts.out)

def main():
    parser = build_parser()
    opts = parser.parse_args()
    ffwd_video(opts.in_path, opts.out, opts.checkpoint, opts.device, opts.batch_size)

 
if __name__ == '__main__':
    main()


