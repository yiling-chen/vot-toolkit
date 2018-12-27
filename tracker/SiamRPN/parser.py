from __future__ import print_function
from os import makedirs
from os.path import exists, join
from glob import glob
from shutil import copyfile
import json
import cv2


class JigsawJsonParser(object):
    def __init__(self, json_file):
        assert exists(json_file)
        with open(json_file, 'r') as f:
            data = f.read()
        self.metadata = json.loads(data)

    def get_fps(self):
        return self.metadata['annotations'][0]['frameMethod']['fps']

    def num_of_tracklets(self):
        return len(self.metadata['annotations'][0]['data']['objects'])

    def length_of_tracklet(self, idx):
        return len(self.metadata['annotations'][0]['data']['objects'][idx]['frames'])

    def get_frame_number(self, idx, frame_idx):
        return self.metadata['annotations'][0]['data']['objects'][idx]['frames'][frame_idx]['frameNumber']

    def get_bounding_box(self, idx, frame_idx):
        bbox = self.metadata['annotations'][0]['data']['objects'][idx]['frames'][frame_idx]['boundingBox']
        return bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']


class JigsawSequenceProcessor(object):
    def __init__(self, folder, json_file):
        self.parser = JigsawJsonParser(json_file)
        self.folder = folder
        self._get_sequence_info()

    def _get_sequence_info(self):
        filenames = glob(join(self.folder, '*.jpg'))
        self.img_prefix = filenames[0].split('--')[0]
        img = cv2.imread(filenames[0])
        self.height, self.width = img.shape[:2]
        # print(self.img_prefix)
        # print(self.width, self.height)

    def extract_tracklet(self, prefix, idx):
        print(self.parser.length_of_tracklet(idx))

        # create output folder if not exist
        output_folder = join(self.folder, prefix + str(idx))
        if not exists(output_folder):
            makedirs(output_folder)
            makedirs(join(output_folder, 'color'))
        # create groundtruth.txt
        with open(join(output_folder, 'groundtruth.txt'), 'w') as f:
            start_idx = 1
            for i in range(self.parser.length_of_tracklet(idx)):
                # convert bounding box annotation
                frm_no = self.parser.get_frame_number(idx, i)
                bbox = self.parser.get_bounding_box(idx, i)
                left = bbox[0] * float(self.width)
                top = bbox[1] * float(self.height)
                width = (bbox[2] - bbox[0]) * self.width
                height = (bbox[3] - bbox[1]) * self.height
                # print(left, bbox[0], self.width)
                f.writelines("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(left, top, width, height))
                # copy and rename image
                src_filename = self.img_prefix + '--{:03d}.jpg'.format(frm_no)
                if not exists(src_filename):
                    print('Error: source file not exists')
                dst_filename = join(output_folder, 'color/{:08d}.jpg'.format(start_idx))
                copyfile(src_filename, dst_filename)
                start_idx += 1

        with open(join(output_folder, 'sequence'), 'w') as f:
            f.write('name={:s}\n'.format(prefix+str(idx)))
            f.write('fps={:d}\n'.format(self.parser.get_fps()))
            f.write('format=default\n')
            f.write('channels.color=color/%08d.jpg')


    def extract_all_tracklet(self, prefix):
        for i in range(self.parser.num_of_tracklets()):
            self.extract_tracklet(prefix, i)


if __name__ == "__main__":
    processor = JigsawSequenceProcessor(
        'sequences/sequence-1/', 
        'sequences/sequence-1/vlc-record-2018-08-29-11h48m10s-108NE-MAIN-2018-08-28-1538-1638.mp4-.zip.judgements.json'
    )
    processor.extract_all_tracklet('bellevue-')
