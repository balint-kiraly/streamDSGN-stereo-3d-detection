import json
import argparse
from pathlib import Path


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument(
        '--root_dir', type=str, default='./data/kitti_tracking', help='root dir for kitti_tracking dataset')
    parser.add_argument(
        '--sample_stride', type=int, default=1, help='sample stride for kitti_tracking dataset')
    parser.add_argument(
        '--len_frames', type=int, default=40, help='len of each frame patch')
    parser.add_argument('--sample_mode', type=int, default=3,
                        help='len of each frame patch')
    parser.add_argument('--save_split', action='store_true', default=False)
    args = parser.parse_args()

    args.frame_tag = {
        'prev2': -2,
        'prev': -1,
        'token': 0,
        'next': 1
    }

    if args.sample_mode == 0:
        args.sample_list = ['token']
    elif args.sample_mode == 1:
        args.sample_list = ['token', 'next']
    elif args.sample_mode == 2:
        args.sample_list = ['token', 'prev', 'next']
    elif args.sample_mode == 3:
        args.sample_list = ['token', 'prev2', 'prev', 'next']
    else:
        raise NotImplementedError

    return args


def split_patch_in_scene(file_list):
    if len(file_list) < 2*args.len_frames:
        return [file_list[:len(file_list) // 2], file_list[len(file_list) // 2:]]

    n = len(file_list) // args.len_frames - 1
    result = [
        file_list[i*args.len_frames:(i+1)*args.len_frames] for i in range(n)]
    if len(file_list) % args.len_frames != 0:
        result.append(file_list[n*args.len_frames:])
    return result


def split_files(root_dir, mode='train'):
    dataset_split = 'training' if mode == 'train' or mode == 'val' else 'testing'
    scenes_list = sorted(
        [x.stem for x in (root_dir / 'training' / 'calib').iterdir()])

    ret_list = []
    for scene in scenes_list:
        lidar_dir = root_dir / dataset_split / 'velodyne' / scene
        file_list = sorted([x.stem for x in (lidar_dir).glob('*.bin')])

        file_split_list = split_patch_in_scene(file_list)
        split_id_list = [i for i in range(len(file_split_list))]
        file_split_list = file_split_list[::
                                          2] if mode == 'train' else file_split_list[1::2]
        split_id_list = split_id_list[::2] if mode == 'train' else split_id_list[1::2]
        
        for idx, sub_file_list in enumerate(file_split_list):
            min_file_ind = min(sub_file_list)
            max_file_ind = max(sub_file_list)

            for _, file_ind in enumerate(sub_file_list):
                image_2_path = Path(
                    root_dir / dataset_split / 'image_02' / scene / (file_ind + '.png'))
                assert image_2_path.exists()
                image_3_path = Path(
                    root_dir / dataset_split / 'image_03' / scene / (file_ind + '.png'))
                assert image_3_path.exists()

                prev2_name = f'{int(file_ind) - 2 * args.sample_stride:06d}' if int(
                    file_ind) - 2 * args.sample_stride >= int(min_file_ind) else ''
                prev_name = f'{int(file_ind) - args.sample_stride:06d}' if int(
                    file_ind) - args.sample_stride >= int(min_file_ind) else ''
                next_name = f'{int(file_ind) + args.sample_stride:06d}' if int(
                    file_ind) + args.sample_stride <= int(max_file_ind) else ''

                # scene 0001 frame 000177-000180 files is not exist
                prev2_name = prev2_name if prev2_name in sub_file_list else ''
                prev_name = prev_name if prev_name in sub_file_list else ''
                next_name = next_name if next_name in sub_file_list else ''

                file_dict = {'scene': '{}_{}'.format(scene, split_id_list[idx]), 'frame_tag': {}}

                for tag in args.sample_list:
                    file_num = int(file_ind) + \
                        args.frame_tag[tag] * args.sample_stride
                    tag_name = f'{file_num:06d}' if file_num >= int(
                        min_file_ind) else ''
                    tag_name = tag_name if tag_name in sub_file_list else ''
                    file_dict['frame_tag'][tag] = tag_name
                ret_list.append(file_dict)

    save_tag = '_'.join(args.sample_list)
    save_dir = Path(
        root_dir / f'frame_stride_{args.sample_stride}-len_frames_{args.len_frames}-{save_tag}' / 'ImageSets')
    save_dir.mkdir(exist_ok=True, parents=True)
    if args.save_split:
        with open((save_dir / f'{mode}.json'), 'w') as f:
            json.dump(ret_list, f, ensure_ascii=False, indent=2)
    return ret_list, save_dir


def get_sample_split(file_list, save_dir, mode='train'):
    with open((save_dir / f'{mode}_sample.json'), 'w') as f:
        json.dump(file_list[-397:-357], f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    args = parse_config()
    root_dir = Path(args.root_dir)

    train_files, save_dir = split_files(root_dir, mode='train')
    val_files, _ = split_files(root_dir, mode='val')

    get_sample_split(train_files, save_dir=save_dir, mode='train')
    get_sample_split(val_files, save_dir=save_dir, mode='val')
