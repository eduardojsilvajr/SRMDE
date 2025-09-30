import argparse
import glob
import os


def main(args):
    txt_file = open(args.meta_info, 'w')
    # sca images
    img_paths_gt = sorted(glob.glob(os.path.join(args.input[0], '*')))
    img_paths_lq = sorted(glob.glob(os.path.join(args.input[1], '*')))

    assert len(img_paths_gt) == len(img_paths_lq), ('GT folder and LQ folder should have the same length, but got '
                                                    f'{len(img_paths_gt)} and {len(img_paths_lq)}.')

    for img_path_gt, img_path_lq in zip(img_paths_gt, img_paths_lq):
        # get the relative paths
        img_name_gt = os.path.relpath(img_path_gt, args.root[0])
        img_name_lq = os.path.relpath(img_path_lq, args.root[1])
        print(f'{img_name_gt}, {img_name_lq}')
        txt_file.write(f'{img_name_gt}, {img_name_lq}\n')


if __name__ == '__main__':
    """This script is used to generate meta info (txt file) for paired images.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        nargs='+',
        default=[r'C:\Users\Eduardo JR\Fast\Flickr2K\valid\HR', r'C:\Users\Eduardo JR\Fast\Flickr2K\valid\LR'],
        help='Input folder, should be [gt_folder, lq_folder]')
    parser.add_argument('--root', nargs='+', default=[r'C:\Users\Eduardo JR\Fast', r'C:\Users\Eduardo JR\Fast'], help='Folder root, will use the ')
    parser.add_argument(
        '--meta_info',
        type=str,
        default=r'D:\Documentos\OneDrive\Documentos\Mestrado\Super Resolução Imagem Única\SRIU\test_Flickr_pair.txt',
        help='txt path for meta info')
    args = parser.parse_args()

    assert len(args.input) == 2, 'Input folder should have two elements: gt folder and lq folder'
    assert len(args.root) == 2, 'Root path should have two elements: root for gt folder and lq folder'
    os.makedirs(os.path.dirname(args.meta_info), exist_ok=True)
    for i in range(2):
        if args.input[i].endswith('/'):
            args.input[i] = args.input[i][:-1]
        if args.root[i] is None:
            args.root[i] = os.path.dirname(args.input[i])

    main(args)
    
# default=[r'C:\Users\Eduardo JR\Fast\Flickr2K\train\HR', r'C:\Users\Eduardo JR\Fast\Flickr2K\train\LR'],
# default=[r'C:\Users\Eduardo JR\Fast\DIV2K\DIV2K_valid_HR', r'C:\Users\Eduardo JR\Fast\DIV2K\DIV2K_valid_LR_bicubic\X2']
