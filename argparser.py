import argparse

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--category', default='screw_bag',
                        help='category')
    parser.add_argument('-o', '--output_dir', default='./results/')
    parser.add_argument('-w', '--weights', default='models/teacher_medium.pth')
    parser.add_argument('-i', '--imagenet_train_path', default='./data/imagenet/train',)
    parser.add_argument('--mvtec_loco_path', default='./data/mvtec_loco'),
    parser.add_argument('--mvtec_loco_seg_path', default='./data/mvtec_loco_composition_maps/',)
    parser.add_argument('-t', '--train_steps', type=int, default=70000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--fusion_weights',
        default='1,1,1',
        help='Comma-separated img,mahalanobis,composition weights used for final-score fusion.'
    )
    parser.add_argument(
        '--save_branch_scores',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Save per-image branch scores for each evaluation checkpoint.'
    )
    parser.add_argument(
        '--split',
        choices=('test', 'validation'),
        default='test',
        help='Dataset split to evaluate in test_salad.py.'
    )
    parser.add_argument(
        '--checkpoint',
        choices=('final', 'best', 'tmp'),
        default='final',
        help='Checkpoint suffix to load in test_salad.py.'
    )
    parser.add_argument(
        '--composition_num_classes',
        type=int,
        default=6,
        help='Number of composition-map classes, including background.'
    )
    return parser.parse_args()
