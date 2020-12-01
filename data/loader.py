from torch.utils.data import DataLoader
from data.loader_utils import ModifiedDataloader
from data.trajectories import TrajectoryDataset, seq_collate
from data.trajectories_scene import TrajectoryDatasetEval, seq_collate_eval


def data_loader(args, config, cnn, phase, logger=None):
    if phase == "test":
        collate_fn = seq_collate_eval
        dataset_fn = TrajectoryDatasetEval
        data_augmentation = 0
        shuffle = True
        batch_size = 1
        skip = args.skip
        max_num = None

    elif phase == "train":
        if config.social_pooling:
            collate_fn = seq_collate_eval
            dataset_fn = TrajectoryDatasetEval
            data_augmentation = args.data_augmentation
            shuffle = True
            batch_size = 16
            skip = args.skip
            max_num = args.max_num
        else:
            collate_fn = seq_collate
            dataset_fn = TrajectoryDataset
            data_augmentation = args.data_augmentation
            shuffle = True
            batch_size = args.batch_size
            skip = args.skip
            max_num = args.max_num

    elif phase == "val":
        collate_fn = seq_collate_eval
        dataset_fn = TrajectoryDatasetEval
        data_augmentation = 0
        shuffle = True
        batch_size = 128
        skip = args.skip
        max_num = args.max_num
    else:
        raise AssertionError('"phase" must be either train, val or test.')

    dset = dataset_fn(
        dataset_name=args.dataset_name,
        phase=phase,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        data_augmentation=data_augmentation,
        skip=skip,
        debug=args.debug,
        cnn=cnn,
        max_num=max_num,
        load_occupancy=config.load_occupancy,
        logger=logger,
        margin_in=config.margin_in,
        margin_out=config.margin_out,
        scaling_small=config.scaling_small,
        scaling_tiny=config.scaling_tiny,
        margin_tiny=config.margin_tiny

    )
    if phase == "train":
        loader = ModifiedDataloader(
            dset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=args.loader_num_workers,
            collate_fn=collate_fn
        )
    else:
        loader = DataLoader(
            dset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=args.loader_num_workers,
            collate_fn=collate_fn
        )

    logger.info("Loading %s set: %s samples" % (phase, dset.__len__()))

    return dset, loader


