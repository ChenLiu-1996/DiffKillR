from torch.utils.data import DataLoader
from datasets.synthetic import SyntheticDataset
from datasets.augmented import AugmentedDataset
from datasets.A28 import A28Dataset
from datasets.A28Axis import A28AxisDataset
# from datasets.tissuenet import TissueNetDataset
from datasets.MoNuSeg import MoNuSegDataset
from datasets.augmented_GLySAC import AugmentedGLySACDataset
from datasets.BCCD_augmented import BCCD_augmentedDataset
from utils.split import split_dataset
from utils.attribute_hashmap import AttributeHashmap
from utils.extend import ExtendedDataset


def prepare_dataset(config: AttributeHashmap):
    # Read dataset.
    if config.dataset_name == 'synthetic':
        dataset = SyntheticDataset(base_path=config.dataset_path,
                                   target_dim=config.target_dim)

    elif config.dataset_name == 'A28':
        aug_lists = config.aug_methods.split(',')
        dataset = A28Dataset(augmentation_methods=aug_lists,
                             base_path=config.dataset_path,
                             target_dim=config.target_dim,
                             n_views=config.n_views)

    elif config.dataset_name == 'A28Axis':
        aug_lists = config.aug_methods.split(',')
        dataset = A28AxisDataset(augmentation_methods=aug_lists,
                                 base_path=config.dataset_path,
                                 target_dim=config.target_dim,
                                 n_views=config.n_views)

    elif config.dataset_name == 'augmented':
        aug_lists = config.aug_methods.split(',')
        dataset = AugmentedDataset(augmentation_methods=aug_lists,
                                   base_path=config.dataset_path,
                                   target_dim=config.target_dim,
                                   has_labels=config.has_labels)
    # elif config.dataset_name == 'tissuenet':
    #     dataset = TissueNetDataset(base_path=config.dataset_path,
    #                                target_dim=config.target_dim)
    elif config.dataset_name == 'MoNuSeg':
        aug_lists = config.aug_methods.split(',')
        dataset_trainval = MoNuSegDataset(subset='train',
                                          no_background=config.no_background,
                                          augmentation_methods=aug_lists,
                                          organ=config.organ,
                                          base_path=config.dataset_path,
                                          target_dim=config.target_dim,
                                          n_views=config.n_views,
                                          percentage=config.percentage)
        dataset_test = MoNuSegDataset(subset='test',
                                      no_background=config.no_background,
                                      augmentation_methods=aug_lists,
                                      organ=config.organ,
                                      base_path=config.dataset_path,
                                      target_dim=config.target_dim,
                                      n_views=config.n_views,
                                      percentage=config.percentage)

    elif config.dataset_name == 'GLySAC':
        aug_lists = config.aug_methods.split(',')
        dataset = AugmentedGLySACDataset(augmentation_methods=aug_lists,
                                        base_path=config.dataset_path,
                                        target_dim=config.target_dim)
    elif config.dataset_name == 'BCCD':
        aug_lists = config.aug_methods.split(',')
        dataset = BCCD_augmentedDataset(augmentation_methods=aug_lists,
                                        base_path=config.dataset_path,
                                        target_dim=config.target_dim)
    else:
        raise ValueError(
            'Dataset not found. Check `dataset_name` in config yaml file.')

    if config.dataset_name in ['MoNuSeg']:
        dataset = dataset_trainval  # for compatibility
        ratios = [float(c) for c in config.train_val_test_ratio.split(':')]
        ratios = tuple([c / sum(ratios) for c in ratios])
        train_set, val_set = split_dataset(
            dataset=dataset_trainval, splits=ratios[:2], random_seed=config.random_seed)
        test_set = dataset_test
    else:
        ratios = [float(c) for c in config.train_val_test_ratio.split(':')]
        ratios = tuple([c / sum(ratios) for c in ratios])
        train_set, val_set, test_set = split_dataset(
            dataset=dataset, splits=ratios, random_seed=config.random_seed)

    if len(train_set) < 20 * config.batch_size:
        train_set = ExtendedDataset(train_set, desired_len=20 * config.batch_size)
    if len(val_set) < config.batch_size:
        val_set = ExtendedDataset(val_set, desired_len=2 * config.batch_size)
    if len(test_set) < config.batch_size:
        test_set = ExtendedDataset(test_set, desired_len=2 * config.batch_size)

    train_loader = DataLoader(dataset=train_set,
                              batch_size=config.batch_size,
                              shuffle=True,
                              num_workers=config.num_workers)
    val_loader = DataLoader(dataset=val_set,
                            batch_size=config.batch_size,
                            shuffle=False,
                            num_workers=config.num_workers)
    test_loader = DataLoader(dataset=test_set,
                             batch_size=config.batch_size,
                             shuffle=False,
                             num_workers=config.num_workers)

    return dataset, train_loader, val_loader, test_loader
