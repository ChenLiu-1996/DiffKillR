from torch.utils.data import DataLoader
from datasets.synthetic import SyntheticDataset
from datasets.augmented import AugmentedDataset
from utils.split import split_dataset
from utils.attribute_hashmap import AttributeHashmap


def prepare_dataset(config: AttributeHashmap):
    # Read dataset.
    if config.dataset_name == 'synthetic':
        dataset = SyntheticDataset(base_path=config.dataset_path,
                                   target_dim=config.target_dim)
    elif config.dataset_name == 'augmented':
        aug_lists = config.aug_methods.split(',')
        dataset = AugmentedDataset(augmentation_methods=aug_lists,
                                    base_path=config.dataset_path,
                                    target_dim=config.target_dim)
    else:
        raise ValueError(
            'Dataset not found. Check `dataset_name` in config yaml file.')

    # Load into DataLoader
    ratios = [float(c) for c in config.train_val_test_ratio.split(':')]
    ratios = tuple([c / sum(ratios) for c in ratios])
    train_set, val_set, test_set = split_dataset(
        dataset=dataset, splits=ratios, random_seed=config.random_seed)

    train_loader = DataLoader(dataset=train_set,
                           batch_size=config.batch_size,
                           shuffle=True,
                           num_workers=config.num_workers)
    val_loader = DataLoader(dataset=val_set,
                         batch_size=len(val_set),
                         shuffle=False,
                         num_workers=config.num_workers)
    test_loader = DataLoader(dataset=test_set,
                          batch_size=len(test_set),
                          shuffle=False,
                          num_workers=config.num_workers)
    
    # NOTE:Make sure no leakage between train/val/test sets when sampling augmentation.
    # A hacky way, but works for now.
    img_path_to_split = {}
    for _, (_, _, _, _, img_path, _) in enumerate(train_set):
        img_path_to_split[img_path] = 'train'
    for _, (_, _, _, _, img_path, _) in enumerate(val_set):
        img_path_to_split[img_path] = 'val'
    for _, (_, _, _, _, img_path, _) in enumerate(test_set):
        img_path_to_split[img_path] = 'test'
    dataset._set_img_path_to_split(img_path_to_split=img_path_to_split)

    return dataset, train_loader, val_loader, test_loader
