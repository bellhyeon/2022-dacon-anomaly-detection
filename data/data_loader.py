from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from .dataset import AnomalyDataset


def make_dataset(train_image_paths, train_labels, test_image_paths, args):

    train_img_paths, valid_img_paths, train_labels, valid_labels = train_test_split(
        train_image_paths,
        train_labels,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=train_labels,
    )

    train_dataset = AnomalyDataset(
        train_img_paths,
        train_labels,
        img_size=args.img_size,
        training=True,
        use_augmentation=True,
    )

    valid_dataset = AnomalyDataset(
        valid_img_paths,
        valid_labels,
        img_size=args.img_size,
        training=True,
        use_augmentation=False,
    )

    test_dataset = AnomalyDataset(
        test_image_paths,
        labels=None,
        img_size=args.img_size,
        training=False,
        use_augmentation=False,
    )

    return train_dataset, valid_dataset, test_dataset


def get_data_loader(
    train_dataset, valid_dataset, test_dataset, batch_size: int, num_workers: int
):
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, valid_loader, test_loader
