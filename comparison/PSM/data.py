import torch.utils.data as data
import os
from sklearn.model_selection import train_test_split


class Dataset(data.Dataset):
    def __init__(self, x , y ):
        self.input = x
        self.label = y

    def __len__(self):
        return len(self.input)

    def __getitem__(self, item):
        input = self.input[item]
        label = self.label[item]
        return input, label


class Data:
    def __init__(self, args, input_train, label_train, input_val, label_val, input_test, label_test):
        self.train = args.data_train
        self.test = args.data_test
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.train_loader = data.DataLoader(
            Dataset(input_train, label_train),
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=False,
            num_workers=args.n_threads,
            drop_last=True,
        )

        self.val_loader = data.DataLoader(
            Dataset(input_val, label_val),
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=False,
            num_workers=args.n_threads,
        )

        self.test_loader = data.DataLoader(
            Dataset(input_test, label_test),
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=False,
            num_workers=args.n_threads,
        )

def get_dataset(epoch, args):
    list_all = None
    image_list = None
    anno_list = None

    if not args.mode == 'train_second_stage':
        for i, j, k in os.walk(args.data_train + '/images'):
            image_list = k[:]
            image_list = sorted(image_list)

        for i, j, k in os.walk(args.data_train + '/masks'):
            anno_list  = k[:]
            anno_list  = sorted((anno_list))

        test_list = None
        for i, j, k in os.walk(args.data_test + '/images'):
            test_list = k[:]
            test_list = sorted(test_list)

        input_test = sorted([item for item in test_list if item.endswith('.png')])
        label_test = sorted([item for item in test_list if item.endswith('.png')])

    if args.mode in('train_second_stage', 'generate_voronoi', 'train_final_stage'):

        for i, j, k in os.walk('./data_%s/data_second_stage_train' % args.dataset_name):
            list_all = sorted(k[:])
        image_list = sorted([item for item in list_all if item.endswith('_original.png')])

        if args.mode == 'generate_voronoi':
            anno_list = sorted([item for item in list_all if item.endswith('_pos.png')])
        else:
            anno_list = sorted([item for item in list_all if item.endswith('_pos.png')])

        for i, j, k in os.walk('./data_%s/data_second_stage_test' % args.dataset_name):
            test_list = k[:]

        input_test = sorted([item for item in test_list if item.endswith('_original.png')])
        label_test = sorted([item for item in test_list if item.endswith('_gt.png')])


    input_train, input_val, label_train, label_val = train_test_split(image_list, anno_list,
                                                                      train_size=int(0.9 * len(image_list)),
                                                                      random_state=1)

    o = Data(args, input_train, label_train, input_val, label_val, input_test, label_test)

    return o
