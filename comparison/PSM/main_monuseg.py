import cv2
import numpy as np
from options import args
from utils.optimization import make_optimizer
from utils.generate_point_label import  peak_point
from utils.generate_voronoi import  create_Voronoi_label
import torch
import torch.nn as nn
import torch.nn.functional as F
import data
import os
import cv2
from tqdm import tqdm
from model.resunet import ResUNet34
import timm
import skimage
import torchvision.transforms as transforms
from metrics import metrics
import psm
import warnings
import random
warnings.filterwarnings('ignore')

torch.manual_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = args.set_gpu
EPOCH = args.epochs

def save_model(dict, epoch, name):
    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')
    torch.save(dict, f'checkpoint/{name}_{epoch}.pth')

def crop(x, size):
    if not size:
        return x
    a, c, h, w = x.shape
    H = int(h / 2 - size / 2)
    W = int(w / 2 - size / 2)
    o = x[:, :, H:H+size, W:W+size]
    if isinstance(o, torch.Tensor):
        o = o.contiguous()
    return o

def crop_random(x, size):
    a, c, h, w = x.shape
    xmin = random.randint(0, h-size)
    ymin = random.randint(0, w-size)

    o = x[:, :, xmin:xmin+size, ymin:ymin+size]
    o = o.contiguous()
    return o

def preProcess_train(x_fname_list, y_fname_list, args):
    '''
    Returned image in [0, 1], labels not much bigger than [0, 1].
    '''

    input_data = []
    label = []
    edge = []
    for i in range(len(x_fname_list)):
        x_fname = x_fname_list[i]
        y_fname = y_fname_list[i]

        assert x_fname[:16] == y_fname[:16]
        x_path = args.data_train + '/images/' + x_fname
        y_path = args.data_train + '/masks/' + y_fname
        if args.mode in ('train_second_stage', 'generate_voronoi', 'train_final_stage'):
            x_path = '/'.join(args.data_train.split('/')[:-1]) + '/data_second_stage_train/' + x_fname
            y_path = '/'.join(args.data_train.split('/')[:-1]) + '/data_second_stage_train/' + y_fname

            if args.mode == 'train_final_stage':
                path_edge = '/'.join(args.data_train.split('/')[:-1]) + '/data_second_stage_train/' + y_fname.split('_')[-2] + '_vor.png'

        x_img = cv2.cvtColor(cv2.imread(x_path, cv2.IMREAD_COLOR), code=cv2.COLOR_BGR2RGB)
        y_img = cv2.imread(y_path, cv2.IMREAD_GRAYSCALE)

        x_img, y_img = torch.from_numpy(x_img).unsqueeze(0), torch.from_numpy(y_img).unsqueeze(0).unsqueeze(3)
        input_data.append(x_img)
        label.append(y_img)
        if args.mode in ('train_final_stage'):
            edge_data = cv2.imread(path_edge, cv2.IMREAD_GRAYSCALE)
            edge_data = torch.from_numpy(edge_data).unsqueeze(0).unsqueeze(3)
            edge.append(edge_data)

    input_data, label = torch.cat(input_data, 0), torch.cat(label, 0)
    input_data, label = input_data.permute(0, 3, 1, 2).contiguous().float().to(device), label.permute(0, 3, 1, 2).contiguous().float().to(device)
    input_data, label = crop(input_data, args.crop_edge_size), crop(label, args.crop_edge_size)

    if args.mode == 'train_final_stage':
        edge_data = torch.cat(edge)
        edge_data = edge_data.permute(0, 3, 1, 2).contiguous().float().to(device)
        edge_data = crop(edge_data, args.crop_edge_size)

        label[label==255] = 2
        edge_data[edge_data==0] = 2
        edge_data[edge_data==255] = 0
        edge_data[edge_data==120] = 1
        assert int(torch.max(edge_data))<3, f"path: {path_edge}. edge_data:{torch.unique(edge_data)}"
        return input_data/255, label, edge_data

    return input_data/255, label/255


def preProcess_test(x_fname_list, y_fname_list, args):
    '''
    Returned image in [0, 1], labels not much bigger than [0, 1].
    '''
    input_data = []
    label = []
    for i in range(len(x_fname_list)):
        x_fname = x_fname_list[i]
        y_fname = y_fname_list[i]
        assert x_fname[:16] == y_fname[:16]
        if args.mode in ('train_second_stage', 'generate_voronoi', 'train_final_stage'):
            x_path = '/'.join(args.data_test.split('/')[:-1]) + '/data_second_stage_test/' + x_fname
            y_path = '/'.join(args.data_test.split('/')[:-1]) + '/data_second_stage_test/' + y_fname
        else:
            x_path = args.data_test + '/images/' + x_fname
            y_path = args.data_test + '/masks/' + y_fname

        x_img = cv2.cvtColor(cv2.imread(x_path, cv2.IMREAD_COLOR), code=cv2.COLOR_BGR2RGB)
        y_img = cv2.imread(y_path, cv2.IMREAD_GRAYSCALE)

        x_img, y_img = torch.from_numpy(x_img).unsqueeze(0), torch.from_numpy(y_img).unsqueeze(0).unsqueeze(3)
        input_data.append(x_img)
        label.append(y_img)

    input_data, label = torch.cat(input_data, 0), torch.cat(label, 0)
    input_data, label = input_data.permute(0, 3, 1, 2).contiguous().float().to(device), label.permute(0, 3, 1, 2).contiguous().float().to(device)
    input_data, label = crop(input_data, args.crop_edge_size), crop(label, args.crop_edge_size)

    return input_data/255, label/255


def trainer_selfsupervised(EPOCH, args, model, loss_func, optimizer):
    best_loss = np.inf

    for epoch in range(EPOCH):
        print(F'--EPOCH : {epoch} ')

        loss_epoch, num_items = 0, 0
        loader = data.get_monuseg(epoch, args)
        for x_fname_list, y_fname_list in tqdm(loader.train_loader, desc='training'):
            x_img, _ = preProcess_train(x_fname_list, y_fname_list, args)

            x_img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(x_img)
            x_rotate = transforms.functional.rotate(x_img, 180)

            output1 = model(x_img)
            output2 = model(x_rotate)

            loss = torch.sum(torch.abs(output1 - output2)) / output1.shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_epoch += loss.item()
            num_items += 1

        loss_mean = loss_epoch / num_items

        if loss_mean < best_loss:
            save_model(model.state_dict(), 'best', 'self_stage')

        print(F'loss:{loss_mean}')

        if (epoch + 1) % args.test_interval == 0:
            save_model(model.state_dict(), epoch, 'self_stage')


def trainer_selfsupervised_contrastive(EPOCH, args, model, loss_func, optimizer):
    for epoch in range(EPOCH):
        print(F'--EPOCH : {epoch} ')

        loss_epoch, num_items = 0, 0
        loader = data.get_monuseg(epoch, args)
        for x_fname_list, y_fname_list in tqdm(loader.train_loader, desc='training'):
            x_img, _ = preProcess_train(x_fname_list, y_fname_list, args)
            x_img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(x_img)

            bs = x_img.shape[0]
            if bs < 4:
                continue
            anchor = x_img[:3, :, :, :]
            positive = transforms.functional.rotate(anchor, 180)
            negative = x_img[3:, :, :, :]

            output1 = model(anchor.to(device).float())
            output2 = model(positive.to(device).float())
            output3 = model(negative.to(device).float())

            loss_sim = loss_func(output2, output1)
            loss_cont = loss_func(output3, output1)
            loss = torch.max(loss_sim - loss_cont + 10, torch.tensor([0]).to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_epoch += loss.item()
            num_items += 1

        loss_mean = loss_epoch / num_items
        print(F'loss:{loss_mean}')

        if (epoch + 1) % args.test_interval == 0:
            save_model(model.state_dict(), epoch, args.model)

def trainer_selfsupervised_random_rotate(EPOCH, args, model, loss_func, optimizer):
    for epoch in range(EPOCH):
        print(F'--EPOCH : {epoch} ')
        loss_list = []
        loader = data.get_monuseg(epoch, args)
        for x_fname_list, y_fname_list in tqdm(loader.train_loader, desc='training'):
            x_img, _ = preProcess_train(x_fname_list, y_fname_list, args)
            x_img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(x_img)

            # print(x.shape)
            n = random.randint(0, 3)
            x_rotate = transforms.functional.rotate(x_img, 90 * n)

            output2 = model(x_rotate)

            loss = loss_func(output2, torch.tensor([n]).to(device).float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss)

        loss_mean = sum(loss_list) / len(loss_list)
        print(F'loss:{loss_mean}')
        if (epoch + 1) % args.test_interval == 0:

            save_model(model.state_dict(), epoch, args.model)

def trainer_selfsupervised_simsiam(EPOCH, args, model, loss_func, optimizer):
    encoder = model
    predictor1 = nn.Linear(1000, 1).to(device)
    predictor2 = nn.Linear(1000, 1).to(device)
    model_whole = nn.Sequential(encoder, predictor1)
    for epoch in range(EPOCH):
        print(F'--EPOCH : {epoch} ')
        loss_list = []
        loader = data.get_monuseg(epoch, args)
        for x_fname_list, y_fname_list in tqdm(loader.train_loader, desc='training'):
            x_img, _ = preProcess_train(x_fname_list, y_fname_list, args)
            x_img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(x_img)

            n = random.randint(0, 3)
            x_rotate = transforms.functional.rotate(x_img, 90 * n)
            x_img, x_rotate = x_img.to(device), x_rotate.to(device)

            x1, x2 = encoder(x_img), encoder(x_rotate)
            z1, z2 = predictor1(x1), predictor1(x2)
            p1, p2 = predictor2(x1), predictor2(x2)

            def D(p, z): #negative cosine similarity
                z = z.detach()
                return torch.abs(p-z).sum(axis=1).mean()

            loss = D(p1, z2)/2 + D(p2, z1)/2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss)

        loss_mean = sum(loss_list) / len(loss_list)
        print(F'loss:{loss_mean}')
        if (epoch + 1) % args.test_interval == 0:

            save_model(model_whole.state_dict(), epoch, args.model)

def trainer_selfsupervised_mean_value(EPOCH, args, model, loss_func, optimizer):
    for epoch in range(EPOCH):
        print(F'--EPOCH : {epoch} ')
        loss_list = []
        loader = data.get_monuseg(epoch, args)
        for x_fname_list, y_fname_list in tqdm(loader.train_loader, desc='training'):
            x_img, _ = preProcess_train(x_fname_list, y_fname_list, args)
            x_img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(x_img)
            # print(x.shape)
            n = torch.mean(torch.reshape(x_img, (x_img.shape[0], -1)), dim=1).unsqueeze(1)

            output = model(x_img)

            loss = sum(torch.abs(output-n))/len(output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss)

        loss_mean = sum(loss_list) / len(loss_list)
        print(F'loss:{loss_mean}')
        if (epoch + 1) % args.test_interval == 0:

            save_model(model.state_dict(), epoch, args.model)


def trainer_second_stage(EPOCH, args, model, loss_func, optimizer):
    best_aji = 0
    best_epoch = 0
    best_model = 0

    for epoch in range(EPOCH):
        print(F'--EPOCH : {epoch} ')
        loss_list = []

        loader = data.get_monuseg(epoch, args)
        model.train()

        for x_fname_list, y_fname_list in tqdm(loader.train_loader, desc='training'):
            x_img, y_img = preProcess_train(x_fname_list, y_fname_list, args)
            x_img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(x_img)

            output = model(x_img)
            output = crop(output, args.crop_edge_size)

            # prob_maps = F.softmax(output, dim=1)
            log_prob_maps = F.log_softmax(output, dim=1)

            loss_all = loss_func(log_prob_maps, y_img.long().squeeze(1))

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

            loss_list.append(loss_all)

        loss_mean = sum(loss_list) / len(loss_list)

        print(F'loss: {loss_mean}')

        if (epoch + 1) % args.test_interval == 0 and (epoch +1) > 1:

            print('validating:')
            print('-----------------------')
            # model.eval()
            with torch.no_grad():
                loss_list = []

                dic = []

                for x_test_fname_list, y_test_fname_list in tqdm(loader.test_loader, desc='testing'):
                    x_test_img, y_test_img = preProcess_test(x_test_fname_list, y_test_fname_list, args)
                    x_test_img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(x_test_img)

                    output = model(x_test_img)
                    output = crop(output, args.crop_edge_size)
                    prob_maps = F.softmax(output, dim=1)
                    pred = np.argmax(prob_maps.cpu(), axis=1)

                    for i in range(pred.shape[0]):
                        label = y_test_img[i].cpu().permute(1, 2, 0).squeeze(2).long()
                        metric = metrics.compute_metrics(pred[i], label, ['p_F1', 'aji', 'iou'])
                        dic.append(metric)

                for key in dic[0].keys():
                    num = sum([item[key] for item in dic]) / len(dic)
                    var = np.var(np.array([item[key] for item in dic]))
                    print(F'{key}: {num} var: {var}')

                    if key == 'aji':
                        if num > best_aji:
                            best_aji = num
                            best_epoch = epoch
                            import copy
                            best_model = copy.deepcopy(model.state_dict())

        optimizer.schedule()
    save_model(best_model, 'best', 'second_stage')

    print(f'best-epoch: {best_epoch}, aji: {best_aji}')

def trainer_final_stage(EPOCH, args, model, loss_func, optimizer):
    best_aji = 0
    best_epoch = 0
    best_model = 0

    for epoch in range(EPOCH):
        print(F'--EPOCH : {epoch} ')
        loss_list = []
        loss_seg =[]
        loss_edge = []
        loader = data.get_monuseg(epoch, args)
        model.train()

        for x_fname_list, y_fname_list in tqdm(loader.train_loader, desc='training'):
            x_img, y_img, edge = preProcess_train(x_fname_list, y_fname_list, args)

            x_img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(x_img)

            output = model(x_img)
            output = crop(output, args.crop_edge_size)

            # prob_maps = F.softmax(output, dim=1)
            log_prob_maps = F.log_softmax(output, dim=1)

            loss1 = loss_func(log_prob_maps, y_img.long().squeeze(1))
            loss2 = loss_func(log_prob_maps, edge.long().squeeze(1))

            loss_all = loss1 + 1.5 * loss2

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

            loss_list.append(loss_all)
            loss_seg.append(loss1)
            loss_edge.append(loss2)

        loss_mean = sum(loss_list) / len(loss_list)
        loss_seg_mean = sum(loss_seg) / len(loss_seg)
        loss_edge_mean = sum(loss_edge) / len(loss_edge)

        print(F'loss: {loss_mean}')

        if (epoch + 1) % args.test_interval == 0 and (epoch +1) > 9:

            print('validating:')
            print('-----------------------')
            model.eval()
            with torch.no_grad():
                loss_list = []

                dic = []

                for x_test_fname_list, y_test_fname_list in tqdm(loader.test_loader, desc='testing'):
                    x_test_img, y_test_img = preProcess_test(x_test_fname_list, y_test_fname_list, args)
                    x_test_img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(x_test_img)

                    output = model(x_test_img)
                    output = crop(output, args.crop_edge_size)
                    prob_maps = F.softmax(output, dim=1)
                    pred = np.argmax(prob_maps.cpu(), axis=1)

                    for i in range(pred.shape[0]):
                        label = y_test_img[i].cpu().permute(1, 2, 0).squeeze(2).long()
                        metric = metrics.compute_metrics(pred[i], label, ['p_F1', 'aji', 'iou'])
                        dic.append(metric)


                for key in dic[0].keys():
                    num = sum([item[key] for item in dic]) / len(dic)
                    var = np.var(np.array([item[key] for item in dic]))
                    print(F'{key}: {num} var: {var}')

                    if key == 'aji':
                        if num > best_aji:
                            best_aji = num
                            best_epoch = epoch
                            import copy
                            best_model = copy.deepcopy(model.state_dict())

        optimizer.schedule()
    save_model(best_model, 'best', 'final_stage')

    print(f'best-epoch: {best_epoch}, aji: {best_aji}')

def test_stage(EPOCH, args, model):
    loader = data.get_monuseg(0, args)
    dic = []

    with torch.no_grad():
        for x_test_fname_list, y_test_fname_list in tqdm(loader.test_loader, desc='testing'):
            x_test_img, y_test_img = preProcess_test(x_test_fname_list, y_test_fname_list, args)
            x_test_img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(x_test_img)

            output = model(x_test_img)
            output = crop(output, args.crop_edge_size)
            prob_maps = F.softmax(output, dim=1)
            pred = np.argmax(prob_maps.cpu(), axis=1)

            for i in range(pred.shape[0]):
                label = y_test_img[i].cpu().permute(1, 2, 0).squeeze(2).long()
                metric = metrics.compute_metrics(pred[i], label, ['p_F1', 'aji', 'iou'])
                dic.append(metric)

                img_save = pred[i].detach().cpu().numpy().astype(np.uint8)
                img_save_instances = skimage.measure.label(img_save)
                label_instance = skimage.measure.label(label)

                img_save = (skimage.color.label2rgb(img_save_instances)*255).astype(np.uint8)
                label_save = (skimage.color.label2rgb(label_instance)*255).astype(np.uint8)
                img_save[img_save_instances==0, :] = 0
                label_save[label_instance==0, :] = 0

                img_save = cv2.cvtColor(img_save, cv2.COLOR_RGB2BGR)
                label_save = cv2.cvtColor(label_save, cv2.COLOR_RGB2BGR)
                input_save = cv2.cvtColor(np.uint8(x_test_img[i].permute(1, 2, 0).detach().cpu().numpy() * 255), cv2.COLOR_RGB2BGR)
                os.makedirs('./check_final_results/', exist_ok=True)
                cv2.imwrite('./check_final_results/' + x_test_fname_list[i].replace('.png', '_pred.png'), img_save)
                cv2.imwrite('./check_final_results/' + x_test_fname_list[i].replace('.png', '_label.png'), label_save)
                cv2.imwrite('./check_final_results/' + x_test_fname_list[i].replace('.png', '_input.png'), input_save)

    for key in dic[0].keys():
        num = sum([i[key] for i in dic]) / len(dic)
        var = np.var(np.array([i[key] for i in dic]))
        print(F'{key}: {num} var: {var}')


def generate_voronoi_label( args, model):
    loader = data.get_monuseg(0, args)
    model.eval()
    for sub_loader in [loader.train_loader, loader.val_loader, loader.test_loader]:
        for x_fname_list, y_fname_list in tqdm(sub_loader, desc='testing'):
            if sub_loader == loader.test_loader:
                x_img, _ = preProcess_test(x_fname_list, y_fname_list, args)
            else:
                x_img, _  = preProcess_train(x_fname_list, y_fname_list, args)
            x_img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(x_img)

            output = model(x_img)
            output = crop(output, args.crop_edge_size)

            prob_maps = F.softmax(output, dim=1)
            pred = np.argmax(prob_maps.cpu().detach().numpy(), axis=1)

            for i in range(pred.shape[0]):
                if sub_loader == loader.test_loader:
                    path = '/'.join(args.data_test.split('/')[:-1]) + '/data_second_stage_test/' + y_fname_list[i]
                else:
                    path = '/'.join(args.data_train.split('/')[:-1]) + '/data_second_stage_train/' + y_fname_list[i]

                label = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                point_list, bp, prob = peak_point(prob_maps.detach().cpu()[i][1].numpy(), 20, 0.6)

                # if sub_loader == loader.test_loader:
                #     import pdb
                #     pdb.set_trace()

                # generate voronoilabel
                if point_list.shape[0] < 4:
                   point_list = np.random.randint(0, 255, (4, 2))
                voronoi_label = create_Voronoi_label(point_list, label.shape)

                if sub_loader == loader.test_loader:
                    cv2.imwrite('/'.join(args.data_test.split('/')[:-1]) + '/data_second_stage_test/' + x_fname_list[i].split('_')[-2] + '_vor.png', voronoi_label)
                else:
                    cv2.imwrite('/'.join(args.data_train.split('/')[:-1]) +  '/data_second_stage_train/' + x_fname_list[i].split('_')[-2] + '_vor.png', voronoi_label)

                fig_for_save = x_img[i].cpu().permute(1, 2, 0).contiguous().numpy()
                for j in range(point_list.shape[0]):
                    cv2.line(fig_for_save, (point_list[j][1] - 3, point_list[j][0]),
                             (point_list[j][1] + 3, point_list[j][0]), color=(0, 0, 255), thickness=1)
                    cv2.line(fig_for_save, (point_list[j][1], point_list[j][0] - 3),
                             (point_list[j][1], point_list[j][0] + 3), color=(0, 0, 255), thickness=1)

                if not os.path.exists('./data/voronoi'):
                    os.mkdir('./data/voronoi')

                cv2.imwrite('./data/voronoi/' + x_fname_list[i].split('_')[-2] + '_point.png', cv2.cvtColor(np.uint8(fig_for_save*255), cv2.COLOR_RGB2BGR))
                cv2.imwrite('./data/voronoi/' + x_fname_list[i].split('_')[-2] + '_prob.png', cv2.cvtColor(np.uint8(prob*255), cv2.COLOR_RGB2BGR))

    print('end')

def fully_supervised(EPOCH, args, model, optimizer, loss_func):
    loader = data.get_monuseg(0, args)

    for epoch in range(EPOCH):
        model.train()
        #loader = data.get_ten_fold_data_monuseg(epoch, args)
        for x_fname_list, y_fname_list in tqdm(loader.train_loader, desc='training'):
            x_img, y_img = preProcess_train(x_fname_list, y_fname_list, args)
            x_img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(x_img)

            output = model(x_img)
            '''
            t= x1[0].cpu().numpy().transpose(1, 2, 0)
            io.imsave('show0.png', x1[0].cpu().numpy().transpose(1, 2, 0))
            io.imsave('show1.png', (y[0].cpu().numpy()*255).transpose(1, 2, 0))
            import sys
            sys.exit()
            '''

            # prob_maps = F.softmax(output, dim=1)
            log_prob_maps = F.log_softmax(output, dim=1)

            loss = loss_func(log_prob_maps, y_img.long().squeeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % args.test_interval == 0:
            with torch.no_grad():

                dic = []

                for x_test_fname_list, y_test_fname_list in tqdm(loader.test_loader, desc='testing'):
                    x_test_img, y_test_img = preProcess_test(x_test_fname_list, y_test_fname_list, args)
                    x_test_img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(x_test_img)

                    output = model(x_test_img)

                    prob_maps = F.softmax(output, dim=1)
                    pred = np.argmax(prob_maps.detach().cpu(), axis=1)

                    for i in range(pred.shape[0]):
                        label = y_test_img[i].cpu().permute(1, 2, 0).squeeze(2).long()
                        metric = metrics.compute_metrics(pred[i], label, ['p_F1', 'aji', 'iou'])
                        dic.append(metric)

                        os.makedirs('./final_result/')
                        cv2.imwrite('./final_result/' + x_test_fname_list[i], cv2.cvtColor(np.uint8(pred[i].detach().cpu().numpy() * 255), cv2.COLOR_RGB2BGR))

                for key in dic[0].keys():
                    num = sum([i[key] for i in dic]) / len(dic)
                    var = np.var(np.array([i[key] for i in dic]))
                    print(F'{key}: {num} var: {var}')



if __name__ == "__main__":
    #loader = data.Data(args)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if args.mode == 'train_base':
        args.batch_size=1
        model = timm.create_model('res2net101_26w_4s', num_classes=1, pretrained=False).to(device)
        optimizer = make_optimizer(args, model)
        loss_func = nn.MSELoss()
        model.train()
        trainer_selfsupervised(EPOCH, args, model, loss_func, optimizer)

    elif args.mode == 'train_contrastive':
        model = timm.create_model('res2net101_26w_4s', num_classes=1, pretrained=False).to(device)
        optimizer = make_optimizer(args, model)
        loss_func = nn.MSELoss()
        model.train()
        trainer_selfsupervised_contrastive(EPOCH, args, model, loss_func, optimizer)

    elif args.mode == 'train_random_rotate':
        model = timm.create_model('res2net101_26w_4s', pretrained=False).to(device)
        optimizer = make_optimizer(args, model)
        loss_func = nn.MSELoss()
        model.train()
        trainer_selfsupervised_random_rotate(EPOCH, args, model, loss_func, optimizer)

    elif args.mode == 'train_simsiam':
        model = timm.create_model('res2net101_26w_4s', pretrained=False).to(device)

        optimizer = make_optimizer(args, model)
        loss_func = nn.MSELoss()
        model.train()
        trainer_selfsupervised_simsiam(EPOCH, args, model, loss_func, optimizer)

    elif args.mode == 'train_mean_value':
        model = timm.create_model('res2net101_26w_4s', pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 1)
        model = model.to(device)
        optimizer = make_optimizer(args, model)
        loss_func = nn.MSELoss()
        model.train()
        trainer_selfsupervised_mean_value(EPOCH, args, model, loss_func, optimizer)

    elif args.mode == 'generate_label':
        print('testing:')
        print('-----------------------')
        model = timm.create_model('res2net101_26w_4s', num_classes=1, pretrained=True).to(device)

        model.load_state_dict(torch.load('./checkpoint/self_stage_best.pth', map_location=device))
        optimizer = make_optimizer(args, model)

        loader = data.get_monuseg(0, args)
        model.eval()

        loss_list = []

        for step, (x_fname_list, y_fname_list) in enumerate(tqdm(loader.test_loader)):
            for i in range(len(x_fname_list)):
                psm.psm_for_seg(x_fname_list[i], y_fname_list[i], model, args, 'test_set', device)

        for step, (x_fname_list, y_fname_list) in enumerate(tqdm(loader.train_loader)):
            for i in range(len(x_fname_list)):
                psm.psm_for_seg(x_fname_list[i], y_fname_list[i], model, args, 'train_set', device)

        for step, (x_fname_list, y_fname_list) in enumerate(tqdm(loader.val_loader)):
            for i in range(len(x_fname_list)):
                psm.psm_for_seg(x_fname_list[i], y_fname_list[i], model, args, 'train_set', device)

        print('end')

    elif args.mode == 'train_second_stage':
        model = ResUNet34(pretrained=True).to(device)
        optimizer = make_optimizer(args, model)
        #loader = data.get_ten_fold_data_monuseg(0, args)
        loss_func = torch.nn.NLLLoss(ignore_index=2).to(device)
        trainer_second_stage(EPOCH, args, model, loss_func, optimizer)

    elif args.mode == 'generate_voronoi':
        model = ResUNet34(pretrained=True).to(device)
        model.load_state_dict(torch.load('./checkpoint/second_stage_best.pth', map_location=device))
        generate_voronoi_label(args, model)

    elif args.mode == 'train_final_stage':
        model = ResUNet34(pretrained=True).to(device)
        optimizer = make_optimizer(args, model)
        loss_func = torch.nn.NLLLoss(ignore_index=2).to(device)
        trainer_final_stage(EPOCH, args, model, loss_func, optimizer)

    elif args.mode == 'test':
        model = ResUNet34(pretrained=True).to(device)
        # Second stage gives better result than final stage.
        best_path = './checkpoint/second_stage_best.pth'
        model.load_state_dict(torch.load(best_path, map_location=device))
        model.eval()
        test_stage(EPOCH, args, model)

    elif args.mode == 'fully-supervised':
        model = ResUNet34(pretrained=True).to(device)
        optimizer = make_optimizer(args, model)
        loss_func = torch.nn.NLLLoss(ignore_index=2).to(device)
        model.train()
        fully_supervised(EPOCH, args, model, optimizer, loss_func)

    else:
        raise NotImplementedError(F"process not found {args.mode}.")

