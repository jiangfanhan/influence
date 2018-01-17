import torch
from torch.autograd import grad ,Variable
from torchvision import models, transforms, datasets
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.optim as optim
import argparse
import os
from PIL import Image
import time
from skimage import io
import torch.nn.functional as F
from torch.utils.data import DataLoader
from folder import ImageFolder
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import numpy as np
parser = argparse.ArgumentParser(description='train imagenet or webvision')
parser.add_argument('--dataset', default='imagenet', type=str, help='which dataset to use imagenet or webvision')
parser.add_argument('--lr', default=0.001, type=float, help='the learning rate')
parser.add_argument('--batch_size', default=64, type=int, help='mini batch size')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--pretrained', action='store_true', help='use pretrained model')
parser.add_argument('--epochs', default=50, type=int, help='epochs to train')
parser.add_argument('--resume', action='store_false', help='if need use ckpt' )
parser.add_argument('--ckpt_path',default='/home/jfhan/infl_test/ckpt-3000-epoch50',
                    type=str, help='path to loaded checkpoint')
parser.add_argument('--lr_decay', default=0.1, type=float, help='learning rate decay factor')
parser.add_argument('--decay_epochs', default=15, type=int, help='frequency to decay the lr')
parser.add_argument('--validation',action='store_true', help='turn the model to validation mode')
parser.add_argument('--test_num', default='0', type=int, help='the picture want to test')
parser.add_argument('--iteration',default=200, type=int, help='repeat times')
parser.add_argument('--damping',default=0.01, type=float, help='the damping term')
parser.add_argument('--scale', default=10 ,type=int, help='scale the hessian'  )
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
net = models.alexnet(pretrained=True)
net.classifier._modules['6'] = nn.Linear(4096, 2)

transform_train = transforms.Compose([transforms.Scale(256),
                                transforms.RandomCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])
                                ])
transform_val = transforms.Compose([transforms.Scale(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])
                                ])

data_dir = 'hymenoptera_data'

train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform_train)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

train_infl = ImageFolder(os.path.join(data_dir, 'train'), transform=transform_train)

loader_hess = DataLoader(train_infl, batch_size=args.batch_size, shuffle=True, num_workers=10)

loader_infl = DataLoader(train_infl, batch_size=1, shuffle=False)

val_dataset = ImageFolder(os.path.join(data_dir, 'val'), transform=transform_val)

val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

file_time = time.strftime('%Y-%m-%d-%H:%M:%S')
def train_model(model, criterion, optimizer ,scheduler ,start_iter, best_acc ,start_epoch, num_epochs):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    subtotal = 0
    to_num = start_iter
    since = time.time()
    remain = start_epoch % scheduler.step_size
    for i in range(remain):
        scheduler.step()
    for iter_epoch in range(num_epochs):
        epoch = start_epoch + iter_epoch
        scheduler.step()
        for num , (image_batched,label_batched)in enumerate(train_loader):
            if use_cuda:
                image_batched, label_batched = image_batched.cuda(), label_batched.cuda()
            image_batched, label_batched = Variable(image_batched), Variable(label_batched)
            optimizer.zero_grad()
            outputs =model(image_batched)
            loss = criterion(outputs, label_batched)
            loss.backward()
            optimizer.step()
            train_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            total += label_batched.size(0)
            correct += predicted.eq(label_batched.data).cpu().sum()
            subtotal +=label_batched.size(0)
            to_num +=1
            if to_num % 100 == 0:
                time1 = time.time()
                duration = time1 - since
                acc_top1 = 100.*correct/subtotal
                print ('[{} {}] Loss: {:.3f} | Acc_top1: {:.3f}  ({}/{})'.format(
                    to_num, args.batch_size*(to_num), train_loss/100, acc_top1,  correct, subtotal))
                print ('sample/second = {}'.format(args.batch_size* 100 / duration))
                # print >> f, ('[{} {}] Loss: {:.3f} | Acc_top1: {:.3f}  ({}/{})'.format(
                #     to_num, args.batch_size*(to_num), train_loss/100, acc_top1,  correct, subtotal))
                # print >> f, ('sample/second = {}'.format(args.batch_size * 100 / duration))
                correct = 0
                subtotal = 0
                train_loss = 0.0
                since = time1
                if (best_acc < acc_top1 or to_num % 1000 == 0):
                    if best_acc < acc_top1:
                        best_acc = acc_top1
                    print 'saving..'
                    state = {
                        'state_dict': net.state_dict(),
                        'acc': acc_top1,
                        'num_iter' : to_num,
                        'learning_rate': optimizer.param_groups[0]['lr'],
                        'epochs' :  epoch
                        }
                    if not os.path.isdir('checkpoint/{}-{}-{}'.format(args.dataset, args.lr, file_time)):
                        os.mkdir('checkpoint/{}-{}-{}'.format(args.dataset, args.lr, file_time))
                    torch.save(state, './checkpoint/{}-{}-{}/ckpt-{}-epoch{}'.format(
                        args.dataset, args.lr, file_time, to_num, epoch+1))



best_acc = 0.0
start_iter = 0
lr = args.lr
start_epoch = 0
if use_cuda:
    net.cuda()
    # net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=args.momentum, weight_decay=0.0005)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_epochs, gamma=args.lr_decay)

if args.resume:
    print '=>loading from checkpoint from {}'.format(args.ckpt_path)
    checkpoint = torch.load(args.ckpt_path)
    state_dict = checkpoint['state_dict']
    net.load_state_dict(state_dict)
    best_acc = checkpoint['acc']
    start_iter = checkpoint['num_iter']
    lr = checkpoint['learning_rate']
    start_epoch = checkpoint['epochs']

def get_stest(num):
    i = 0
    for img, label, path in val_loader:
        if i == num:
            if use_cuda:
                img, label = img.cuda(), label.cuda()
            img, label = Variable(img, volatile=False), Variable(label, volatile=False)
            output = net(img)
            loss = criterion(output, label)
            pa,= path
            return grad(loss, net.parameters(), create_graph=True), pa
        else:
            i += 1
            continue
def hvp(loss, para, s_last):
    grad1 = grad(loss, para, create_graph=True)
    grad_temp = 0
    for gr , vi in zip(grad1, s_last):
        grad_temp += torch.sum(gr * vi)
    return grad(grad_temp, para, create_graph=True)

def s_test(vector):
    time_start = time.time()
    h_es = vector

    for epoch in range(args.iteration):
        for img, label , _ in loader_hess:
            if use_cuda:
                img, label = img.cuda(), label.cuda()
            img, label = Variable(img, volatile=False), Variable(label, volatile=False)
            output = net(img)
            loss = criterion(output, label)
            grad1 = grad(loss, list(net.parameters()), create_graph=True)
            grad_temp = 0
            for gr, vi in zip(grad1, h_es):
                grad_temp += torch.sum(gr * vi)
            hv = grad(grad_temp, list(net.parameters()), create_graph=True)
            h_es = [v1 + (1 - args.damping) * h_es1 - hv1 / args.scale for v1, h_es1, hv1 in zip(vector, h_es, hv)]
            h_temp = h_es
            h_es = []
            for i in h_temp:
                i = i.data
                h_es.append(Variable(i))
        print 'finish {} epochs'.format(epoch)
        print 'time spend = {}'.format(time.time()-time_start)
        time_start = time.time()
    h_es = [a / args.scale for a in h_es]
    return h_es

test_index = 4

s_grad , test_path = get_stest(test_index)

h_es_init = []

for i in s_grad:
    i = i.data
    h_es_init.append(Variable(i))

infl_list = []
path_list = []
n = 244
repeat = 2
s_vec = None
for i in range(repeat):
    if s_vec == None:
        s_vec = s_test(h_es_init)
    else:
        s_vec = [a + b for (a, b)in zip(s_vec, s_test(h_es_init))]
    print 'finish {} repeats'.format(i + 1)
s_vec = [a/repeat for a in s_vec]
for img, label, path in loader_infl:
    if use_cuda:
        img, label = img.cuda(), label.cuda()
    img, label = Variable(img, volatile=False), Variable(label, volatile=False)
    output = net(img)
    loss = criterion(output, label)
    grad_t = grad(loss, net.parameters())
    influence = 0
    for g, k in zip(s_vec, grad_t):
        influence += torch.sum(g * k)
    influence = influence/n
    infl_list.append(influence)
    pa, = path
    path_list.append(pa)

final = zip(infl_list, path_list)
influence_negative = sorted(final, key=lambda x: x[0].data[0])
influence_positive = sorted(final, key=lambda x: x[0].data[0], reverse=True)

for i in range(6):
    print ('influence_negative{:.7f}'.format(influence_negative[i][0].data[0]))
for i in range(6):
    print ('influence_positive{:.7f}'.format(influence_positive[i][0].data[0]))

with open('./log-{}.txt'.format(file_time), 'a+') as f:
    print >> f, ('batch size:{}'.format(args.batch_size))
    print >> f, ('iteration:{}'.format(args.iteration))
    print >> f, ('damping:{}'.format(args.damping))
    print >> f, ('scale:{}'.format(args.scale))
    for i in range(6):
        print >> f, ('influence_negative{:.7f}'.format(influence_negative[i][0].data[0]))
    for i in range(6):
        print >> f, ('influence_positive{:.7f}'.format(influence_positive[i][0].data[0]))


log_dir = os.path.join('runs', '{}-{}-{}-{}-{}'.format(args.batch_size, args.iteration, args.scale, args.damping, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
writer = SummaryWriter(log_dir=log_dir)
tran = transforms.Compose([transforms.Scale(256),
                           transforms.CenterCrop(224),
                           transforms.ToTensor()])
for i in range(5):
    pic = Image.open(influence_negative[i][1]).convert('RGB')
    pic = tran(pic)
    writer.add_image('nega-influ{:.7f}'.format(influence_negative[i][0].data[0]),pic, i)
for j in range(5):
    pic = Image.open(influence_positive[j][1]).convert('RGB')
    pic = tran(pic)
    writer.add_image('posi-influ{:.7f}'.format(influence_positive[j][0].data[0]), pic, j)
writer.close()



