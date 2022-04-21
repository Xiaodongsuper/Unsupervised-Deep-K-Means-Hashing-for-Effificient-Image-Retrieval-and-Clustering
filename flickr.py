__author__ = 'DX'

from torchvision import transforms
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import models
import os
import numpy as np
import pickle
from datetime import datetime
import torch.nn as nn
import utils.DataProcessing as DP
import utils.CalcHammingRanking as CalcHR
from sklearn.cluster import KMeans
import CNN_model

os.environ['CUDA_VISIBLE_DEVICE'] = '0,1,2,3'
print(torch.cuda.device_count())
class FeatureExtractor(nn.Module):
    def __init__(self, sub_module,extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.sub_module = sub_module
        self.extracted_layers = extracted_layers
    def forward(self, x):
        output = []
        for name, module in self.sub_module._modules.items():
            if name is 'fc': x= x.view(x.size(0), -1)
            x = module(x)
            if name in self.extracted_layers:
                output.append(x)
        return output

img_to_tensor = transforms.ToTensor()





def LoadLabel(filename, DATA_DIR):
    path = os.path.join(DATA_DIR, filename)
    fp = open(path, 'r')
    labels = [x.strip() for x in fp]
    fp.close()
    return torch.LongTensor(list(map(int, labels)))

def EncodingOnehot(target, nclasses):
    target_onehot = torch.FloatTensor(target.size(0), nclasses)
    target_onehot.zero_()
    target_onehot.scatter_(1, target.view(-1, 1), 1)
    return target_onehot

def CalcSim(batch_label, train_label):
    S = (batch_label.mm(train_label.t()) > 0).type(torch.FloatTensor)
    return S

def CreateModel(model_name, bit, use_gpu):
    if model_name == 'vgg11':
        vgg11 = models.vgg11(pretrained=True)
        cnn_model = CNN_model.cnn_model(vgg11, model_name, bit)
    if model_name == 'vgg16':
        vgg16 = models.vgg16(pretrained=True)
        cnn_model = CNN_model.cnn_model(vgg16, model_name, bit)
    if model_name == 'alexnet':
        alexnet = models.alexnet(pretrained=True)
        cnn_model = CNN_model.cnn_model(alexnet, model_name, bit)
    if use_gpu:
        cnn_model = nn.DataParallel(cnn_model.cuda(), device_ids=[0,1,2,3])
        #cnn_model = cnn_model.cuda()
    return cnn_model

def AdjustLearningRate(optimizer, epoch, learning_rate):
    lr = learning_rate * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def GenerateCode(model, data_loader, num_data, bit, use_gpu):
    B = np.zeros([num_data, bit], dtype=np.float32)
    for iter, data in enumerate(data_loader, 0):
        data_input, _, data_ind = data
        if use_gpu:
            data_input = Variable(data_input.cuda())
        else: data_input = Variable(data_input)
        output = model(data_input)
        if use_gpu:
            B[data_ind.numpy(), :] = torch.sign(output.cpu().data).numpy()
        else:
            B[data_ind.numpy(), :] = torch.sign(output.data).numpy()
    return B

def Logtrick(x, use_gpu):
    if use_gpu:
        lt = torch.log(1+torch.exp(-torch.abs(x))) + torch.max(x, Variable(torch.FloatTensor([0.]).cuda()))
    else:
        lt = torch.log(1+torch.exp(-torch.abs(x))) + torch.max(x, Variable(torch.FloatTensor([0.])))
    return lt
def Totloss(U, B, lamda, num_train):
    theta = U.mm(U.t()) / 2
    t1 = (theta*theta).sum() / (num_train * num_train)
    l2 = (U - B).pow(2).sum()
    l = lamda * l2
    return l,  l2, t1


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output
    return hook

def ind2vec(ind, N=None):
    ind = np.asarray(ind)
    if N is None:
        N = ind.max()
    return (np.arange(N) == ind[:,None]).astype(int)

def UDKH_algo(bit, param, gpu_ind=[0,1,2,3]):
    # parameters setting
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_ind)

    DATA_DIR = './data/flickr'
    DATABASE_FILE = 'retrieval_file_list.txt'
    TRAIN_FILE = 'train-file-list.txt'
    TEST_FILE = 'test-file-list.txt'

    DATABASE_LABEL = 'retrieval_label.txt'
    TRAIN_LABEL = 'train-label.txt'
    TEST_LABEL = 'test-label.txt'

    batch_size = 64
    epochs = 15
    learning_rate = 0.005
    weight_decay = 10 ** -5
    model_name = 'vgg16'
    nclasses = 10
    ncluster = 3
    use_gpu = torch.cuda.is_available()

    filename = param['filename']
    gamma = param['gamma']
    lamda = param['lambda']
    param['bit'] = bit
    param['epochs'] = epochs
    param['learning rate'] = learning_rate
    param['model'] = model_name

    ### data processing
    transformations = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dset_train = DP.DatasetProcessingCIFAR_10(
        DATA_DIR, TRAIN_FILE, TRAIN_LABEL, transformations)

    dset_test = DP.DatasetProcessingCIFAR_10(
        DATA_DIR, TEST_FILE, TEST_LABEL, transformations)

    num_train, num_test = len(dset_train), len(dset_test)



    train_loader = DataLoader(dset_train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4
                             )

    test_loader = DataLoader(dset_test,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=4
                             )

    ### create model
    model = CreateModel(model_name, bit, use_gpu)

    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay)
    VGGs = models.vgg16(pretrained=True)
    print(VGGs)
    VGGs.classifier = nn.Sequential(*list(VGGs.classifier._modules.values())[:-1])
    for param in VGGs.parameters():
        param.requires_grad = False
    use_gpu = torch.cuda.is_available()

    train_feature = np.zeros((num_train, 4096))
    for iter, traindata in enumerate(train_loader, 0):
        train_input, train_label, batch_ind = traindata
        if use_gpu:
            train_input = train_input.cuda()
            VGGs = VGGs.cuda()

        temp_feature = VGGs(train_input)
        train_feature[batch_ind, :] = temp_feature.cpu().numpy()
    G = KMeans(n_clusters=ncluster, random_state=0).fit_predict(train_feature)
    G = np.transpose(ind2vec(G+1))

    # parameters setting
    B = torch.zeros(num_train, bit)
    U = torch.zeros(num_train, bit)
    H = np.zeros((num_train, bit))
    C = np.transpose(H[np.random.choice(num_train, ncluster),:])
    train_labels_onehot = torch.LongTensor(np.transpose(G)) #torch.LongTensor(dset_train.label)  #EncodingOnehot(G, nclasses)
    test_labels_onehot =  torch.LongTensor(dset_test.label) #EncodingOnehot(test_labels, nclasses)
    CG = np.mat(C) * np.mat(G)
    train_loss = []
    map_record = []

    totloss_record = []
    totl1_record = []
    totl2_record = []
    t1_record = []


    for epoch in range(epochs):
        epoch_loss = 0.0
        ## training epoch
        for iter, traindata in enumerate(train_loader, 0):
            B = torch.Tensor(B)
            train_input, train_label, batch_ind = traindata
            train_label = torch.squeeze(train_label)
            if use_gpu:
                train_label_onehot = torch.LongTensor(np.transpose(G[:, batch_ind]))#EncodingOnehot(train_label, nclasses)
                train_input, train_label = Variable(train_input.cuda()), Variable(train_label.cuda())
                S = CalcSim(train_label_onehot, train_labels_onehot)
            else:
                train_label_onehot = EncodingOnehot(train_label, nclasses)
                train_input, train_label = Variable(train_input), Variable(train_label)
                S = CalcSim(train_label_onehot, train_labels_onehot)

            model.zero_grad()
            train_outputs = model(train_input)
            for i, ind in enumerate(batch_ind):
                U[ind, :] = train_outputs.data[i]
                B[ind, :] = torch.sign(train_outputs.data[i])
            Bbatch = torch.sign(train_outputs)
            if use_gpu:
                theta_x = train_outputs.mm(Variable(U.cuda()).t()) / 2
                logloss = (Variable(S.cuda())*theta_x - Logtrick(theta_x, use_gpu)).sum() \
                        / (num_train * len(train_label))
                regterm = (Bbatch-train_outputs).pow(2).sum() / (num_train * len(train_label))
            else:
                theta_x = train_outputs.mm(Variable(U).t()) / 2
                logloss = (Variable(S)*theta_x - Logtrick(theta_x, use_gpu)).sum() \
                        / (num_train * len(train_label))
                regterm = (Bbatch-train_outputs).pow(2).sum() / (num_train * len(train_label))

            loss = - logloss + lamda * regterm
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print('[Train Phase][Epoch: %3d/%3d][Loss: %3.5f]' % (epoch+1, epochs, epoch_loss / len(train_loader)), end='')
        optimizer = AdjustLearningRate(optimizer, epoch, learning_rate)

        l, l2, t1 = Totloss(U, B, lamda, num_train)
        totloss_record.append(l)
        #totl1_record.append(l1)
        totl2_record.append(l2)
        t1_record.append(t1)
        for i in range(bit):
            H[i, :] = lamda * U[i, :].numpy() + gamma*np.transpose(CG[:,i])
        B = torch.sign(B)
        H = torch.sign(torch.from_numpy(H))
        B = np.mat(B.numpy())
        H = np.mat(H.numpy())
        C = np.mat(C)
        G = np.mat(G)
        while(1):
            Func_value1 = np.linalg.norm(B-np.transpose(C*G), 2)
            C = np.sign(np.transpose(H)*np.transpose(G))
            C[np.where(C==1)] = 1
            rho = 0.001
            mu = 0.01
            for iterIn in range(5):
                grad = -np.transpose(H)*np.transpose(G) + rho*np.tile(np.sum(C),(bit, 1))
                C = np.sign(C-1/mu*grad)
                C[np.where(C==0)] = 1
            Hamdist = 0.5*(bit - H*C)
            indx = np.argmin(Hamdist, 1)
            for i in range(num_train):
                if(Hamdist[i, indx[i]]<bit/3):
                    G[:,i]=0
                    indx2 = np.where(Hamdist[i,:]==Hamdist[i, indx[i]])
                    G[indx2, i] = 1
                else:
                    continue
            Func_value2 = np.linalg.norm(B-np.transpose(C*G), 2)
            if(np.abs(Func_value1-Func_value2)<0.01):
                break
        CG = C*G
        print('[Total Loss: %10.5f][total L2: %10.5f][norm theta: %3.5f]' % (l, l2, t1), end='')

        ### testing during epoch
        qB = GenerateCode(model, test_loader, num_test, bit, use_gpu)
        tB = GenerateCode(model, train_loader, num_train, bit, use_gpu)
        #tB = torch.sign(torch.Tensor(B)).numpy()
        map_ = CalcHR.CalcMap(qB, tB, test_labels_onehot, dset_train.label)
        train_loss.append(epoch_loss / len(train_loader))
        map_record.append(map_)

        print('[Test Phase ][Epoch: %3d/%3d] MAP(retrieval train): %3.5f' % (epoch+1, epochs, map_))

    model.eval()

    database_labels_onehot = dset_train.label 
    qB = GenerateCode(model, test_loader, num_test, bit, use_gpu)
    dB = GenerateCode(model, train_loader, num_train, bit, use_gpu)

    map = CalcHR.CalcMap(qB, dB, test_labels_onehot, database_labels_onehot)
    print('[Retrieval Phase] MAP(retrieval database): %3.5f' % map)

    result = {}
    result['qB'] = qB
    result['dB'] = dB
    result['train loss'] = train_loss
    result['map record'] = map_record
    result['map'] = map
    result['param'] = param
    result['total loss'] = totloss_record
    result['l1 loss'] = totl1_record
    result['l2 loss'] = totl2_record
    result['norm theta'] = t1_record
    result['filename'] = filename

    return result

if __name__=='__main__':
    lamda = 200
    gpu_ind = [0,1,2,3]
    gamma = 3
    bits = [32, 64]
    param = {}
    param['lambda'] = lamda
    param['gamma'] = gamma
    for bit in bits:
        filename = 'flickr_log/UDKH_' + str(bit) + 'bits_nuswide-10' + '_' + datetime.now().strftime(
            "%y-%m-%d-%H-%M-%S") + '.pkl'
        param['filename'] = filename
        print('---------------------------------------')
        print('[#bit: %3d]' % (bit))
        result = UDKH_algo(bit, param, gpu_ind)
        print('[MAP: %3.5f]' % (result['map']))
        print('---------------------------------------')
        fp = open(result['filename'], 'wb')
        pickle.dump(result, fp)
        fp.close()

