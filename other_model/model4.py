import numpy as np
import sys
import paddle
import random
from paddle.nn import Transformer,TransformerEncoderLayer,TransformerEncoder,TransformerDecoderLayer,TransformerDecoder
from paddle.nn import Linear
from paddle.io import Dataset
import paddle.fluid as fluid
import paddle.distributed as dist
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

max_embedding_length=1024
attention_heads=8

def check(A,B):
    if (A=='A' and B=='U') or (A=='U' and B=='A'):
        return True
    elif (A=='G' and B=='C') or (A=='C' and B=='G'):
        return True
    elif (A=='U' and B=='G') or (A=='G' and B=='U'):
        return True
    else:
        return False

def calculate_RNAseq(RNA_sqe,RNA_struct):
    data=np.zeros((len(RNA_sqe),15))
    ids=np.zeros(len(RNA_sqe))
    mask=np.zeros((len(RNA_sqe),len(RNA_sqe)))
    for i in range(len(RNA_sqe)):
        if RNA_sqe[i]=='A':
            data[i][0]=1
            id=0
        elif RNA_sqe[i]=='G':
            data[i][1]=1
            id=1
        elif RNA_sqe[i]=='C':
            data[i][2]=1
            id=2
        elif RNA_sqe[i]=='U':
            data[i][3]=1
            id=2
        else:
            print("error:",RNA_sqe[i])
            
        if RNA_struct[i]=='(':
            data[i][4]=-1
            type_id=-1
        elif RNA_struct[i]==')':
            data[i][4]=1
            type_id=1
        elif RNA_struct[i]=='.':
            data[i][4]=0
            type_id=0
        else:
            print("error:",RNA_struct[i])
        
        if i>0:
            data[i-1][5+id]=1
            data[i-1][9]=type_id
        if i<len(RNA_sqe)-1:
            data[i+1][10+id]=1
            data[i-1][14]=type_id
        
    for i in range(len(RNA_sqe)):
        if i<len(RNA_sqe)-1:
            mask[i][i+1]=1
        if i>0:
            mask[i][i-1]=1
        if RNA_struct[i]=='(':
            for j in range(len(RNA_sqe)):
                if RNA_struct[j]==')' and check(RNA_sqe[i],RNA_sqe[j]):
                    mask[i][j]=1
                    mask[j][i]=1
        elif RNA_struct[i]=='.':
            mask[i][:]=1
                    
    return data,ids,mask

def load_data(file_name):
    read_file=open(file_name,'r')
    line=read_file.readline()
    num=0
    all_data=[]
    #all_label=[]
    max_length=0
    while True:
        if line=='':
            break
        if line[0]==">":
            num+=1
            RNA_sqe=read_file.readline().strip()
            RNA_struct=read_file.readline().strip()
            assert(len(RNA_sqe)==len(RNA_struct)),"error,{},{}".format(len(RNA_sqe),len(RNA_struct))
            data,ids,mask=calculate_RNAseq(RNA_sqe,RNA_struct)
            #print(len(RNA_sqe))
            if len(RNA_sqe)>max_length:
                max_length=len(RNA_sqe)
            label=np.zeros(len(RNA_sqe)) #
            for i in range(len(RNA_sqe)):
                line=read_file.readline().strip().split(' ')
                #print(line[0])
                if int(line[0])!=int(i+1):
                    print("error label:",line)
                    sys.exit()
                label[i]=float(line[1])
            all_data.append((data,label,len(RNA_sqe),mask))
            #all_label.append(label)
        line=read_file.readline()
    print(max_length)
    return all_data


train_data=load_data("../RNA_data/train.txt")
val_data=load_data("../RNA_data/dev.txt")
#np.save("train_data.npy",np.array(train_data))
#np.save("val_data.npy",np.array(val_data))
#train_data=np.load("train_data_2.npy",allow_pickle=True).tolist()
#val_data=np.load("val_data_2.npy",allow_pickle=True).tolist()
print(len(train_data),len(val_data))


class RNA_net(paddle.nn.Layer):

    # self代表类的实例自身
    def __init__(self):
        # 初始化父类中的一些参数
        super(RNA_net, self).__init__()
        
        #self.position=self.position_embedding(max_embedding_length,d_model)
        #self.fc_embedding = Linear(in_features=13, out_features=d_model)
        self.prelu=paddle.nn.PReLU()
        
        cell_fw1 = paddle.nn.LSTMCell(15, 16)
        cell_bw1 = paddle.nn.LSTMCell(15, 16)
        self.rnn1 = paddle.nn.BiRNN(cell_fw1, cell_bw1)
        
        cell_fw2 = paddle.nn.LSTMCell(32, 32)
        cell_bw2 = paddle.nn.LSTMCell(32, 32)
        self.rnn2 = paddle.nn.BiRNN(cell_fw2, cell_bw2)
        
        cell_fw3 = paddle.nn.LSTMCell(64, 64)
        cell_bw3 = paddle.nn.LSTMCell(64, 64)
        self.rnn3 = paddle.nn.BiRNN(cell_fw3, cell_bw3)

        self.transformer = Transformer(d_model=128, nhead=attention_heads, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward = 256,dropout=0.3,normalize_before=True)
        
        self.fc = Linear(in_features=128, out_features=2)
        self.sigmoid= paddle.nn.Sigmoid()
        self.softmax=paddle.nn.Softmax(axis=-1)
    
    # 网络的前向计算
    def forward(self, inputs, length):
        att_mask=np.zeros((inputs.shape[0],attention_heads,inputs.shape[1],inputs.shape[1]))
        out_mask=np.zeros((inputs.shape[0],inputs.shape[1]))
        out_mask1=np.zeros((inputs.shape[0],inputs.shape[1],32))
        out_mask2=np.zeros((inputs.shape[0],inputs.shape[1],64))
        for i in range(inputs.shape[0]):
            att_mask[i,:,:length[i],:length[i]]=1
            out_mask[i,:length[i]]=1
            out_mask1[i,:length[i],:]=1
            out_mask2[i,:length[i],:]=1
        att_mask=paddle.to_tensor(att_mask)
        out_mask=paddle.to_tensor(out_mask)
        out_mask1=paddle.to_tensor(out_mask1)
        out_mask2=paddle.to_tensor(out_mask2)
        outputs, final_states=self.rnn1(inputs)
        outputs, final_states=self.rnn2(outputs*out_mask1)
        outputs, final_states=self.rnn3(outputs*out_mask2)
        outputs = self.transformer(outputs,outputs,att_mask,att_mask,att_mask)
        #output = self.sigmoid(self.fc(outputs))
        output = self.softmax(self.fc(outputs))[:,:,0]
        output=paddle.reshape(output,[output.shape[0],output.shape[1]])
        return output*out_mask
        
    def position_embedding(self,max_length,embedding_size):
        position_embedding=np.zeros((max_length,embedding_size))
        for pos in range(max_length):
            for i in range(embedding_size//2):
                position_embedding[pos,2*i]=np.sin(pos/np.power(10000,2*i/embedding_size))
                position_embedding[pos,2*i+1]=np.cos(pos/np.power(10000,2*i/embedding_size))
        return paddle.to_tensor(position_embedding,dtype="float32")
        
model = RNA_net()

#layer_state_dict = paddle.load("save/RNA_net.pdparams")
#model.set_state_dict(layer_state_dict)

dist.init_parallel_env()
model = paddle.DataParallel(model)
# 开启模型训练模式
model.train()
boundaries=[20,100,1000,30000]
values = [0.1,0.01,0.001,0.0001,0.00001]
opt=paddle.optimizer.Adam(learning_rate=0.00025,parameters=model.parameters())#fluid.layers.piecewise_decay(boundaries=boundaries, values=values)
loss_fn=paddle.nn.MSELoss(reduction='mean')



EPOCH_NUM = 100   # 设置外层循环次数
BATCH_SIZE = 16  # 设置batch大小

# 定义外层循环
for epoch_id in range(EPOCH_NUM):
    random.shuffle(train_data)
    mini_batches = [train_data[k:k+BATCH_SIZE] for k in range(0, len(train_data), BATCH_SIZE)]
    # 定义内层循环
    if epoch_id==30:
        opt.set_lr(0.000025)
    '''
    if epoch_id==30:
        opt.set_lr(0.0000025)
    if epoch_id==60:
        opt.set_lr(0.000001)
    '''
    for iter_id, mini_batch in enumerate(mini_batches):
        X=[]
        Y=[]
        L=[]
        max_length=0
        for d in mini_batch:
            if d[2]>max_length:
                max_length=d[2]
        data=np.zeros((max_length,15))
        label=np.zeros(max_length)
        mask=np.zeros((max_length,max_length))
        for d in mini_batch:
            data[:d[2],:]=d[0]
            label[:d[2]]=d[1]
            mask[:d[2],:d[2]]=d[3]
            X.append(data)
            Y.append(label)
            L.append(d[2])

        x = np.array(X) 
        y = np.array(Y)
        l = np.array(L)
        #print(x.shape,y.shape)
        # 将numpy数据转为飞桨动态图tensor形式
        data = paddle.to_tensor(x,dtype="float32")
        label = paddle.to_tensor(y,dtype="float32")
        
        # 前向计算
        predicts = model(data,l)
        
        # 计算损失
        loss=loss_fn(predicts,label)
        #avg_loss = paddle.mean(loss)
        if iter_id%20==0:
            print("epoch: {}, iter: {}, loss is: {}".format(epoch_id, iter_id, loss.numpy()))
        
        # 反向传播
        loss.backward()
        # 最小化loss,更新参数
        opt.step()
        # 清除梯度
        opt.clear_grad()
    
    loss=0
    rsmd=0
    for d in val_data:
        X=d[0]
        Y=d[1]
        L=d[2]
        x = np.array([X]) 
        y = np.array([Y])
        l = np.array([L])
        data = paddle.to_tensor(x,dtype="float32")
        label = paddle.to_tensor(y,dtype="float32")
        predicts = model(data,l)
        loss+=loss_fn(predicts,label).numpy()
        rsmd+=np.sqrt(np.sum((predicts.numpy()-label.numpy())**2)/l)
    print("val_loss:{}, rmsd={}".format(loss/len(val_data),rsmd/len(val_data)))
    
    paddle.save(model.state_dict(), "save/model4/RNA_net.pdparams")

