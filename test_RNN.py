import numpy as np
import sys
import paddle
import random
from paddle.nn import Transformer,TransformerEncoderLayer,TransformerEncoder,TransformerDecoderLayer,TransformerDecoder
from paddle.nn import Linear
from paddle.io import Dataset
import paddle.fluid as fluid
import os
max_embedding_length=10000
attention_heads=8
d_model=128
out_file="predict/"

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
        
    '''
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
    '''
    return data,ids

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
            data,ids=calculate_RNAseq(RNA_sqe,RNA_struct)
            if len(RNA_sqe)>max_length:
                max_length=len(RNA_sqe)
            all_data.append((data,len(RNA_sqe),num))
        line=read_file.readline()
    print(max_length)
    return all_data
    
test_data=load_data("RNA_data/B_board_112_seqs.txt")
#np.save("test_data.npy",np.array(test_data))
#test_data=np.load("test_data.npy",allow_pickle=True).tolist()
print(len(test_data))


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

        #self.transformer = Transformer(d_model=128, nhead=attention_heads, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward = 256,dropout=0.3,normalize_before=True)
        
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
        #outputs = self.transformer(outputs,outputs,att_mask,att_mask,att_mask)
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

layer_state_dict = paddle.load("save/model3/RNA_net.pdparams")
model.set_state_dict(layer_state_dict)

for d in test_data:
    X=d[0]
    L=d[1]
    num=d[2]
    x = np.array([X]) 
    l = np.array([L])
    data = paddle.to_tensor(x,dtype="float32")
    predicts = model(data,l).numpy()[0]
    f=open(out_file+str(num)+'.predict.txt','w')
    for i in range(len(predicts)):
        f.write(str(predicts[i])+'\n')
    f.close()




