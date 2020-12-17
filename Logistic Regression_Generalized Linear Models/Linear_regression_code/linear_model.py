import numpy as np
import struct

###############################  start ###################################################
#### 从同目录下的 dataset 中取出 Mnist 数据集部分内容
def decode_idx3_ubyte(idx3_ubyte_file):
    """
    解析idx3文件的通用函数
    :param idx3_ubyte_file: idx3文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii' #因为数据结构中前4行的数据类型都是32位整型，所以采用i格式，但我们需要读取前4行数据，所以需要4个i。我们后面会看到标签集中，只使用2个ii。
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    #print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))

    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)  #获得数据在缓存中的指针位置，从前面介绍的数据结构可以看出，读取了前4行之后，指针位置（即偏移位置offset）指向0016。
    #print(offset)
    fmt_image = '>' + str(image_size) + 'B'  #图像数据像素值的类型为unsigned char型，对应的format格式为B。这里还有加上图像大小784，是为了读取784个B格式数据，如果没有则只会读取一个值（即一副图像中的一个像素值）
    #print(fmt_image,offset,struct.calcsize(fmt_image))
    images = np.zeros((num_images, num_rows, num_cols))
    #plt.figure()
    for i in range(num_images):
        #if (i + 1) % 10000 == 0:
            #print('已解析 %d' % (i + 1) + '张')
            #print(offset)
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        #print(images[i])
        offset += struct.calcsize(fmt_image)
#        plt.imshow(images[i],'gray')
#        plt.pause(0.00001)
#        plt.show()
    #plt.show()

    return images

def decode_idx1_ubyte(idx1_ubyte_file):
    """
    解析idx1文件的通用函数
    :param idx1_ubyte_file: idx1文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    #print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.zeros((num_images,1 ))
    for i in range(num_images):
        #if (i + 1) % 10000 == 0:
            #print ('已解析 %d' % (i + 1) + '张')
        labels[i] = int(struct.unpack_from(fmt_image, bin_data, offset)[0])
        offset += struct.calcsize(fmt_image)
    return labels

def load_train_images(idx_ubyte_file):
    """
    TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象，n为图片数量
    """
    return decode_idx3_ubyte(idx_ubyte_file)


def load_train_labels(idx_ubyte_file):
    """
    TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  60000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """
    return decode_idx1_ubyte(idx_ubyte_file)


def load_test_images(idx_ubyte_file):
    """
    TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  10000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象，n为图片数量
    """
    return decode_idx3_ubyte(idx_ubyte_file)


def load_test_labels(idx_ubyte_file):
    """
    TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  10000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """
    return decode_idx1_ubyte(idx_ubyte_file)

def load_mnist(data_url = "dataset/"):
    train_images_idx3_ubyte_file = data_url + "train-images.idx3-ubyte"
    train_labels_idx1_ubyte_file = data_url + "train-labels.idx1-ubyte"
    test_images_idx3_ubyte_file = data_url + "t10k-images.idx3-ubyte"
    test_labels_idx1_ubyte_file = data_url + "t10k-labels.idx1-ubyte"
    
    train_images = load_train_images(train_images_idx3_ubyte_file)
    test_images = load_test_images(test_images_idx3_ubyte_file)
    train_labels = load_train_labels(train_labels_idx1_ubyte_file)
    test_labels = load_test_labels(test_labels_idx1_ubyte_file)
    
    return train_images, train_labels.reshape(-1,1), test_images, test_labels.reshape(-1,1)

###############################  end  ###################################################

def label_transform(labels, nums):
    """
    将数字标签转换为独热编码标签
    :param labels 待转换的标签
    :param nums 标签的种类数
    :return labels.shape[0] * nums 
    """
    m = labels.shape[0]
    transformed_label = np.zeros((m, nums))
    for i in range(m):
        transformed_label[i][int(labels[i])] = 1
        
    return transformed_label

def train_dev_split(X, Y, dev_ratio = 0.25):
    # This function spilts data into training set and development set.
    train_size = int(len(X) * (1 - dev_ratio))
    return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]

def load_binary_dataset(data_url = "dataset/"):
    X_train_fpath = data_url + 'X_train'
    Y_train_fpath = data_url + 'Y_train'
    X_test_fpath = data_url + 'X_test'
    
    with open(X_train_fpath) as f:
        next(f)
        X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
    with open(Y_train_fpath) as f:
        next(f)
        Y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype = float)
    with open(X_test_fpath) as f:
        next(f)
        X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
        
    dev_ratio = 0.1
    X_train, Y_train, X_dev, Y_dev = train_dev_split(X_train, Y_train, dev_ratio = dev_ratio)
    
    return X_train, Y_train.reshape(-1,1), X_dev, Y_dev.reshape(-1,1)

def sigmoid(z):
    """
    计算输入的向量的 sigmoid 函数结果
    """
    return 1 / (1 + np.exp(-z))

def normalization(X, train = True, X_mean = None, X_std = None):
    """
    X_mean 为 None 时表示传入的是训练集，进行归一化并返回 X_mean 和 X_std
    X_mean 不为 None 时表示传入的是测试集，进行归一化
    """
    if(train):
        flag = 0
        X_mean = np.mean(X, 0).reshape(1, -1)
        X_std  = np.std(X, 0).reshape(1,-1)
        
    X = (X - X_mean) / (X_std + 1e-8)
    
    if(train):
        return X, X_mean, X_std
    else:
        return X
    
def shuffle(X, Y):
    """
    将输入的数据随机打乱
    """
    randomize = np.arange(X.shape[0])
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

def split_to_miniBatch(X, Y, batchSize = 64):
    """
    将数据集按 batchSize 划分成多个子集，batchSize 默认值为 64
    返回结果：一个 [] 列表，列表中每个元素为一个元组 ()，元组中包含两个参数 (X_miniBatch, Y_miniBatch)
    """
    X_shuffled, Y_shuffled = shuffle(X, Y)    #对样本进行随机打乱
    m, n = X.shape                            #获取样本个数
    completeBatchNum = int(np.ceil(m / batchSize)) #计算完整的 batch 个数
    miniBatchs = []
    
    for i in range(completeBatchNum):
        tempBatch_X = X[i * batchSize : (i + 1) * batchSize]
        tempBatch_Y = Y[i * batchSize : (i + 1) * batchSize]
        miniBatchs.append((tempBatch_X, tempBatch_Y))
    
    if(m % batchSize != 0):
        tempBatch_X = X[completeBatchNum * batchSize :]
        tempBatch_Y = Y[completeBatchNum * batchSize :]
        miniBatchs.append((tempBatch_X, tempBatch_Y))
    
    return miniBatchs

def add_col(X):
    """
    在输入向量的第一列前加一列全为 1 的列向量
    """
    m, n = X.shape
    ones = np.ones((m, 1))
    return np.hstack((ones, X))

class LogisticRegressionModel:
    def __init__(self, X, Y):
        (temp_X, self.X_mean, self.X_std) = normalization(X)
        self.Y = Y
        self.X_train = add_col(temp_X)
        self.m, self.n = self.X_train.shape
        self.parameters = np.zeros((self.n, 1))
        self.learning_rate = 0.00003
        self.iteration = 10000
        self.batchSize = 64
        self.seed = 0
        self.method = "batch"
        self.printLossFlag = True
        self.recordTimes = 0
        self.recordInterval = 100
        self.lossHistory = 0
        self.regularizationFlag = False
        self.accuracy = 0
        self.trainFlag = False
    
    def modify_variables(self, learning_rate = None, iteration = None, method = None, batchSize = None, seed = None, printLossFlag = None, recordTimes = None, recordInterval = None, regularizationFlag = None):
        if(learning_rate != None):
            self.learning_rate = learning_rate
        if(iteration != None):
            self.iteration = iteration
        if(method != None):
            if(method == "batch" or method == "b"):
                self.method = "batch"
            elif(method == "miniBatch" or method == "mb"):
                self.method = "miniBatch"
            elif(method == "stochastic" or method == "s"):
                self.method = "stochastic"
        if(batchSize != None):
            self.batchSize = batchSize
        if(seed != None):
            self.seed = seed
        if(printLossFlag != None):
            self.printLossFlag = printLossFlag
        if(recordTimes != None):
            self.recordTimes = recordTimes
        if(recordInterval != None):
            self.recordInterval = recordInterval
        if(regularizationFlag != None):
            self.regularizationFlag = regularizationFlag
        
    def hypothese(self, X, parameters):
        return sigmoid(np.dot(X, parameters))
    
    def cal_gred(self, X, sub, regularizationFlag = False):
        gred = np.dot(X.T, sub)
        return gred
    
    def cal_loss(self, y, h, regularizationFlag = False):
        loss = - np.dot(y.T, np.log(h + 1e-8)) - np.dot((1-y).T, np.log(1 - h + 1e-8))
        return loss
    
    def train(self):
        X = self.X_train
        y = self.Y
        train_method = self.method
        m = self.m
        iteration = self.iteration
        learning_rate = self.learning_rate
        batchSize = self.batchSize
        printLossFlag = self.printLossFlag
        regularizationFlag = self.regularizationFlag
        
        
        if(train_method == "miniBatch"):   #判断是否为 miniBatch ，并划分子集
            miniBatchs = split_to_miniBatch(X, y, batchSize)
            batchNum = len(miniBatchs)
    
        if (train_method == "miniBatch"):  #计算内循环次数
            subIteration = batchNum
        elif (train_method == "stochastic"):
            subIteration = m
        else:
            subIteration = 1
        
        if(int(self.recordTimes) > 0):    #计算记录间隔
            recordInterval = int(np.floor(iteration * subIteration / self.recordTimes))
        else:
            recordInterval = int(self.recordInterval)
        
        loss_history = np.zeros((int(np.floor(iteration * subIteration / recordInterval)),1))
        loss_i = 0
        
        for ite in range(iteration):
            for subIte in range(subIteration):
                parameters = self.parameters
                if (train_method == "miniBatch"):    #取出当前循环需要用于计算的 batch
                    batch_X, batch_y = miniBatchs[subIte]
                elif (train_method == "stochastic"):
                    batch_X, batch_y = (X[subIte, : ].reshape(1 , -1), y[subIte].reshape(-1 , 1))
                else:
                    batch_X, batch_y = (X, y)
                    
                h = self.hypothese(batch_X, parameters)
                sub = h - batch_y
                gred = self.cal_gred(batch_X, sub, regularizationFlag)
            
                if(printLossFlag and (ite * subIteration + subIte) % recordInterval == 0):
                    X_h = self.hypothese(X, parameters)
                    loss_history[loss_i] = self.cal_loss(y, X_h, regularizationFlag)
                    print("iteration "+ str(ite * subIteration + subIte) +" loss: " +  str(loss_history[loss_i]))
                    loss_i = loss_i + 1
                self.parameters = self.parameters - learning_rate * gred
        pre = self.predict(X)
        accuracy = self._accuracy(pre, y)
        if(printLossFlag):
            print("Accuracy: " + str(accuracy))
        self.accuracy = accuracy
        self.trainFlag = True
                
    def initial_parameters(self):
        self.parameters = np.zeros((self.n, 1))
    
    def predict(self, X):
        if(X.shape[1] != self.n):
            X = add_col(normalization(X, train = False, X_mean = self.X_mean, X_std = self.X_std))
        h = self.hypothese(X, self.parameters)
        return np.round(h).astype(np.int)
    
    def _accuracy(self, Y_pred, Y_label):
        # This function calculates prediction accuracy
        acc = 1 - np.mean(np.abs(Y_pred - Y_label))
        return acc
    
    def print_info(self):
        print("Train data shape: ("+ str(self.m) +", "+ str(self.n)+ ")")
        print("learning_rate: " + str(self.learning_rate))
        print("iteration: " + str(self.iteration))
        print("trainFlag: " + str(self.trainFlag))
        print("accuracy: " + str(self.accuracy))
        print("regularizationFlag: " + str(self.regularizationFlag))
        print("printLossFlag: " + str(self.printLossFlag))
        print("method: " + str(self.method))
        print("batchSize: " + str(self.batchSize))