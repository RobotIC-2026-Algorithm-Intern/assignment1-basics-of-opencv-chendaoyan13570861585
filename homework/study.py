import cv2 #计算机视觉库
import numpy as np  #矩阵和数组计算
import matplotlib.pyplot as plt #静态绘图

#1.1
tsubaki = cv2.imread('res/tsubaki.png')#读入一张图像 cv2.imread(path) 默认彩图（BGR）
kita = cv2.imread('res/kita.png')
lena = cv2.imread('res/lena.png', cv2.IMREAD_GRAYSCALE)#指定读取灰度图，单通道
#cv2.imread(path, flag)
#cv2.IMREAD_COLOR(默认值)，以彩色方式读入BGR三通道
#cv2.IMREAD_GRAYSCALE   以灰度方式读入图像
#cv2.IMREAD_UNCHANGED   保留原有格式，可能有4通道->透明度（alpha）

#OpenCV是BGR顺序
#纯红[0,0,255]

#在numpy数组里，shape会返回（height, width, channels)
#height, width 告诉你竖直、水平有多少个像素点; channels是通道数, 灰度图没有这个（但是称作单通道，有亮度信息）, 彩色图的通道有3个(BGR), 如果是带透明度的PNG图， 可能会有四个通道（BGRA）

print(type(lena), type(kita)) #告诉你它是什么类型 <class 'numpy.ndarray'> ->以NumPy的数组的形式保存 ->因此可以做矩阵运算
print(lena.shape, kita.shape) #告诉你矩阵的维度(heignt, width, channels)
print(lena.dtype, kita.dtype) #告诉你数组的数据类型->通常为uint8(unsignint), 取值为0~255

#cv2.imshow('kita', kita)#打开一个名称为'kita', 显示图像kita的窗口 最基本的是imshow('pic')
#cv2.waitKey(0)  #按任意键继续执行   cv2.waitKey(1)表示等待一毫秒

#while True:
#    cv2.imshow('lena',lena)
#
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break
#ord('')可以将字符自动转换为ASCII码
#& 0xFF 位运算 保证标准ASCII码？

#waitKey(1)的作用是按帧更新窗口(视频那样) 没有的话窗口可能会闪退

kita_grey = cv2.cvtColor(kita, cv2.COLOR_BGR2GRAY) #cv2.cvtColor() 是颜色空间转换函数
#cv2.COLOR_BGR2GRAY 表示把kita改成单通道
#cv2.imshow('kita_grey',kita_grey)
#cv2.waitKey(0)

cv2.imwrite('kita_grey.png', kita_grey)#把kita_grey在当前文件夹下以kita_grey.png的形式保存
#可以自定义路径 cv2.imwrite('C:/Users/用户名/Desktop/kita_grey.png'， kita_grey) 保存到桌面
#True 就是成功了，失败可能是没权限什么的

plt.rcParams['figure.figsize'] = (12, 8)#设置绘图窗口的大小
#plt.rcParams Matplotlib的全局配置字典   figure.figsize 控制新建图(plt.figure()/plt.subplots())的默认尺寸

def show_img(*imgs: np.ndarray) -> None: # --> None = (void)
# *img 表示你可以传任意数量的参,  并*希望*是numpy.ndarray类型(不会报错，类似注释)
# *img 会将参数收集到一个元组里面 f(x: int, y: str):...
    plt.figure() #创建一个画布
    for idx, img in enumerate(imgs): #enumerate相当于map<key, value>
        plt.subplot(1, len(imgs), idx + 1)
        #plt.subplot(x, y, index)将画布划成x行y列的网络,在第idx个格子里作图
        #index从1开始计数
        if len(img.shape) == 2:#维度为2，也就是灰度图
            plt.imshow(img, cmap = 'gray')  #cmap：把数值当作灰度表示
        elif len(img.shape) == 3:
            plt.imshow(img[:, :, ::-1]) #:-1 表示逆序 -> BRG -> GRB

show_img(tsubaki, kita_grey, lena)
#显示不出来图，我也不知道为什么
#2.1

#OpenCV是BGR -> 要转成RGB
b, g, r = cv2.split(tsubaki) #返回的是图像，分别以b,g,r强度表示亮度
show_img(tsubaki, r, g, b)
tsubaki_r, tsubaki_g, tsubaki_b = np.zeros_like(tsubaki), np.zeros_like(tsubaki), np.zeros_like(tsubaki) #一个五彩斑斓的黑，但保留了RGB的强度，后面方便只保留某一个颜色通道
tsubaki_b[:, :, 0] = b
tsubaki_g[:, :, 1] = g
tsubaki_r[:, :, 2] = r
#反正就是把对应颜色取出来，3会报错
show_img(tsubaki, tsubaki_r, tsubaki_g, tsubaki_b)

#2.2
#HSV    H:色相 红橙黄绿等(正常为0-360,  OpenCV为0-255)
#S：饱和度 V：亮度
tsubaki_hsv = cv2.cvtColor(tsubaki, cv2.COLOR_BGR2HSV_FULL)#cv2.cvtColor 转换颜色空间的函数 cv2.COLOR_BGR2HSV_FULL 将BGR转换成HSV图像, 范围 0-255
h, s, v = cv2.split(tsubaki_hsv) #和上面cv2.split(tsubaki)基本一样
show_img(tsubaki, h, s, v)

#3.1
#将一个单通道的值映射到两种不同的取值（0/255）
#THRESH_BINARY 二值化 大于阈值的像素变成白色(255), 否则黑色
#THRESH_BINARY_INV  二值化取反，和上面完全相反 
#THRESH_TRUNC   截断，超过阈值的像素压到阈值，其他保持原值
#THRESH_TOZERO  阈值归零，低于阈值的像素归零
#THRESH_TOZERO_INV  对应阈值归零取反，高于阈值的像素归零

img = cv2.imread('res/kita.png')
show_img(img)
img_b , img_g, img_r =  cv2.split(img)
thr, img_bin = cv2.threshold(img_r, 200, 255, cv2.THRESH_BINARY)
#cv2.threshold(src, thresh, maxval, type)   src是输入图像，单通道灰度图  thresh是阈值，OpenCV会逐像素比较src(x,y)和thresh
#maxval是最大值，在二值化和取反中会用到，通常为255  type是阈值化方法
#会返回retval(实际使用的阈值) 和 dst（处理后的结果图）      
#如果你设置了阈值，那么retval就是传进去的值; 如果是自动阈值算法，那么会返回一个算出来的最优值
#det的类型是numpy的矩阵
img[img_bin == 0] = 0   #*修改对象是img而不是img_r!!!*
#img_bin是二值化的图(0 or 255)  表达式img_bin == 0返回一个bool矩阵: 0 -> True   255 -> False  可以判定哪些地方是黑色
#选中img_bin中所有的0像素位置(黑色位置) 并 将这些位置赋值为0(三个通道一起生效)
show_img(img_bin, img)

red_l = np.array([220, 30, 150])    #红色的下界和上界(HSV色彩空间)
red_u = np.array([250, 255, 255])
r_mask = cv2.inRange(tsubaki_hsv, red_l, red_u)#每个像素点是否在range内, 在的返回255 不在返回0
hana = tsubaki.copy()#复制喵
hana[r_mask == 0] = 0
show_img(tsubaki, r_mask, hana)

#try

img = cv2.imread('res/ex1.jpg')
show_img(img)
blue_l = np.array([100, 100, 100])
blue_u = np.array([255, 255, 255])
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
b_mask = cv2.inRange(img_hsv, blue_l, blue_u)
ball = img.copy()
ball[b_mask == 0] = 0
show_img(ball)

#4.0 滤波与去噪

#均值滤波 -> 用周围像素的平均值替换(图会糊)
#高斯滤波 -> 在均值滤波上根据距离加权
#双边滤波 -> 同时考虑空间距离和灰度差异(相似度),能保留图像边缘

#4.1    均值滤波(Mean Filter)
#用一个小窗口覆盖图像，每一个像素都用窗口所有像素的平均值来替换
#用于处理轻微噪声，边缘处理效果差
img = cv2.imread('res/rabbit.jpg')
img_mean = cv2.blur(img, (5, 5))#均值滤波 cv2.blur(src, ksize[, dst, [, anchor[, borderType]]]) -> dst
#ksize: 卷积核大小(就是那个小窗口)  dst：输出图像   anchor:锚点(卷积核的中心, 默认(-1, -1)表示自动取中心)   borderType: 边界模式 决定边缘像素怎么处理
show_img(img, img_mean)
cv2.imwrite("img_mean.png", img_mean)

###############################################################################################################################################################################第一反应是好像是感觉可以用来自动步兵转骑兵qwq(逃)orz

#4.2 高斯滤波(Gaussian Filter)
img = cv2.imread('res/rabbit.jpg')
img = img[150:550, 250:650]#裁剪，保留范围内的图像
kernel_gauss = cv2.getGaussianKernel(5, sigma=1)  
#getGaussianKernel(ksize, sigma)
#ksize 核的大小 int && odd  sigma 高斯分布的标准差(控制分布的'宽度') -> 小sigma -> 分布窄，权重集中在中间，模糊效果弱; 大的反之
#如果你写 sigma = 0, 系统会根据ksize自动计算一个合适的值
print(kernel_gauss)

img_gauss1 = cv2.GaussianBlur(img, (11, 11), 1)#一维高斯滤波
#dst = cv2.GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType]]])
#返回dst, 滤波之后的图像 sigmaX,Y  水平/竖直方向的标准差，Y默认0时等于sigmaX
matrix_kernel = kernel_gauss.dot(kernel_gauss.T)#将一维的高斯核转置后相乘，得到二维的高斯卷积核矩阵
#.T 转置，将(1XN)变成(NX1)  .dot()矩阵乘法/向量内积（行乘列）
print(matrix_kernel)
img_gauss2 = cv2.GaussianBlur(img, (5,5), 1)#二维高斯滤波
show_img(img, img_gauss1, img_gauss2)

#4.3双边滤波(Bilateral Filter)  边缘保护效果好，但效率低
img2 = cv2.imread("res/test_bi.png")
img2_bi = cv2.bilateralFilter(img2, 9, 75, 75)
#dst = cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace)
#d -> 邻域直径（以目标像素为中心的正方形窗口）sigmaColor -> 颜色标准差 差越大，越允许相差大的颜色融合 -> 模糊更强
show_img(img2, img2_bi)

#5
#Range of Interest, ROI, 需要着重处理的区域，用切片得到
img_far = cv2.imread('res/far.jpg')
img_ex = cv2.imread('res/ex1.jpg')
show_img(img_far, img_ex)
show_img(img_far, img_ex)
show_img(img_far[200: , :, :], img_ex[200: , :, :]) #start stop step, 这里从200开始，把0-200切了
#别忘，彩图是(height,width, channels)
roi = tsubaki[40: 200, 90: 240, :]
roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV_FULL)

red_l = np.array([190, 5, 150])
red_u = np.array([255, 255, 255])
r_mask = cv2.inRange(roi_hsv, red_l, red_u)
r_mask = cv2.dilate(r_mask, (5, 5), iterations=4)   #迭代四次
#dilaton，膨胀 用一个卷积核在二值图像上滑动，覆盖区域内如果至少有一个白色(255), 就把中心点设为白色
#cv.dilate()    处理时用的是原图，不会左脚踩右脚
#颜色分割(cv2.inRange)得到的掩码通常有噪声，这玩意可以更平滑
hana = roi.copy()
hana[r_mask == 0] = 0
hana = cv2.erode(hana, (2, 2), iterations=2)
#erosion, 腐蚀 和上面相反，让白色区域"变瘦"，只有卷积核覆盖范围全白，中心点菜白蛇，否则变黑
show_img(roi, r_mask, hana)

#6.0
#面向对象编程   (但牌佬不取对象)

#6.1
#①自己·对方回合1次，把以「阿尔白斯之落胤」为融合素材的1只融合怪兽从额外卡组送去墓地才能发动。选场上1只怪兽除外。下个回合，这张卡不能使用这个效果。

#6.2.1
#init -> initialzation, 初始化
class Participant:
    #class来定义类，类包含了属性和方法，__init__是其中一种方法，叫做构造函数
    def __init__(self, name, Echo, dao=0):#每次调用类的实例对象时，__init__就会自动被调用
        #init方法的第一个参数必须是self, 后面和正常定义函数没区别
#init实际上就是用来初始化的 我的理解是把外部参传给属性的 self.属性名 = 外部参
        self.name = name
        self.Echo = Echo
        self.dao = dao
        self.is_alive = True
    
    def activate_skill(self, game_name):
        print(f"{self.name}在\"{game_name}\"游戏中觉醒了{self.Echo}")
    
        
    def earn_dao(self, amount):
        """获得「道」的方法"""
        self.dao += amount
        print(f"{self.name}获得{amount}颗「道」，当前总计：{self.dao}颗")
    
    def die(self):
        """死亡方法（但会复活）"""
        self.is_alive = False
        print(f"{self.name}在游戏中死亡...但将在第十一日复活")

# 创建参与者对象——也叫实例化
qi_xia = Participant("齐夏", "生生不息", 0)
qiao_jiajin = Participant("乔家劲", "破万法", 0)

# 使用对象方法
print("=== 参与者入场 ===")
qi_xia.activate_skill("说谎者游戏")
qiao_jiajin.activate_skill("力量竞技场")

qi_xia.earn_dao(3)
qiao_jiajin.earn_dao(2)

print("\n=== 游戏结果 ===")
qi_xia.die()

#6.3封装

class EchoAbility:
    """回响能力基类 - 封装回响能力名称，只暴露效果"""
    
    def __init__(self, name, effect, eye_position):
        # 成员变量
        # 有__的成员变量是私有的，不能被外部访问
        # 没有__的成员变量是公开的，可以被外部访问
        self.effect = effect        # 能力效果（公开）
        self.__eye_position = eye_position  # 眼睛位置（核心机密）
        self.__name = name          # 回响能力名称（私有，如"夺心魄"）
    
    # 成员函数：执行代码块
    def ability_effect(self):
        return f"能力效果：{self.effect}"

# 测试基础封装
print("=== 回响能力基础封装 ===")
duoxingpo = EchoAbility("夺心魄", "精神控制，强迫他人模仿自己的动作", "心脏")
print(duoxingpo.ability_effect())  # 只能看到效果

# 尝试访问私有属性会失败

#try是什么？
#try-except语句用于捕获和处理异常。
#在try块中放置可能引发异常的代码，如果发生异常，程序不会崩溃，而是跳转到except块中执行相应的处理代码。
try:
    print(duoxingpo.__name)  # 访问私有属性会失败
except Exception as e:
    print(f"机密保护：{e}")
