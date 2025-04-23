import os
import sys
import numpy as np
from os import getcwd
import cv2
from ctypes import *
import getch

sys.path.append("MvImport/")
from MvCameraControl_class import *

# 显示图像
def image_show(image , name):
    image = cv2.resize(image, (600, 400), interpolation=cv2.INTER_AREA)
    name = str(name)
    cv2.imshow(name, image)
    # cv2.imwrite("name.bmp", image)
    k = cv2.waitKey(1)



# 枚举设备
def enum_devices(device = 0 , device_way = False):
    """
    device = 0  枚举网口、USB口、未知设备、cameralink 设备
    device = 1 枚举GenTL设备
    """
    if device_way == False:
        if device == 0:
            tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE | MV_UNKNOW_DEVICE | MV_1394_DEVICE | MV_CAMERALINK_DEVICE
            deviceList = MV_CC_DEVICE_INFO_LIST()
            # 枚举设备
            ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
            if ret != 0:
                print("enum devices fail! ret[0x%x]" % ret)
                sys.exit()
            if deviceList.nDeviceNum == 0:
                print("find no device!")
                sys.exit()
            print("Find %d devices!" % deviceList.nDeviceNum)
            return deviceList
        else:
            pass
    elif device_way == True:
        pass

def creat_camera(deviceList , nConnectionNum ,log = True , log_path = getcwd()):
    """
    :param deviceList:        设备列表
    :param nConnectionNum:    需要连接的设备序号
    :param log:               是否创建日志
    :param log_path:          日志保存路径
    :return:                  相机实例和设备列表
    """
    # 创建相机实例
    cam = MvCamera()
    # 选择设备并创建句柄
    stDeviceList = cast(deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents
    if log == True:
        ret = cam.MV_CC_SetSDKLogPath(log_path)
        print(log_path)
        if ret != 0:
            print("set Log path  fail! ret[0x%x]" % ret)
            sys.exit()
        # 创建句柄,生成日志
        ret = cam.MV_CC_CreateHandle(stDeviceList)
        if ret != 0:
            print("create handle fail! ret[0x%x]" % ret)
            sys.exit()
    elif log == False:
        # 创建句柄,不生成日志
        ret = cam.MV_CC_CreateHandleWithoutLog(stDeviceList)
        print(1111)
        if ret != 0:
            print("create handle fail! ret[0x%x]" % ret)
            sys.exit()
    return cam , stDeviceList

# 打开设备
def open_device(cam):
    # ch:打开设备 | en:Open device
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print("open device fail! ret[0x%x]" % ret)
        sys.exit()

# 开启取流并获取数据包大小
def start_grab_and_get_data_size(cam):
    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        print("开始取流失败! ret[0x%x]" % ret)
        sys.exit()


# 需要显示的图像数据转换
def image_control(data , stFrameInfo):
    if stFrameInfo.enPixelType == 17301505:
        image = data.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
        # image_show(image=image , name = stFrameInfo.nHeight)
    elif stFrameInfo.enPixelType == 17301514:
        data = data.reshape(stFrameInfo.nHeight, stFrameInfo.nWidth, -1)
        image = cv2.cvtColor(data, cv2.COLOR_BAYER_GB2RGB)
        # image_show(image=image, name = stFrameInfo.nHeight)
    elif stFrameInfo.enPixelType == 35127316:
        data = data.reshape(stFrameInfo.nHeight, stFrameInfo.nWidth, -1)
        image = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        # image_show(image=image, name = stFrameInfo.nHeight)
    elif stFrameInfo.enPixelType == 34603039:
        data = data.reshape(stFrameInfo.nHeight, stFrameInfo.nWidth, -1)
        image = cv2.cvtColor(data, cv2.COLOR_YUV2BGR_Y422)
        # image_show(image = image, name = stFrameInfo.nHeight)
    return image

# 主动图像采集
def access_get_image(cam , active_way = "getImagebuffer"):
    """
    :param cam:     相机实例
    :active_way:主动取流方式的不同方法 分别是（getImagebuffer）（getoneframetimeout）
    :return:
    """
    if active_way == "getImagebuffer":
        stOutFrame = MV_FRAME_OUT()
        memset(byref(stOutFrame), 0, sizeof(stOutFrame))
        while True:
            ret = cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
            if None != stOutFrame.pBufAddr and 0 == ret and stOutFrame.stFrameInfo.enPixelType == 17301505:
                print("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (stOutFrame.stFrameInfo.nWidth, stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nFrameNum))
                pData = (c_ubyte * stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight)()
                memmove(byref(pData), stOutFrame.pBufAddr, stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight)
                data = np.frombuffer(pData, count=int(stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight),dtype=np.uint8)
                image_control(data=data, stFrameInfo=stOutFrame.stFrameInfo)
            elif None != stOutFrame.pBufAddr and 0 == ret and stOutFrame.stFrameInfo.enPixelType == 17301514:
                print("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (stOutFrame.stFrameInfo.nWidth, stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nFrameNum))
                pData = (c_ubyte * stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight)()
                memmove(byref(pData), stOutFrame.pBufAddr, stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight)
                data = np.frombuffer(pData, count=int(stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight),dtype=np.uint8)
                image_control(data=data, stFrameInfo=stOutFrame.stFrameInfo)
            elif None != stOutFrame.pBufAddr and 0 == ret and stOutFrame.stFrameInfo.enPixelType == 35127316:
                print("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (stOutFrame.stFrameInfo.nWidth, stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nFrameNum))
                pData = (c_ubyte * stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight*3)()
                memmove(byref(pData), stOutFrame.pBufAddr, stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight*3)
                data = np.frombuffer(pData, count=int(stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight*3),dtype=np.uint8)
                image_control(data=data, stFrameInfo=stOutFrame.stFrameInfo)
            elif None != stOutFrame.pBufAddr and 0 == ret and stOutFrame.stFrameInfo.enPixelType == 34603039:
                print("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (stOutFrame.stFrameInfo.nWidth, stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nFrameNum))
                pData = (c_ubyte * stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight * 2)()
                memmove(byref(pData), stOutFrame.pBufAddr, stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight*2)
                data = np.frombuffer(pData, count=int(stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight * 2),dtype=np.uint8)
                image_control(data=data, stFrameInfo=stOutFrame.stFrameInfo)
            else:
                print("no data[0x%x]" % ret)
            nRet = cam.MV_CC_FreeImageBuffer(stOutFrame)
 
    elif active_way == "getoneframetimeout":
        stParam = MVCC_INTVALUE_EX()
        memset(byref(stParam), 0, sizeof(MVCC_INTVALUE_EX))
        ret = cam.MV_CC_GetIntValueEx("PayloadSize", stParam)
        if ret != 0:
            print("get payload size fail! ret[0x%x]" % ret)
            sys.exit()
        nDataSize = stParam.nCurValue
        pData = (c_ubyte * nDataSize)()
        stFrameInfo = MV_FRAME_OUT_INFO_EX()
        memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
        return pData, nDataSize, stFrameInfo

def export_frame(cam, pData, nDataSize, stFrameInfo):
    ret = cam.MV_CC_GetOneFrameTimeout(pData, nDataSize, stFrameInfo, 1000)
    if ret == 0:
        # print("get one frame: Width[%d], Height[%d], nFrameNum[%d], enPixelType[%d] " % (stFrameInfo.nWidth, stFrameInfo.nHeight, stFrameInfo.nFrameNum, stFrameInfo.enPixelType))
        image = np.asarray(pData)
        frame = image_control(data=image, stFrameInfo=stFrameInfo)
        return frame
    else:
        print("no data[0x%x]" % ret)



# 关闭设备与销毁句柄
def close_and_destroy_device(cam , data_buf=None):
    # 停止取流
    ret = cam.MV_CC_StopGrabbing()
    if ret != 0:
        print("stop grabbing fail! ret[0x%x]" % ret)
        sys.exit()
    # 关闭设备
    ret = cam.MV_CC_CloseDevice()
    if ret != 0:
        print("close deivce fail! ret[0x%x]" % ret)
        del data_buf
        sys.exit()
    # 销毁句柄
    ret = cam.MV_CC_DestroyHandle()
    if ret != 0:
        print("destroy handle fail! ret[0x%x]" % ret)
        del data_buf
        sys.exit()
    del data_buf

def open_camera():
    # 枚举设备
    deviceList = enum_devices(device=0, device_way=False)
    # 创建相机实例并创建句柄,(设置日志路径)
    cam, stDeviceList = creat_camera(deviceList, 0, log=False)
    # 打开设备
    open_device(cam)
    # 开启设备取流
    start_grab_and_get_data_size(cam)
    # 主动取流方式抓取图像
    pData, nDataSize, stFrameInfo = access_get_image(cam, active_way="getoneframetimeout")
    #输出相机实例
    return cam, pData, nDataSize, stFrameInfo



#显示相机捕捉的画面
def main():
    # 主动取流方式抓取图像
    cam, pData, nDataSize, stFrameInfo = open_camera()
    #处理图像
    while True:
        frame = export_frame(cam, pData, nDataSize, stFrameInfo)
        image_show(frame,"img")
    # 关闭设备与销毁句柄
    close_and_destroy_device(cam)


if __name__=="__main__":
    main()