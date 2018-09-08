# encoding=utf-8

__Author__ = "李思原"
__Describe__ = "CPU与GPU算力进行比较"
__Time__ = "2018-09-08"

import tensorflow as tf
import numpy as np
import time
from matplotlib.font_manager import *
import matplotlib.pyplot as plt
import os


# 这里可以指定字体
myfont = FontProperties(fname='./微软vista仿宋.ttf', size=23)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

run_time = 101

record_t_normal = []
record_t_tf = []
record_i = []

# 规定图片尺寸
plt.figure(figsize=(10, 10))

for run_i in range(1, run_time):

    plt.clf()
    n = run_i * 1000

    time0_tf = time.time()

    mat_tf = tf.random_normal((n, n), 0, 1)
    mat_tf_inv = tf.matrix_inverse(mat_tf)

    sess = tf.Session()
    sess.run(mat_tf_inv)
    sess.close()
    etime_tf = time.time() - time0_tf

    time0_normal = time.time()
    mat_normal = np.mat(np.random.normal(0, 1, (n, n)))
    mat_normal_inv = np.linalg.inv(mat_normal)
    etime_normal = time.time() - time0_normal

    record_t_normal.append(etime_normal)
    record_t_tf.append(etime_tf)
    record_i.append(n)

    # 打印出每一轮需要的时间
    print('**' * 50)
    print('n:', n)
    print('etime_tf:', etime_tf)
    print('etime_normal:', etime_normal)

    gpu_time = '%.4f' % etime_tf
    cpu_time = '%.4f' % etime_normal

    plt.plot(record_i, record_t_tf, label='GPU:GeForce GTX 1070 Ti>> %s >> %sS' % (n, gpu_time), linewidth=2.0, ms=10)
    plt.plot(record_i, record_t_normal, label='CPU:Intel® Core™ i5-8500>> %s >> %sS' % (n, cpu_time), linewidth=2.0, ms=10)

    plt.legend()
    plt.grid()
    plt.title('GPU VS CPU', fontproperties=myfont)
    plt.xlabel('对n*n的二维矩阵求逆', fontproperties=myfont)
    plt.ylabel('求逆耗费时间（单位：秒）', fontproperties=myfont)

    # 保存图片
    plt.savefig("./picture/%d_gpu#%s_cpu#%s.png" % (n, etime_tf, etime_normal))
    # plt.pause(0.01)
# plt.show()
