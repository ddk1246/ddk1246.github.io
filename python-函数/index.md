# Python函数记录


<!--more--> 

 {{< admonition tip"" >}}

   本篇记录阅读项目中发现的工具包使用方法

  {{< /admonition >}}



## basic

- while 
  ```python
  name = str("some")
  names = [Str(),Str()]
  while name in names:
  name = name+"_"
  ```

- split() 和 rsplit()

  ```python
  # split() 
  # return [list] ,使用seq作为分割字符串（默认' ' and '\n'），maxsplit为最大分割次数
  str.split(sep=None, maxsplit=-1)
  
  # rsplit() 
  # 与split相似，分割从右侧开始
  str.rsplit(sep=None, maxsplit=-1)
  ```

<br/>

## os

- 使用 `os.path.dirname` 替代切换至上级目录`../`

  ```python
  from os.path import dirname
  BASE_DIR = dirname(abspath(__file__))
  ROOT_DIR = dirname(BASE_DIR)
  sys.path.append(BASE_DIR)
  sys.path.append(ROOT_DIR)
  ```

- `os.path.join(*List(PATH))` 路径拼接

  win32与linux使用不同的分隔符，可使用 `.replace()` 替换后再使用 `.split()` 分割

  **注意：多个路径组合后返回，第一个绝对路径之前的参数将被忽略**

- `os.path.basename` 返回文件名

  ```python
  >>> os.path.basename(r'c:\test.csv')
  'test.csv'
  >>> os.path.basename('c:\csv')
  'csv' （这里csv被当作文件名处理了）
  >>> os.path.basename('c:\csv\')
  ''
  ```

- 用户路径名

  ```python
  os.path.expanduser(dst)
  ```

## pathlib

- pathlib.Path(path)

  ```python
  from pathlib import Path
  source_path = ""
  file = Path(source_paht)
  file.parent
  file.touch() # 创建
  file.unlink() # 删除
  file.relative_to(dir_1) # file 相对于dir_1的位置
  #or
  files = Path(file).glob("*.txt") # 分级查找：*第一级 */* 第二级
  files = Paht(flie).rglob("*.mp4") # 递归查找
  for f in files:
    print(f)
  
  # 路径解码 -> windows路径分隔符更改，更换为绝对路径
  file = Path().resolve()
  
  # 将~替换为用户家目录 ,~user替换为user的家目录
  Path().expanduser()
  ```

## glob

- `glob.glob()` 返回所有匹配的文件路径列表

  ```python
  img_path = sorted(glob.glob(os.path.join(images,"*.jpg")))
  ```

  ```abc
  ”*”匹配0个或多个字符；
  ”?”匹配单个字符；
  ”[]”匹配指定范围内的字符，如：[0-9]匹配数字。  
  ```

<br/>

## numpy

- 数组拼接 `np.concatenate(arrays, axis=i)` e.g. `i`记为可滑动第i维索引

  ```python
  arr = []
  for subarr in iterator:
    arr.append(subarr)
  outarr = np.concatenate(arr,0)
  ```

- 随机选取

  ```python
  choices = np.random.choice(A,i) # A必须为一维数组 1—D
  choices = A[np.random.choice(A.shape[0],i),:] # A为高维数组
  choices = random.choice(A) # A维度不作要求
  ```

- 维度增加

  ```python
  np.newaxis == None
  img[np.newaxis, :, :] # -> 1,w,h
  ```

## collections

- 双向队列

  ```python
  from collections import deque
  ```
  
  <div align=center>
  <img src="/images/093fe6f890144c20c9434b2aebdb2550.png"/>
  </div>

- 有序集合

  ```python
  from collections import OrderedDict
  ```

  

## cv2

  {{< admonition note  "" >}}

​	opencv中的颜色读取默认为BGR通道

  {{< /admonition >}}

- 视频文件切换至指定帧

  ```python
  # 视频流信息（文件）读取
  capture = cv2.VideoCapture(os.path.join(videoPath, video))
  # 读取FPS
  fps = int(capture.get(cv2.CAP_PROP_FPS))
  # 设置将读取的帧数｛count｝，而后读取即可
  capture.set(cv2.CAP_PROP_POS_FRAMES,count)
  ret, img = capture.read()
  ```

- 仿射变换与透视变换

  ```python
  # 获得仿射变换矩阵M，由对应点获得(都需转换为np.float32)
  src_point = np.array([],dtype=np.float32)
  src_point = np.float32([[],])
  src_point = np.random.random((3,2)).asdtype(np.float32)
  dst_point = np.float([[],])
  
  # 仿射变换满足平直性和平行性
  M = cv2.getAffineTransform(src_point,dst_point)
  M = cv2.getPerspectiveTransform(src_point,dst_point)
  
  # 图像变换使用warp*,对索引做映射
  outImg = cv2.warpAffine(inImg,M,(w,h)) #仿射变换
  outImg = cv2.warpPerspective(inImg,M,(w,h)) #透视变换
  
  # 点集的变换使用perspectiveTransform映射,对值做映射
  points = points.reshape(1, -1, 2).astype(np.float32)
  out_point = cv2.perspectiveTransform(points, M).astype(np.int32).reshape(-1,2)
  
  
  ```

- 仿射变换估计

  ```python
  a, b = np.array(),np.array()
  
    # fullAffine 六自由度仿射变换,至少需要3个点,多余的点使用最小二乘法拟合
  
    # 尺度、旋转、位移、斜切
  
    # [[a b c]
  
    #  [d e f]]
  
    m = cv2.estimateAffine2D(a, b)
  
  # partialAffine 四自由度仿射变换,至少需要2个点
  
  # 尺度、旋转、位移
  
  # [[cos*s -sin*s tx]
  
  #  [sin*s  cos*s ty]]
  
  m = cv2.estimateAffinePartial2D(a, b)
  ```

  URL:[https://blog.csdn.net/dongfang1984/article/details/52959308](https://)
- 特征点匹配与映射
  
    ```python
    def sift_detect(srcImg, dstImg, detector='sift', matcher ='FLANN'):
        # 查找关键点
        if detector.startswith('si'):
            print("sift detector......")
            detector = cv2.SIFT_create()
        else:
            print("ORB detector......")
            detector = cv2.ORB_create()
            
        kp1, des1 = detector.detectAndCompute(srcImg, None)
        kp2, des2 = detector.detectAndCompute(dstImg, None)
        # kp1 返回类型为[cv2.KeyPoint,···]
        # KeyPoint.pt 关键点位置坐标
        # KeyPoint.size 关键点邻域直径
        # KeyPoint.angle 特征点的方向，值为[零, 三百六十)，负值表示不使用
        # KeyPoint.response KeyPoint.octave特征点所在的图像金字塔的组 KeyPoint.class_id 用于聚类的id
    
        # 关键点匹配
        if matcher.startswith('BFM'):
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2) #k=2为查找前两个相似点
        else:
            FLANN_INDEX_KDTREE = 0
            indexParams = dict(algorithm=FLANN_INDEX_KDTREE, tree=5)
            searchParams = dict(checks=100)  # 指定索引树被遍历多少次，次数越多，计算时间越长，越精确
    
            flann = cv2.FlannBasedMatcher(indexParams, searchParams)
            matches = flann.knnMatch(des1, des2, k=2) #k=2为查找前两个相似点
    
        # matches 返回类型为[cv2.DMatch,···]
        # • DMatch.distance - 描述符之间的距离。越小越好。
        # • DMatch.trainIdx - 目标图像中描述符的索引。
        # • DMatch.queryIdx - 查询图像中描述符的索引。
        # • DMatch.imgIdx - 目标图像的索引。 
        # 若最相似点的相似距离 小于 次相似点相似距离的0.5倍，则认为相似
        good = [m for m, n in matches if m.distance < 0.5 * n.distance] 
    
        min_match_count = 5
        if len(good) > min_match_count:
            # 匹配点
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            # 找到变换矩阵m
            m, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5, 0) #return 3*3
            matchmask = mask.ravel().tolist()
    
- 图像变换

  ```python
  # 注意交换坐标位置
  h, w = image_mask.shape[:2]
  image = cv.resize(image, (w, h), interpolation=cv.INTER_LINEAR)
  ```
	
- 圆形绘制

  ```python
  # 被绘制图像img | 点坐标w,h |半径:radius | 填充像素 | 线宽 -1为内部填充
  cv.circle(img, (point[1], point[0]), radius, (0), -1)
  ```

## logging

- 日志文件配置

  ```python
  https://blog.csdn.net/pansaky/article/details/90710751
  ```

## multiprocessing

- 设置守护进程

  ```python
  [setattr(process, "daemon", True) for process in processes]
  ```

## pandas
**筛选**

- 筛选指定行

  ```python
  data = df.loc[2:5]
  ```

- 筛选出数据某列为某值的所有数据记录

  ```python
  data = df[(df['列名1']== ‘列值1’)]
  # 多条件匹配时，使用 &|等符号连接各个条件
  # 注意：使用()包围各个条件
  data_many=df[(df['列名1']== ‘列值1’)&(df['列名2']==‘列值2’)]
  # 多值匹配时 条件取反可使用==False 和 ~
  data = data[~data["label"].str.contains('falling|squat')]
  str.startswith
  data = data[~data["NodeId"].isin (["fe50c54a-5091-4fb2-8487-efeffd10592d","fcc671c2-c92c-4285-934f-dda38a2ed475"])]
  ```
  
- 模式匹配

  ```python
  # 开头包含某值的模式匹配
  cond=df['列名'].str.startswith('值')
  # 中间包含某值的模式匹配,可添加正则表达式
  cond=df['列名'].str.contains('apple|banana')
  ```

- 数字值筛选

  ```python
  data = flags[(flags['num']>=1) | (flags['bars']>=1)]
  ```

- 参考：

  ```
  [https://blog.csdn.net/weixin_42322206/article/details/123607271](https://)
  ```

**表格编辑**

- 列表头替换

  ```python
  dt = pd.DataFrame()
  dt.columns = ['a','b','c'] # 缺点为必须与源表头数目对应
  dt.rename(columns={'A':'a','C':'c'}, inplace = True) #源表头与目标表头映射
  
  ```

## einops

与enisum函数相似，但enisum从索引的角度出发计算 **一个或两个矩阵** 的计算法则，但einops更应该应用在 **单个矩阵** 的变换。

- rearrange() 维数变换

  ```python
  images = rearrange(imgs, "(b1 b2) h w c -> (b1 h) (b2 w) c",b1 = 3)
  ```

- reduce() 维数减少可用于min、max、mean

  ```python
  reduce(obj, "(h h_s) (w w_s) c -> h w c", reduction="max", h_s=step, w_s=step)
  
  # 'b h w c -> b () () c' 等价于 'b h w c -> b 1 1 c'  
  ```

- repeat() 使用重复的方法增加维数

  ```python
  repeat(obj, "h w c -> (h h_s) (w w_s) c", h_s=step, w_s=step)
  ```

- parse_shape() 解析张量形状，将轴名称映射到其长度。

  ```python
  x = np.zeros([2, 3, 5, 7])
  parse_shape(x, 'batch _ h w')
  {'batch': 2, 'h': 5, 'w': 7}
  
  y = np.zeros([700])
  rearrange(y, '(b c h w) -> b c h w', **parse_shape(x, 'b _ h w')).shape
  (2, 10, 5, 7)
  ```

  https://zhuanlan.zhihu.com/p/372692913

  https://github.com/arogozhnikov/einops/blob/master/docs/1-einops-basics.ipynb

## Albumentations (图像数据增强)

- 加载

  ```python
  import albumentations as A
  import cv2
  
  image = cv2.imread("/path/to/image.jpg")
  image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) # cv2默认读取BGR格式，Albumentations使用RGB格式
  ```

- 管道组合 Compose ,单图像增强

  ```python
  transform = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
  ])
  
  transformed = transform(image = image) #內键为"image"
  transformed_image = transformed["image"]
  ```

- 图像 蒙版增强

  ```python
  mask = cv2.imread("/path/to/mask.png")
  transformed = transform(image = image, mask=mask) #內键为"image"
  transformed_image = transformed["image"]
  transformed_mask = transformed['mask']
  
  # masks = [mask1,mask2,···] 多蒙版模式
  transformed = transform(image=image, masks=masks)
  ```

- 图像 包围框增强

  ```python
  transform = A.Compose([
      A.RandomCrop(width=450, height=450),
      A.HorizontalFlip(p=0.5),
      A.RandomBrightnessContrast(p=0.2),
  ], bbox_params=A.BboxParams(format='coco', min_area=1024, min_visibility=0.1, label_fields=['class_labels']))
  # class_labels与定义时label_fields对应
  transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
  transformed_image = transformed['image']
  transformed_bboxes = transformed['bboxes']
  transformed_class_labels = transformed['class_labels']
  ```
  
  <div align=center>
  <img src="/images/86a306d2b2155ed71c10b961557135ab.png"/>
  </div>

- 图像 关键点 增强

  ```python
  transform = A.Compose([
      A.RandomCrop(width=330, height=330),
      A.RandomBrightnessContrast(p=0.2),
  ], keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels'], remove_invisible=True, angle_in_degrees=True))
  
  transformed = transform(image=image, keypoints=keypoints)
  transformed_image = transformed['image']
  transformed_keypoints = transformed['keypoints']
  ```

- 图像 包围框 关键点 增强

  ```
  pass
  ```

    https://blog.csdn.net/weixin_36670529/article/details/116626394

## natsort

- 自然排序 natsorted（本质套皮sorted，整合key）

  ```python
  from natsort import natsorted,ns
  
  # ns 为自然排序方法标志 
  # eg: ns.REAL = ns.FLOAT | ns.SIGNED 解析参数为有符号浮点数
  # eg：ns.PATH
  ret = natsorted(a, alg=ns.REAL | ns.LOCALE | ns.IGNORECASE)
  ```

## tensorboard

- https://zhuanlan.zhihu.com/p/484289017

