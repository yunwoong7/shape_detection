# Shape Detection
이미지에서 원하는 영역을 찾기 위해 DeepLearning 기반의 모델을 사용하지만, 매우 간단하게 이미지 연산 방법을 통해 영역을 찾을 수도 있습니다. 그 첫번째로 **윤곽선의 속성을 이용하여 모양을 검출**하는 방법에 대해 알아 보겠습니다.

<div align="center">
  <img src="https://blog.kakaocdn.net/dn/bum4sV/btrRTESPo2j/tomZB0jaACWEJeQGAmjEo0/img.png" width="50%">
</div>

- 이미지를 그레이스케일로 변환
- 노이즈를 줄이기 위한 이미지 블러링
- 이미지의 이진화
- 윤곽선 검출
- 추출된 윤곽선을 기준으로 근사 다각형 검출
- Shape 종류 판단

------

#### **Import packages**

```python
import cv2
import imutils
import matplotlib.pyplot as plt
```

#### **Function declaration**

Jupyter Notebook 및 Google Colab에서 이미지를 표시할 수 있도록 Function으로 정의

```python
def img_show(title='image', img=None, figsize=(8 ,5)):
    plt.figure(figsize=figsize)
 
    if type(img) == list:
        if type(title) == list:
            titles = title
        else:
            titles = []
 
            for i in range(len(img)):
                titles.append(title)
 
        for i in range(len(img)):
            if len(img[i].shape) <= 2:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_GRAY2RGB)
            else:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB)
 
            plt.subplot(1, len(img), i + 1), plt.imshow(rgbImg)
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
 
        plt.show()
    else:
        if len(img.shape) < 3:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
        plt.imshow(rgbImg)
        plt.title(title)
        plt.xticks([]), plt.yticks([])
        plt.show()
```

Shape 을 판단하는 Function (이 부분을 수정하여 특정 Shape만 추출을 하는 기능을 구현할 수도 있습니다.)

```python
def shape_label(c):
    shape = "unidentified"
    # cv2.arcLength를 이용해 윤곽선의 전체 길이를 계산
    peri = cv2.arcLength(c, True)
    # cv2.approxPolyDP를 활용해 윤곽선의 근사 다각형 검출
    approx_poly = cv2.approxPolyDP(c, 0.04 * peri, True)
    
    # 꼭지점이 3개이면 삼각형
    if len(approx_poly) == 3:
        shape = "triangle"
    # 꼭지점이 4개이면 사각형
    elif len(approx_poly) == 4:
        # 종횡비를 구해서 정사각형 판단
        (x, y, w, h) = cv2.boundingRect(approx_poly)
        ar = w / float(h)
        
        if ar >= 0.95 and ar <= 1.05:
            shape = "square"  
        else:
            shape = "rectangle"
    # 꼭지점이 5개이면 오각형
    elif len(approx_poly) == 5:
        shape = "pentagon"
    # 꼭지점이 6개이면 육각형
    elif len(approx_poly) == 6:
        shape = "hexagon"
    # 나머지는 원으로 판단
    else:
        shape = "circle"
        
    return shape
```

#### **Load Image**

```python
cv2_image = cv2.imread('asset/images/shape.jpg', cv2.IMREAD_COLOR)
img_show('original image', cv2_image)
```

<div align="center">
  <img src="https://blog.kakaocdn.net/dn/ckFpP9/btrRXunYStS/H1ijkowNux3awSMWnMFZXk/img.png" width="50%">
</div>

#### **Shape Detection**

아래 과정을 통해 이미지를 그레이스케일로 변환하고 노이즈를 줄이기 위한 이미지 블러링 후 이진화 합니다.

```python
resized = imutils.resize(cv2_image, width=640)
ratio = cv2_image.shape[0] / float(resized.shape[0])
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)[1]
 
img_show(['GaussianBlur', 'Threshold'], [blurred, thresh])
```

<div align="center">
  <img src="https://blog.kakaocdn.net/dn/cpwSPR/btrRShKFiJp/y0bUkiTNuFQggiQ5RJwDFk/img.png" width="50%">
</div>

이진화 이미지에서 윤곽선을 검출합니다.

```python
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
```

추출된 윤곽선을 기준으로 Shape을 판단하여 이미지에 표시합니다.

```python
vis = cv2_image.copy()
 
for c in cnts:
    # cv2.moments를 이용하여 객체의 중심을 계산
    M = cv2.moments(c)
    cX = int((M["m10"] / M["m00"]) * ratio)
    cY = int((M["m01"] / M["m00"]) * ratio)
    shape = shape_label(c)
    
    # 이미지에서 객체의 윤곽선과 Shape명을 표시
    c = c.astype("float")
    c *= ratio
    c = c.astype("int")
    cv2.drawContours(vis, [c], -1, (0, 255, 0), 10)
    cv2.circle(vis, (cX, cY), 20, (0, 255, 0), -1); 
    cv2.putText(vis, shape, (cX-80, cY-50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
```

Shape를 표현한 이미지를 확인합니다.

```python
img_show('Shape Detection', vis, figsize=(16,10))
```

<div align="center">
  <img src="https://blog.kakaocdn.net/dn/eldsrj/btrRTupe551/FpbvXZAjk09QlhTu8X3UeK/img.png" width="50%">
</div>

------

하지만 이러한 이미지 연산 방식은 수동으로 입력하는 매개변수 값에 따라 결과가 달라질 수 있다는 단점이 있다는 점을 참고하시길 바랍니다.

`thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]`

<div align="center">
  <img src="https://blog.kakaocdn.net/dn/eIi2eC/btrRXeZ2y5l/qv42Qs13lzKSPUCwz6Xy2K/img.png" width="50%">
</div>
