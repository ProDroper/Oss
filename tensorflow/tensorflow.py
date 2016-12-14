# MNIST 는 간단한 컴퓨터 비전 데이터 세트

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 데이터 가져오기
mnist = input_data.read_data_sets("./samples/MNIST_data/", one_hot=True)

# Set up model
x = tf.placeholder(tf.float32, [None, 784])	# [None, 784] 형태의 부정소숫점으로 이루어진 2차원 텐서로 표현합니다.
W = tf.Variable(tf.zeros([784, 10])) # 행렬 w는 10개의 숫자 클래스에 대해 이미지 벡터의 784개 픽셀과 곱셈하기 위한 크기를 가짐.
b = tf.Variable(tf.zeros([10]))	# 구조가 [Dimension(10)]이다.
y = tf.nn.softmax(tf.matmul(x, W) + b)	# 이미지 벡터 x와 가중치 행렬 W를 곱하고 b를 더한 텐서를 matmul 함수에 입력.

y_ = tf.placeholder(tf.float32, [None, 10]) # 교차 엔트로피 

cross_entropy = -tf.reduce_sum(y_*tf.log(y)) # 그래프를 그림.
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy) # 학습 속도 0.01과 경사 하강법 알고리즘을 사용하여 크로스 엔트로피를 최소화하는 역전파 알고리즘을 사용.

# Session
init = tf.initialize_all_variables() # 세션 내에서 변수를 사용할 수 있으려면 먼저 변수가 있어야한다. 해당 세션을 사용하여 초기화한다.

sess = tf.Session()	# 시스템에서 사용 가능한 디바이스 (CPU 또는 GPU)에서 텐서플로의 연산을 실행할 수 있음
sess.run(init)	# 파라미터로 전달된 op들에 해당하는 그래프의 완전한 부분집합을 수행한다.

# Learning
for i in range(1000):	# 1000번 학습
  batch_xs, batch_ys = mnist.train.next_batch(100)	# 학습 세트로부터 100개의 무작위 데이터들의 일괄 처리(batch)들을 가져옵니다
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})	# placeholders를 대체하기 위한 일괄 처리 데이터에 train_step 피딩을 실행합니다.

# Validation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))	# 정답률 예측
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))	# 이 결과는 부울 리스트를 줍니다. 얼마나 많은 비율로 맞았는지 확인하려면, 부정소숫점으로 캐스팅한 후 평균값을 구하면 됩니다. 

# Result should be approximately 91%.
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))	# 마지막으로, 테스트 데이터를 대상으로 정확도를 확인해 봅시다.
