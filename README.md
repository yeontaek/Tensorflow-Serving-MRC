# Tensorflow-Serving-MRC(정리중)

This repository is for Korean MRC service methods using Tensorflow Serving.

* Whale extension, 한국어 MRC : https://store.whale.naver.com/detail/hkmamenommegcobnflgojmfikpkfgjng

* Chrome extension, 한국어 MRC : 준비중

<br>

## Export Estimator model with BERT


### 1. Export Model 

Estimator model을 export할 경우 export_savedmodel을 사용하여, BERT모델을 pb 형태로 구성한다. export_savedmodel의 매개변수는 아래와 같다. 

 - export_dir_base : pb 저장 경로 
 - checkpoint_path : ckpt 모델 경로(e.g. bert_model.ckpt)
 - serving_input_fn : 모델의 입력 스키마 정의 
 
```python

estimator.export_savedmodel(export_dir_base="output_dir/bert-24-layer",
                             checkpoint_path=init_checkpoint,
                             serving_input_receiver_fn=serving_input_receiver_fn)

def serving_input_receiver_fn():
    feature_spec = {
		"unique_ids": tf.FixedLenFeature([], tf.int64),
		"input_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
		"input_mask": tf.FixedLenFeature([max_seq_length], tf.int64),
		"segment_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
	}

    serialized_tf_example = tf.placeholder(dtype=tf.string,
                                           shape=[None],
                                           name='input_example_tensor')
    receiver_tensors = {'examples': serialized_tf_example}
    features = tf.parse_example(serialized_tf_example, feature_spec)
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)
```

### 2. Load Model 

생성된 pb파일은  <b>from_saved_model</b>을 사용하여 호출한다. 모델을 load하고 prediction  방법은 아래와 같다. 

```python

predictor_fn = tf.contrib.predictor.from_saved_model(export_dir="output_dit/bert-24-layer")

predictions = predictor_fn({'examples': examples})

```






<br>

## TensorFlow Serving with Docker

https://www.tensorflow.org/tfx/serving/docker

## TensorFlow Serving REST API

https://hub.docker.com/r/tensorflow/serving/tags/

## Flask 
