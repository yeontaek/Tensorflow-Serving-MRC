# Tensorflow-Serving-MRC(정리중)

This repository is for MRC service methods using Tensorflow Serving.

* Whale extension, 한국어 MRC : https://store.whale.naver.com/detail/hkmamenommegcobnflgojmfikpkfgjng

<br>

## Export Estimator model with BERT


```python

estimator.export_savedmodel(export_dir_base="output_dit/24-layer-none",
                             checkpoint_path=FLAGS.init_checkpoint,
                             serving_input_receiver_fn=serving_input_receiver_fn)

def serving_input_receiver_fn():
    feature_spec = {
		"unique_ids": tf.FixedLenFeature([], tf.int64),
		"input_ids": tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
		"input_mask": tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
		"segment_ids": tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
	}

    serialized_tf_example = tf.placeholder(dtype=tf.string,
                                           shape=[None],
                                           name='input_example_tensor')
    receiver_tensors = {'examples': serialized_tf_example}
    features = tf.parse_example(serialized_tf_example, feature_spec)
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)
```

<br>


## TensorFlow Serving with Docker

https://www.tensorflow.org/tfx/serving/docker

## TensorFlow Serving REST API

https://hub.docker.com/r/tensorflow/serving/tags/

## Flask 
