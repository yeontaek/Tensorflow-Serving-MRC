# Tensorflow-Serving-MRC

This repository is for MRC service methods using Tensorflow Serving.

* Whale extension, 한국어 MRC: https://store.whale.naver.com/detail/hkmamenommegcobnflgojmfikpkfgjng(공개)

## SentencePiece tokenizer 학습
 한국어 전용 BERT 모델을 만들기 위해 Google의 [SentencePiece](https://github.com/google/sentencepiece)을 사용하였습니다. 약 1억 8천만 문장(위키피디아, 뉴스 데이터)을 활용하여 32,000개의 vocabulary (subwords)를 학습하였습니다. 모델 type은 <code>--model_type</code> 옵션을 이용하여 bpe type을 사용 하였습니다. 
