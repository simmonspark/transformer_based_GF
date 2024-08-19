# transformer_based_QA friend
## 챗봇 만들기 from scratch
transformer based QA friend project for my own.

!!note!! 4090 4way 정도의 컴퓨팅 파워와 충분한 전력이 갖춰지지 않은 상태에서는 scratch로 학습하는 것 보단, T5 pretrain model로 transfer learning 지키는 것이 결과를 확인하는데 좋을겁니다.

!!note!! In situations where 4090 4way computing power and sufficient power are not available, it is recommended to use transfer learning with a T5 pretrain model to check the results rather than learning from scratch.

!! Due to problems with triangular matrix implementation, a multihead attention module was used in the decoder part.
Of course, there is also multihead attention implemented by scratch, but it is only used in the encoder.

![image](https://github.com/user-attachments/assets/8054ded3-43d3-470f-8fd6-56ba88e8a832)

in this project the implementation is slightly different as avoiding explosive gradient & not using lr warmup decay

at. Skip connection and normalization were separated.


![스크린샷 2024-08-15 18-10-00](https://github.com/user-attachments/assets/9d4db099-d91a-48d7-b51c-bd788bd2c183)



You can download the data by going to the provided URL. in this project we are assuming it has been prepared.

url as below

[https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71633](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=543)


![image](https://github.com/user-attachments/assets/101371eb-7d35-4751-9dba-bbfe4c65e262)



### Hardware limitations as follow

Vram : 10G
Ram : 27G at preprocessing. in train process, sampling at getitem(20G)

### model config

- n_encoder : 6
- n_decoder : 6
- n_attention_head : 4
- embedding depth : 256
- max token sequence : 1024
- batch : 18
- bit operation : 16bit
- grad accumulation : No
- PARAM : 54M
- 수정중.. embedding depth 관련한 수정사항 256 to 768

### at batch per Vram

- 6 -> 8G
- 12 -> 16G
- 18 -> 23.xx G

My Hardware
- Ram : 64G
- graphic : 4090 24G
- cpu : i7-12700K

![스크린샷 2024-08-17 13-13-57](https://github.com/user-attachments/assets/2d7f4f9a-c266-4640-8229-c57d73afb72b)


![스크린샷 2024-08-17 13-27-29](https://github.com/user-attachments/assets/4f4ee03f-e6d4-451c-80ca-3b7745b413cb)


## HOW to inference
1. 가라 test
   : dataset에서 sampling 해서 테스트 하는데 이건 한국어에서 영어 번역 시험에서 정답지를 힐끔힐끔 보면서 하는 느낌
   ![image](https://github.com/user-attachments/assets/af24e04c-343f-4e93-b646-aea105bb3916)
 
   ![스크린샷 2024-08-19 20-04-15](https://github.com/user-attachments/assets/71e60204-1a05-45dd-8e9b-5f5e11d7d705)


3. Autoregressive method (구현중)
   : next token generation이랑 상당히 비슷한데 t+1 sequence generation에서 target은 0~t가 된다.


video as tutorial

 : To be uploaded

