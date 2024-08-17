# transformer_based_QA friend
## 시언이의 두근두근 챗봇 만들기 from scratch
transformer based QA friend project for my own..

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

![스크린샷 2024-08-17 13-13-57](https://github.com/user-attachments/assets/992931ed-9191-4b0e-902d-8597aecb296b)



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

### at batch per Vram

- 6 -> 8G
- 12 -> 16G
- 18 -> 23.xx G

My Hardware
- Ram : 64G
- graphic : 4090 24G
- cpu : i7-12700K
