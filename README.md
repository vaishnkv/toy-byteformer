# toy-byteformer

A toy implementation for Byteformer 

## Motivation:

I came across a blog post from crowdstrike on they are leveraging do malicious file detection. Byte Back: Next-Generation Malware Classification Using Binary Transformers (https://www.crowdstrike.com/en-us/blog/byte-back-next-gen-malware-classification/?utm_source=chatgpt.com)



## Results:

![W B Chart 25_06_2025, 15_44_14](https://github.com/user-attachments/assets/02774d8f-4cec-4d0d-a53f-ea761f997817)

## Observation:

- if you notice the more earlier if we subsample (I mean    strided convolution), the learning will be difficult. Even though the the time complexity got reduced to linear in the attention , it will take more time (more epochs) to reach the model convergence.


## Reference:

- https://github.com/apple/corenet/tree/main 
- https://www.crowdstrike.com/en-us/blog/byte-back-next-gen-malware-classification/
- Bytes Are All You Need: Transformers Operating Directly On File Bytes (https://arxiv.org/abs/2306.00238 )
