import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel

'''See at how text is recognized by looking at embeddings of an image with text on them. 
1. Feed images through SigLip encoder
2. Feed these embeddings to a linear classifier and see how it performs
'''

model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

def get_image_embeddings(input_path):
    image = Image.open(input_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)
    return embeddings

# image_file = '/home/ko.hyeonmok/local_testing/VLM_interpretability/data/doggos_randomized/images/with_text_ILSVRC2012_val_00001968.png'
image_file = '/home/ko.hyeonmok/local_testing/VLM_interpretability/data/doggos_randomized/images/with_text_n02102040_334.png'


embeddings = get_image_embeddings(image_file)

print(embeddings)
print(embeddings.shape)
print(embeddings.size())