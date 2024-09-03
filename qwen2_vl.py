import os
import time
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from modelscope import snapshot_download


MIN_PIXELS = int(os.getenv("MIN_PIXELS", 56 * 56))
MAX_PIXELS = int(os.getenv("MAX_PIXELS", 28 * 28 * 128))

model_dir = snapshot_download("qwen/Qwen2-VL-2B-Instruct")

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_dir, torch_dtype="auto", device_map="auto" if torch.backends.mps.is_available() else "cpu"
)


processor = AutoProcessor.from_pretrained(model_dir, min_pixels = MIN_PIXELS, max_pixels = MAX_PIXELS)


def extract_info(image_url, prompt):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_url,
                    "resized_height": 200,
                    "resized_width": 200,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("mps") if torch.backends.mps.is_available() else inputs.to("cpu")

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)


if __name__ == '__main__':
    image_url = 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'
    prompt = '描述一下这张图片'
    t1 = time.time()
    extract_info(image_url, prompt)
    t2 = time.time()
    print(t2-t1)
    extract_info(image_url, prompt)
    t3 = time.time()
    print(t3-t2)
    extract_info(image_url, prompt)
    t4 = time.time()
    print(t4-t3)

