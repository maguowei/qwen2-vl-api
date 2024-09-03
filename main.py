import torch
from fastapi import FastAPI
from qwen_vl_utils import process_vision_info
from qwen2_vl import model, processor


app = FastAPI()


@app.get("/api/image")
def read_root(image_url: str, prompt: str = '描述一下这张图片', resized_width: int =200, resized_height: int =200):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_url,
                    "resized_width": resized_width,
                    "resized_height": resized_height,
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
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(f'返回结果: {output_text}')
    return {"output": output_text}