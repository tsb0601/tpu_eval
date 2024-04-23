from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch
from PIL import Image
class llava_hf():
    def __init__(self, model_path="llava-hf/llava-1.5-7b-hf", device_map="auto") -> None:

        model = LlavaForConditionalGeneration.from_pretrained(model_path, low_cpu_mem_usage=True, device_map=device_map, torch_dtype=torch.float16)
        processor = AutoProcessor.from_pretrained(model_path)
        self.model = model
        self.processor = processor

    def inference(self, image, question, temperature=0, top_p=None, num_beams=1, max_length=1024):
        if image is not None:
            prompt = "<image>\nUSER: {}\nASSISTANT:".format(question)
        else:
            prompt = "USER: {}\nASSISTANT:".format(question)
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        for k,v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.cuda()

        if temperature > 0:
            generate_ids = self.model.generate(**inputs, do_sample=True, temperature=temperature, top_p=top_p, num_beams=num_beams,max_length=max_length)
        else:
            generate_ids = self.model.generate(**inputs, do_sample=False, max_length=max_length)
        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        return response.split("\nASSISTANT: ")[-1].strip()

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
import os


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if pil_img.mode == "L":
        background_color = 255
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            new_images.append(image)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images

class llava_origin():
    def __init__(self, model_path="liuhaotian/llava-v1.5-7b", device_map="auto", conv_mode="vicuna_v1") -> None:

        disable_torch_init()
        model_path = os.path.expanduser(model_path)
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base=None, model_name=model_name, device_map=device_map)

        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        # conv_mode = "cambrian_v1"
        self.conv_mode = conv_mode

        self.model.config.image_grid_pinpoints = [[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]]

    def inference(self, image, question, temperature=0, top_p=None, num_beams=1, max_length=1024):
        if image is None:
            images = None
            image_sizes = None
        else:
            if self.model.config.mm_use_im_start_end:
                question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
            else:
                question = DEFAULT_IMAGE_TOKEN + '\n' + question
            image_tensor = process_images([image], self.image_processor, self.model.config)[0]
            images = image_tensor.unsqueeze(0).half().cuda()
            image_sizes=[image.size]
        
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images,
                image_sizes=image_sizes,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                no_repeat_ngram_size=3,
                max_new_tokens=max_length,
                use_cache=True)

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs