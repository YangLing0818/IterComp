import json
import ImageReward as RM
import torch
from tqdm import tqdm
import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


def get_data_iterative_reward(reward_model_path, initial_dataset_json_path, image_path, save_path):
    with open(initial_dataset_json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    reward_model = RM.ImageReward(device='cuda', med_config='train/config/med_config.json').to('cuda')
    my_state_dict = torch.load(reward_model_path, map_location='cpu')
    my_state_dict = {key.split('module.')[1]: value for key, value in my_state_dict.items()}
    reward_model.load_state_dict(my_state_dict,strict=False)
    model_gallery = ["image_sd15", "image_sdxl", "image_sd3", "image_flux", "image_rpg", "image_instancediffusion"]
    for i in tqdm(range(len(data))):
        index = data[i]["index"]
        prompt = data[i]["prompt"]
        ranking = [int(char) for char in data[i]["rank"]]
        initial_rank = ranking.copy()
        image_base_refine = f'{image_path}/sdxl_iteration1_prompt_{index}.png'
        reward_base_refine = reward_model.score(prompt, image_base_refine)
        image_add_model = f'{image_path}/omost_prompt_{index}.png'
        reward_add_model = reward_model.score(prompt, image_add_model)
        insterted = False
        for j in range(len(ranking) - 1, 0, -1):
            image = data[i][model_gallery[ranking[j] - 1]]
            reward_image = reward_model.score(prompt, image)
            if reward_base_refine > reward_image:
                continue  
            else:
                ranking.insert(j + 1, len(model_gallery) + 1)
                insterted = True
                break
        if not insterted:
            ranking.insert(0, len(model_gallery) + 1)
        model_gallery.append(f"image_sdxl_iteration1")
        insterted = False
        for j in range(len(ranking) - 1, 0, -1):
            if ranking[j] == len(model_gallery) + 1:
                image = image_base_refine
            else:
                image = data[i][model_gallery[ranking[j] - 1]]
            reward_image = reward_model.score(prompt, image)
            if reward_add_model > reward_image:
                continue  
            else:
                ranking.insert(j + 1, len(model_gallery) + 2)
                insterted = True
                break
        if not insterted:
            ranking.insert(0, len(model_gallery) + 2)
        with open(save_path, "a") as f:
            json.dump({"index": data[i]["index"],
                        "prompt": data[i]["prompt"],
                        "image_sd15": data[i]["image_sd15"],
                        "image_sdxl": data[i]["image_sdxl"],
                        "image_sd3": data[i]["image_sd3"],
                        "image_flux": data[i]["image_flux"],
                        "image_rpg": data[i]["image_rpg"],
                        "image_instancediffusion": data[i]["image_instancediffusion"],
                        "image_sdxl_iteration1": image_base_refine,  
                        "image_omost": image_add_model,
                        "initial_rank": initial_rank,
                        "rank": ranking   
                        }, f, ensure_ascii=False, indent=4)
            f.write(',\n')


get_data_iterative_reward(reward_model_path='change to your reward model path',
                          initial_dataset_json_path=' change to your initial dataset path',
                          image_path='change to your image path',
                          save_path='change to your save path')
