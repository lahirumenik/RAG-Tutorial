"""
Author: Lahiru Menikdiwela
Date: 16 February 2025
"""

import os
from transformers import AutoModel, AutoTokenizer

model_name = "Qwen/Qwen2.5-0.5B-Instruct" 
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


save_directory = "./model"

if not os.path.exists(save_directory):
    os.makedirs(save_directory)
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
