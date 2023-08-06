#! /bin/bash

mkdir models
mkdir vae
mkdir controlnet
mkdir lora
cd models

wget -O sdxl-base-1.0.safetensors https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors
wget -O sdxl-refiner-1.0.safetensors https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors

cd ../vae

wget -O sdxl-vae.safetensors https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors
