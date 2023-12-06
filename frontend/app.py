
import requests
import io
import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image, ImageEnhance
st.set_option('deprecation.showfileUploaderEncoding', False)

# Upload an image and set some options for demo purposes
st.header("CloseCV: Text-to-Image Generation with Art")
img_file = st.sidebar.file_uploader(label='Upload a file', type=['png', 'jpg'])
realtime_update = st.sidebar.checkbox(label="Update in Real Time", value=True)
# box_color = st.sidebar.color_picker(label="Box Color", value='#0000FF')
box_color ='#0000FF'
aspect_choice = st.sidebar.radio(label="Aspect Ratio", options=["1:1", "16:9", "4:3", "2:3", "Free"])
# contrast_level = st.sidebar.slider('Contrast level', min_value=0.5, max_value=3.5, value=1.0)
# brightness_level = st.sidebar.slider('Brightness level', min_value=0.5, max_value=3.5, value=1.0)
# sharpness_level = st.sidebar.slider('Sharpness level', min_value=0.5, max_value=3.5, value=1.0)
aspect_dict = {
    "1:1": (1, 1),
    "16:9": (16, 9),
    "4:3": (4, 3),
    "2:3": (2, 3),
    "Free": None
}
aspect_ratio = aspect_dict[aspect_choice]

def pil_to_binary(image, enc_format = "png"):
	"""Convert PIL Image to base64-encoded image"""
	buffer = io.BytesIO()
	image.save(buffer, format=enc_format)
	buffer.seek(0)
	return buffer

if img_file:
	img = Image.open(img_file)
# 	contr_enhancer = ImageEnhance.Contrast(img)
# 	img = contr_enhancer.enhance(contrast_level)
# 	bright_enhancer = ImageEnhance.Brightness(img)
# 	img = bright_enhancer.enhance(brightness_level)
# 	sharp_enhancer = ImageEnhance.Sharpness(img)
# 	img = sharp_enhancer.enhance(sharpness_level)
	if not realtime_update:
		st.write("Double click to save crop")
	# Get a cropped image from the frontend
	cropped_img = st_cropper(img, realtime_update=realtime_update, box_color=box_color, aspect_ratio=aspect_ratio, return_type='box')
# 	print(cropped_img.values())
	cropped_img = list(map(int,cropped_img.values()))
	temp_prompt = f"A cat is eating a fish. <{cropped_img[0]}> <{cropped_img[1]}> <{cropped_img[2]}> <{cropped_img[3]}> white cat"
	print(temp_prompt)
	temp_prompt = st.text_input('prompt_example',temp_prompt)
	# Manipulate cropped image at will
	
	
	if st.button('submit'):


		# POST the image data as json to the FastAPI server
		url = "http://127.0.0.1:8000/image_gen"
		temp_json_1 = {
			  "prompt": temp_prompt,
			  "outdir": "outputs",
			  "laion400m": False,
			  "plms": True,
			  "ddim_steps": 50,
			  "ddim_eta": 0,
			  "n_iter": 4,
			  "H": 512,
			  "W": 512,
			  "C": 4,
			  "f": 8,
			  "n_samples": 1,
			  "n_rows": 0,
			  "scale": 2,
			  "fixed_code": False,
			  "skip_grid": False,
			  "skip_save": False,
			  "from_file": False,
			  "config": "ReCo/configs/stable-diffusion/v1-inference-box.yaml",
			  "ckpt": "ReCo/logs/reco_laion_1232.ckpt",
			  "seed": 42,
			  "precision": "autocast",
			  "embedding_path": False
		}
		response = requests.post(url, json=temp_json_1)

		style_prompt = st.text_input('style_prompt_example','a photo of a cartoon cat on the concert field.')
		negative_prompt = st.text_input('negative_prompt_example','ugly, blurry, low res')
		url2 =  "http://127.0.0.1:8000/style_gen"
		data_path = response.json()['res_image_path']
			
		temp_json_2 = {
			# 'data_path':response.json()['res_image_path']
			'data_path': data_path
		}
		response2 = requests.post(url2,json=temp_json_2)

		url3 = "http://127.0.0.1:8000/style_gen2"
		temp_json_3 = {
		    'prompt': style_prompt,
		    'image_path': response.json()['res_image_path'],
		    'output_path':"/home/cvlserver/ssd2tb/hkt/backend/pnp_diffusers/PNP-results/final_result",
		    'negative_prompt': negative_prompt,
		}
		response3 = requests.post(url3,json=temp_json_3)
		import os
		# # Print the response or do something else...
		final_img = Image.open(os.path.join(response3.json()['final_output_path'],f'output-{style_prompt}.png'))
		st.image(final_img)
		