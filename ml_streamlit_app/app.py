import streamlit as st
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import pandas as pd
from PIL import Image
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torchvision
from pathlib import Path

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = Path(__file__).parents[0] / 'model_torch_MNIST_plus_CNN_98_5_streamlit.chk'
# Specify canvas parameters in application
drawing_mode = "freedraw"

stroke_width = 9
if drawing_mode == 'point':
    point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
stroke_color = "#eee"

# Create a canvas component
st.title('MNIST Live Classifier')
left,right = st.columns(2)

with left:
    st.write('### Draw a number (0-9)')
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color= "#000000",
        #background_image=Image.open(bg_image) if bg_image else None,
        #update_streamlit=realtime_update,
        height= 280,
        width = 280,
        drawing_mode=drawing_mode,
        point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
        key="canvas",
    )
probabilities = np.zeros(10)
output = 0
if np.sum(canvas_result.image_data) != 19992000:
    image_np = np.array(canvas_result.image_data)
    input_image = Image.fromarray(image_np.astype('uint8'), 'RGBA')
    input_image_gs = input_image.convert('L')
    input_image_gs_np = np.asarray(input_image_gs.getdata()).reshape(280,280)
    input_image_gs.save('temp_for_cv2.jpg')
    image = cv2.imread('temp_for_cv2.jpg', 0)
    x,y,w,h = cv2.boundingRect(image)
    ROI = image[y:y+h, x:x+w]
    mask = np.zeros([ROI.shape[0]+10,ROI.shape[1]+10])
    width, height = mask.shape
    x = width//2 - ROI.shape[0]//2
    y = height//2 - ROI.shape[1]//2
    x = width//2 - ROI.shape[0]//2
    y = height//2 - ROI.shape[1]//2
    mask[y:y+h, x:x+w] = ROI
    output_image = Image.fromarray(mask)
    compressed_output_image = output_image.resize((22,22), Image.BILINEAR)
    image_tensor = torchvision.transforms.ToTensor()(compressed_output_image)
    image_tensor /= 255
    image_tensor = torch.nn.functional.pad(image_tensor, (3,3,3,3), "constant", 0)
    image_tensor = torchvision.transforms.Normalize((0.1281), (0.3043))(image_tensor)
    im = Image.fromarray(image_tensor.detach().cpu().numpy().reshape(28,28), mode='L')
    model = torch.load(MODEL_PATH)
    with torch.no_grad():
        output0 = model(torch.unsqueeze(image_tensor, dim=0).to(device=DEVICE))
        certainty, output = torch.max(output0[0], 0)
        certainty = certainty.clone().cpu().item()
        output = output.clone().cpu().item()
        certainty1, output1 = torch.topk(output0[0],3)
        certainty1 = certainty1.clone().cpu()#.item()
        output1 = output1.clone().cpu()#.item()
        sm = torch.nn.Softmax()
        probabilities = sm(output0).cpu().detach().numpy()[0]
        

import plotly.express as px
    
with right:
    import pandas as pd
    random_x= np.random.randint(1, 101, 10)
    random_y= np.random.randint(1, 101, 10)
    temp = pd.DataFrame({"Predicted Digit" : np.arange(10), "Probability" : probabilities})
    fig = px.bar(temp, x = "Predicted Digit", y = "Probability", width=400, height=400)
    fig.update_layout(
    xaxis = dict(
        tickmode = 'linear',
        tick0 = 0,
        dtick = 1
        )
    )
    st.plotly_chart(fig)

right, middle, left = st.columns(3)
with middle:
    st.write('### Prediction: ' + str(output)) 
