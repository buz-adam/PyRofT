from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from flask import Flask,request,Response
import contour
import argparse
import cv2
import numpy as np
import tqdm
import calc
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.model import SODModel
from src.dataloader import InfDataloader, SODLoader
result="none"

def parse_arguments():
    parser = argparse.ArgumentParser(description='Parameters to train your model.')
    parser.add_argument('--imgs_folder', default='./upload', help='Path to folder containing images', type=str)
    parser.add_argument('--model_path', default='models/best-model_epoch-204_mae-0.0505_loss-0.1370.pth', help='Path to model', type=str)
    parser.add_argument('--use_gpu', default=True, help='Whether to use GPU or not', type=bool)
    parser.add_argument('--img_size', default=256, help='Image size to be used', type=int)
    parser.add_argument('--bs', default=24, help='Batch Size for testing', type=int)

    return parser.parse_args()


def run_inference(args):
    # Determine device
    i=0
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device(device='cuda')
    else:
        device = torch.device(device='cpu')

    # Load model
    model = SODModel()
    chkpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(chkpt['model'])
    model.to(device)
    model.eval()

    inf_data = InfDataloader(img_folder=args.imgs_folder, target_size=args.img_size)
    # Since the images would be displayed to the user, the batch_size is set to 1
    # Code at later point is also written assuming batch_size = 1, so do not change
    inf_dataloader = DataLoader(inf_data, batch_size=1, shuffle=True, num_workers=2)

    print("Press 'q' to quit."+"\n img size: "+str(args.img_size))
    with torch.no_grad():
        for batch_idx, (img_ss,img_np, img_tor) in enumerate(inf_dataloader, start=1):
            img_tor = img_tor.to(device)
            pred_masks, _ = model(img_tor)
            cv2.imread(args.imgs_folder+'/1.jpg')

            # Assuming batch_size = 1
            img_np = np.squeeze(img_np.numpy(), axis=0)
            ada = img_np
            minEdge=[[0,img_ss[0]],[1,img_ss[1]]][img_ss[0]>img_ss[1]]
            maxEdge=[img_ss[0],img_ss[1]][img_ss[0]<img_ss[1]]
            print(minEdge)
            paddR=maxEdge/256.0
            differ=int((256-(minEdge[1]/paddR))/2)
            print(str(img_ss)+"\n"+str(minEdge)+"minedge "+str(differ) )
            min=[img_np[differ:257-differ,:],img_np[:,differ:257-differ]][minEdge[0]!=0]
            print("print shape: "+str(min.shape))
            cv2.imwrite('images/x.jpg', min)
            print("mindiids: "+str(img_np.shape))
            img_np = img_np.astype(np.uint8)
            imk=img_np
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            blank= np.zeros([min.shape[0],min.shape[1],3],dtype=np.uint8)
            print("img ss: "+str(img_ss[0][0]))
            pred_masks_raw = np.squeeze(pred_masks.cpu().numpy(), axis=(0, 1))
            pred_masks_round = np.squeeze(pred_masks.round().cpu().numpy(), axis=(0, 1))

            print(str(min.shape[0])+'-'+str(min.shape[1])+'-'+str(min.shape[2]))
            if min.shape[0]>=min.shape[1]:
                dif=int((min.shape[0]-min.shape[1])/2)
                print(str(min.shape[0]) + '-' + str(min.shape[1]) + '-' + str(min.shape[2]))
                print("difference asf1:"+str(dif))
                print(min.shape)
                ada=ada[:,dif:(256-dif),:]
            else:
                print(str(min.shape[0]) + '-' + str(min.shape[1]) + '-' + str(min.shape[2]))
                dif = int((min.shape[1] - min.shape[0])/2)
                print("difference asf2:" + str(dif))
                print(min.shape)
                ada = ada[dif:(256 - dif),:,: ]
            print('Image :', batch_idx)
            ada = cv2.cvtColor(ada, cv2.COLOR_BGR2RGB)
         #   cv2.imshow('Input Image', ada)

          #  cv2.imshow('Generated Saliency Mask', pred_masks_raw)
           # cv2.imshow('Rounded-off Saliency Mask', pred_masks_round)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            cv2.imwrite('images/out_' + str(i) + '.jpg',[pred_masks_raw[differ:257-differ,:]*255,pred_masks_raw[:,differ:257-differ]*255][minEdge[0]!=0])
            #cv2.imwrite('images/in_' + str(i) + '.jpg',cv2.cvtColor([ada[differ:257-differ,:]*255,ada[:,differ:257-differ]*255][minEdge[0]!=0], cv2.COLOR_BGR2RGB))
         #   cv2.imwrite('images/out_' + str(i) + '.jpg', [pred_masks_raw[differ:257-differ,:]*255,pred_masks_raw[:,differ:257-differ]*255][minEdge[0]!=0])
            l=contour.cnt('images/out_' + str(i) + '.jpg', ada, blank)
            check=False
            if l==1:
             result=calc.calc('results/z.jpg')
            else:
             result="mux"
            i+=1
            key = cv2.waitKey(0)
            if key == ord('q'):
                break
    return result


def calculate_mae(args):
    # Determine device
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device(device='cuda')

    else:
        device = torch.device(device='cpu')

    # Load model
    model = SODModel()
    chkpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(chkpt['model'])
    model.to(device)
    model.eval()

    test_data = SODLoader(mode='test', augment_data=False, target_size=args.img_size)
    test_dataloader = DataLoader(test_data, batch_size=args.bs, shuffle=False, num_workers=2)

    # List to save mean absolute error of each image
    mae_list = []
    with torch.no_grad():
        for batch_idx, (inp_imgs, gt_masks) in enumerate(tqdm.tqdm(test_dataloader), start=1):
            inp_imgs = inp_imgs.to(device)
            gt_masks = gt_masks.to(device)
            pred_masks, _ = model(inp_imgs)

            mae = torch.mean(torch.abs(pred_masks - gt_masks), dim=(1, 2, 3)).cpu().numpy()
            mae_list.extend(mae)

    print('MAE for the test set is :', np.mean(mae_list))
rt_args=""
app = Flask(__name__)
@app.route('/api/upload', methods=['POST'])
def upload():
        image = np.asarray(bytearray(request.files['image'].read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        cv2.imwrite('upload/inp.jpg',image)
        print("begins\n")
        """   saliency_map_gbvs = gbvs.compute_saliency(image)
        oname = "./vi/{}.jpg".format(1)"""
        #t_args = parse_arguments()
        #calculate_mae(rt_args)
        val=run_inference(rt_args)
        print("-----:::"+str(val)+":::-----")

        return Response(response=str(val),status=200,mimetype="application/json")

if __name__ == '__main__':
    rt_args = parse_arguments()
    calculate_mae(rt_args)
    app.run(host="0.0.0.0", port=5000)
    """
    
    run_inference(rt_args)"""
