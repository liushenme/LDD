import argparse

import toml

from dataset.ffv import LAVDFDataModule
from model import *
import torch.multiprocessing
import torch
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import numpy as np

parser = argparse.ArgumentParser(description="BATFD evaluation")
parser.add_argument("--config", type=str)
parser.add_argument("--data_root", type=str)
parser.add_argument("--checkpoint", type=str)
parser.add_argument("--gpus", type=str, default=None)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--name", type=str)
parser.add_argument("--model", type=str, default=None)

def class_wise_accuracy(y_true, y_prob, threshold):
    acc_0 = []
    acc_1 = []
    for th in threshold:
        y_pred = (y_prob >= th).astype(int)
        

        class_0_indices = np.where(y_true == 0)[0]
        class_1_indices = np.where(y_true == 1)[0]
        
        acc_class_0 = accuracy_score(y_true[class_0_indices], y_pred[class_0_indices])
        acc_class_1 = accuracy_score(y_true[class_1_indices], y_pred[class_1_indices])
        
        acc_0.append(acc_class_0)
        acc_1.append(acc_class_1)        

    return {
        'class_0_accuracy': acc_0,
        'class_1_accuracy': acc_1,
    }


def calculate_accuracy(labels, scores, threshold):
    accs = []
    for th in threshold:
        predictions = (scores >= th).astype(int)
        accuracy = np.mean(predictions == labels)
        accs.append(accuracy)        

    return accs

def evaluate(model, data_loader):

    output_dict = forward(
        model=model, 
        data_loader=data_loader) 
    #print(output_dict.shape)

    statistics = {}

    # Clipwise statistics
    statistics['AUC'] = roc_auc_score(
        output_dict['target'], output_dict['clipwise_output'][:, 1])

    statistics['ACC'] = calculate_accuracy(
        output_dict['target'], output_dict['clipwise_output'][:, 1], [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

    statistics['F1'] = f1_score(
        output_dict['target'], [1 if p >= 0.5 else 0 for p in output_dict['clipwise_output'][:, 1]])

    statistics['ACC_all'] = class_wise_accuracy(
        output_dict['target'], output_dict['clipwise_output'][:, 1], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    return statistics, output_dict

def move_data_to_device(x, device):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        # raise Exception("Error!")
        return x

    return x.to(device)

def append_to_dict(dict, key, value):
    if key in dict.keys():
        dict[key].append(value)
    else:
        dict[key] = [value]

def forward(model, data_loader):

    device = next(model.parameters()).device #see model device
    output_dict = {}
    
    # Evaluate on mini-batch
    for n, wav_dic in enumerate(data_loader):
        video, audio, n_frames, target, name = wav_dic

        video = move_data_to_device(video, device)
        audio = move_data_to_device(audio, device)
        n_frames = move_data_to_device(n_frames, device)

        with torch.no_grad():
            model.eval()
            output = model(video, audio, n_frames)
            if isinstance(output, tuple) :
                batch_output = output[0]
            else:
                batch_output = output
            batch_output = torch.nn.functional.sigmoid(batch_output)
            for i in range(len(batch_output)):
                out = [name[i], batch_output[i][1].cpu().numpy()]
                print(",".join(str(i) for i in out))

        append_to_dict(output_dict, 'clipwise_output', 
            batch_output.data.cpu().numpy())
        append_to_dict(output_dict, 'target', target)
        
    for key in output_dict.keys():
        output_dict[key] = np.concatenate(output_dict[key], axis=0)

    return output_dict




if __name__ == '__main__':
    
    torch.multiprocessing.set_sharing_strategy('file_system')
    args = parser.parse_args()
    config = toml.load(args.config)
    model_name = args.name
    alpha = config["soft_nms"]["alpha"]
    t1 = config["soft_nms"]["t1"]
    t2 = config["soft_nms"]["t2"]

    print(args.model)
    print(args.batch_size)
    model = eval(args.model)(
        v_encoder=config["model"]["video_encoder"]["type"],
        a_encoder=config["model"]["audio_encoder"]["type"],
        weight_decay=config["optimizer"]["weight_decay"],
        learning_rate=config["optimizer"]["learning_rate"],
        distributed=None
    )


    # prepare dataset
    dm = LAVDFDataModule(root=args.data_root,
        batch_size=args.batch_size, num_workers=0,
        get_meta_attr=model.get_meta_attr)
    dm.setup()


    # prepare model
    test_loader = dm.test_dataloader

    #model = model.load_from_checkpoint(args.checkpoint)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])

    if args.gpus != None:
        model = model.cuda()

    (statistics, output_dict) = evaluate(model, test_loader())
    print('test')
    print(statistics['AUC'])
    print(statistics['ACC'])
    print(statistics['F1'])
    print(statistics['ACC_all'])

    test_loader = dm.val_dataloader
    (statistics, output_dict) = evaluate(model, test_loader())
    print('val')
    print(statistics['AUC'])
    print(statistics['ACC'])
    print(statistics['F1'])
    print(statistics['ACC_all'])

