import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from vit_model import vit_base_patch16_224_in21k as create_model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # 输入待识别图像
    img_path = "./00_storage/flower_photos/00_new_for_prediction/bronco.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension （在dim0处增加一个维度）
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './00_storage/class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = create_model(num_classes=5, has_logits=False).to(device)
    # load model weights
    model_weight_path = "./00_storage/weights/model-9.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # predict class
        # 将输入张量形状中的1 去除并返回
        # 如果输入是形如(A×1×B×1×C×1×D)，那么输出形状就为： (A×B×C×D)
        output = torch.squeeze(model(img.to(device))).cpu()
        # 每个位置的概率计算
        predict = torch.softmax(output, dim=0)
        # PyTorch中用来返回指定维度最大值的序号的函数
        predict_cla = torch.argmax(predict).numpy()

    # 用pred_cal序号去相应的list取对应内容
    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()
