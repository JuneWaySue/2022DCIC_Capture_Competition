from torch.nn import Module
from torch.nn import Sequential
from torch.nn import Linear
from torch.nn import Dropout

try:
    import timm
except:
    ! pip install timm
    import timm

#resnet26d,0.68
#resnetrs50,0.76
#gluon_resnet152_v1d,0.76
class TimmModels(Module):
    def __init__(self, model_name='resnetrs50',pretrained=True, num_classes=62*4):
        super(TimmModels, self).__init__()
        self.m = timm.create_model(model_name,pretrained=pretrained)
        model_list = list(self.m.children())
        model_list[-1] = Linear(
            in_features=model_list[-1].in_features, 
            out_features=num_classes, 
            bias=True
        )
#         model_list.insert(-1,Dropout(0.5))
        self.m = Sequential(*model_list)
        
    def forward(self, image):
        out = self.m(image)
        return out
    
class TimmModels_4fc(Module):
    def __init__(self, model_name='resnetrs50',pretrained=True, num_classes=62):
        super(TimmModels_4fc, self).__init__()
        self.m = timm.create_model(model_name,pretrained=pretrained)
        model_list = list(self.m.children())
        in_features = model_list[-1].in_features
        self.m = Sequential(*model_list[:-1])
        self.fc1 = Sequential(Dropout(0.25),Linear(in_features=in_features, out_features=num_classes, bias=True))
        self.fc2 = Sequential(Dropout(0.25),Linear(in_features=in_features, out_features=num_classes, bias=True))
        self.fc3 = Sequential(Dropout(0.25),Linear(in_features=in_features, out_features=num_classes, bias=True))
        self.fc4 = Sequential(Dropout(0.25),Linear(in_features=in_features, out_features=num_classes, bias=True))
        
    def forward(self, image):
        out = self.m(image)
        y1 = self.fc1(out)
        y2 = self.fc2(out)
        y3 = self.fc3(out)
        y4 = self.fc4(out)
        return y1,y2,y3,y4