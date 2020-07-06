from layers import *  
#Create classification and regression model
def create_model(out_features:int=1, dropout_ps:float=0.5, ymin:float=None, ymax:float=None):
    '''
    out_features: number of classes for classification models.
    '''
    block1, block2= get_early_layers(), get_mid_layers()
    final = get_final_layers(out_features=out_features, dropout_ps=dropout_ps, ymin=ymin, ymax=ymax)

    layer_groups = block1+block2+final
    model = nn.Sequential(*layer_groups)
    return model, layer_groups
    
def get_early_layers():
    return [nn.Sequential(conv_layer3d(1,64),
                        nn.MaxPool3d(kernel_size=2, stride=2),
                        conv_and_res3d(64,64),
                        conv_and_res3d(64,64),
                        nn.MaxPool3d(kernel_size=2, stride=2))]   

def get_mid_layers():
    return [nn.Sequential(conv_and_res3d(64,128),
                        nn.MaxPool3d(kernel_size=2, stride=2),
                        conv_and_res3d(128,256),
                        nn.MaxPool3d(kernel_size=2, stride=2),
                        conv_and_res3d(256,512))]   


def get_final_layers(out_features, dropout_ps, ymin:float=None, ymax:float=None):
    ps_1, ps_2 = get_dropout(dropout_ps=dropout_ps)
    
    final_layers = [AdaptiveConcatPool3d(1), 
                    nn.Flatten(),
                    nn.BatchNorm1d(1024),
                    nn.Dropout(p=ps_1),
                    nn.Linear(in_features=1024, out_features=512), 
                    nn.ReLU(), 
                    nn.BatchNorm1d(512), 
                    nn.Dropout(p=ps_2),
                    nn.Linear(in_features=512, out_features=out_features)]
    
    if ymin and ymax: 
        final_layers.append(ScaledSigmoid(ymin=ymin, ymax=ymax))
    
    return [nn.Sequential(*final_layers)]
        
def get_dropout(dropout_ps): 
    if type(dropout_ps) is list: return dropout_ps[0], dropout_ps[1] 
    else: return dropout_ps/2, dropout_ps