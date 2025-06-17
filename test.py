import torch 

img = torch.rand((5, 3 , 17, 64, 64))


def add_pading(img): 
    B , C , T , H , W = img.shape 
    print(T )
    padd = abs(T - 32)
    re_img = img[:, :, :-1 , : , :].repeat(1, 1 , padd, 1 ,1 )
    print(re_img.shape)


add_pading(img)
