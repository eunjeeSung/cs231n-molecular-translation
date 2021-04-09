import torch
import torch.nn as nn
import torchvision

from util import transform

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    MODEL_PATH='../input/model18train/modeltrain18_2'
    model.load_state_dict(torch.load(MODEL_PATH))
    model=model.to(device)

    test = pd.read_csv('../input/sample_submission.csv')
    test['file_path'] = test['image_id'].progress_apply(get_test_file_path)        
    
    dataset_test=InputDatasetTest(test['file_path'].to_numpy(),transform)
    dataloader_test=DataLoader(
        dataset=dataset_test,
        batch_size=300,
        shuffle=False,
        num_workers=6)

    #print out a caption to make sure model working correctly
    model.eval()
    itr=iter(dataloader_test)
    img,idx=next(itr)
    print(img.shape)
    print(img[0:5].shape)
    features=model.encoder(img[0:5].to(device))
    caps = model.decoder.generate_caption(features,stoi=stoi,itos=itos)
    captions=tensor_to_captions(caps)
    plt.imshow(img[0].numpy().transpose((1,2,0)))
    print(captions)    

    model.eval()
    with torch.no_grad():
        for i,batch in enumerate(dataloader_test):
            img,idx=batch[0].to(device),batch[1]
            features=model.encoder(img)
            caps=model.decoder.generate_caption(features,stoi=stoi,itos=itos)
            captions=tensor_to_captions(caps)
            test['InChI'].loc[idx]=captions
            if i%1000==0: print(i)    

    output_cols = ['image_id', 'InChI']
    test[output_cols].to_csv('submission.csv',index=False)
    print(test[output_cols].head())