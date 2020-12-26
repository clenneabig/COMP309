import pickle

if __name__ == '__main__':
    with open('sgdGraphData.pkl', 'rb') as f:
        hist = pickle.load(f)
        print(hist)
        print(hist.keys())
        print(hist['val_accuracy'])
        loss = 9999999
        list = hist['val_loss']
        for i in range(0,len(hist['val_loss'])):
            loss = min(loss, list[i])
        print(loss)