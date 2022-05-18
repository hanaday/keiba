import os
import sys


if __name__ == '__main__':
    x = input(">>>")

    command = x.split()[0]
    data = x.split()[1]
    cfg = x.split()[2]
    mdl = x.split()[3]

    if os.path.splitext(data)[1] != ".data":
        print('Error: The file format of the data file does not match', file=sys.stderr)
        sys.exit(1)
    if os.path.splitext(cfg)[1] != ".cfg":
        print('Error: The file format of the cfg file does not match', file=sys.stderr)
        sys.exit(1)
    if os.path.splitext(mdl)[1] != ".mdl":
        print('Error: The file format of the mdl file does not match', file=sys.stderr)
        sys.exit(1)

    if command == "train":
        from train import TRAIN
        
        train = TRAIN(cfg)
        train.train()
    elif command == "eval":
        from eval import EVAL

        eval = EVAL(data, cfg, mdl)
        eval.eval()
    elif command == "predict":
        from predict import PREDICT

        predict = PREDICT(cfg, data, mdl)
        predict.predict()
    else:
        print('Error: This is an incorrect command', file=sys.stderr)
        sys.exit(1)