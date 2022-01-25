#!/usr/bin/env python
#coding=utf-8
# Zilin Ren

import argparse
from scripts.convert import convert
from scripts.predict import predict

def main():
    args = get_args()
    args.func(args)


def get_args():
    parser = argparse.ArgumentParser(description='SGAN: Semi-supervised learning method to predict oncogenicity of variants')
    subparsers = parser.add_subparsers(help='sub-command help')
    subparsers.required = True
    
    convert_parser = subparsers.add_parser('convert', help = "Convert cancervar output and annovar output into model input data")
    
    convert_parser.add_argument( "-a", "--annovar_path", help = "the path to annovar file", \
        default=None, type=str, required=True )
    
    convert_parser.add_argument( "-c", "--cancervar_path", help = "the path to cancervar file", \
        default=None, type=str, required=True )    

    convert_parser.add_argument( "-m", "--method", help = "output evs features or ensemble features (option: evs, ensemble)", \
        default="ensemble", type=str, required=False )

    convert_parser.add_argument( "-n", "--missing_count", help="variant with more than N missing features will be discarded, (default: 5)", \
        default=5, type=int, required=False )

    convert_parser.add_argument( "-d", "--database", help="database for feature normalization", \
        default = None, type=str, required=True)

    convert_parser.add_argument( "-o", "--output", help="the path to output", \
        default=None, type=str, required=True )

    convert_parser.set_defaults(func=convert)


    ## predict
    predict_parser = subparsers.add_parser('predict', help='Predict oncogenicity of variants')

    predict_parser.add_argument( "-i", "--input",  help = "the path to input feature", \
        default=None, type=str, required=True )

    predict_parser.add_argument( "-v", "--cancervar_path", help = "the path to cancervar file", \
        default=None, type=str, required=True )

    predict_parser.add_argument( "-m", "--method", help = "use evs features or ensemble features (option: evs, ensemble)", \
        default="ensemble", type=str, required=False )

    predict_parser.add_argument( "-d", "--device", help = "device used for dl-based predicting (option: cpu, cuda)", default="cpu", type=str, required=False )

    predict_parser.add_argument( "-c", "--config", help = "the path to trained model file", \
        default=None, type=str, required=True )

    predict_parser.add_argument( "-o", "--output", help = "the path to output", \
        default=None, type=str, required=True )

    predict_parser.set_defaults(func=predict)

    ## run
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()