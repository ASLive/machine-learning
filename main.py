from ml import *
import sys

if __name__ == "__main__":
    """
       python3 main.py # load model files if available, else retrain and save
       python3 main.py retrain # retrain model and save
       python3 main.py retrain no-save # retrain the model but don't save model
       python3 main.py visualize # load model files and visualize predictions
    """
    retrain_arg = len(sys.argv) > 1 and (sys.argv[1].lower() == 'retrain')
    visualize_arg = len(sys.argv) > 1 and (sys.argv[1].lower() == 'visualize')
    no_save_arg = not (len(sys.argv) > 2 and (sys.argv[2].lower() == 'no-save'))

    run(not retrain_arg, no_save_arg, visualize_arg)
