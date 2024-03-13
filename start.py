from run_exps import create_commands, create_submit_files
from utils import average_submissions
import os



if __name__ == "__main__":
    create_submit_files(startmode = True)
    average_submissions(dir_path = "averaging",new_name = 'Yte.csv')
    files = os.listdir('averaging')
    for file in files:
        os.remove(os.path.join('averaging',file))
    os.rmdir("averaging")
    print("Final submission file created")
    
