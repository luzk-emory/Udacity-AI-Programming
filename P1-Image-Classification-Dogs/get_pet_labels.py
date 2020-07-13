#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/get_pet_labels.py
#                                                                             
# PROGRAMMER: Zhikun Lu
# DATE CREATED:  May 15,2020                                
# REVISED DATE:  May 15,2020
# PURPOSE: Create the function get_pet_labels that creates the pet labels from 
#          the image's filename. This function inputs: 
#           - The Image Folder as image_dir within get_pet_labels function and 
#             as in_arg.dir for the function call within the main function. 
#          This function creates and returns the results dictionary as results_dic
#          within get_pet_labels function and as results within main. 
#          The results_dic dictionary has a 'key' that's the image filename and
#          a 'value' that's a list. This list will contain the following item
#          at index 0 : pet image label (string).
#
##
# Imports python modules
from os import listdir

# TODO 2: Define get_pet_labels function below please be certain to replace None
#       in the return statement with results_dic dictionary that you create 
#       with this function
# 
def get_pet_labels(image_dir):
    """
    Creates a dictionary of pet labels (results_dic) based upon the filenames 
    of the image files. These pet image labels are used to check the accuracy 
    of the labels that are returned by the classifier function, since the 
    filenames of the images contain the true identity of the pet in the image.
    Be sure to format the pet labels so that they are in all lower case letters
    and with leading and trailing whitespace characters stripped from them.
    (ex. filename = 'Boston_terrier_02259.jpg' Pet label = 'boston terrier')
    Parameters:
     image_dir - The (full) path to the folder of images that are to be
                 classified by the classifier function (string)
    Returns:
      results_dic - Dictionary with 'key' as image filename and 'value' as a 
      List. The list contains for following item:
         index 0 = pet image label (string)
    """
    
    # Step 1: read file names    

    # Retrieve the filenames from folder pet_images/
    filenames = listdir(image_dir)

    # Step 2: get labels
    pet_labels = []

    for name in filenames:
        name_list = name.lower().split('_')
        for idx, item in enumerate(name_list):
            if not item.isalpha():
                name_list.pop(idx)
        label = ' '.join(name_list)
        pet_labels.append(label)
    
    # Creates empty dictionary named results_dic
    results_dic = dict()

    for idx in range(0, len(filenames), 1):
        if (filenames[idx] not in results_dic) and (filenames[idx][0] != '.'):
             results_dic[filenames[idx]] = [pet_labels[idx]]
        else:
             print("** Warning: duplicate key or .files:", filenames[idx])




    # Replace None with the results_dic dictionary that you created with this
    # function
    return results_dic


if __name__ == "__main__": 
    from get_input_args import get_input_args
    print("------------ testing ------------")
    image_dir = get_input_args().dir
    aa = get_pet_labels(image_dir)
    for key in aa:
        print()
        print("Filename  =", key)
        print("Pet Label =", aa[key][0])
        

