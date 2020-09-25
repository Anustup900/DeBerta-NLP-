import os
import re
from extract_files import extract_files

original_dir = "../Dataset/sorted_data_acl"

data_list = ["/books","/dvd","/electronics","/kitchen_and_housewares"]
	


for i in range(4):
	dir_path = original_dir+data_list[i]

	
	pos_file = dir_path+"/positive.review"
	neg_file = dir_path+"/negative.review"

	extract_files(pos_file,neg_file,i)
