import pandas as pd
#from tqdm import tqdm
import ast
import csv

def convert_mpii_to_coco_openpose(mpii_list):
    convert_list = [-1 for x in range(0,18)]
    #exchange_pair = ((1,8),(2,12),(3,11),(4,10),(5,8))
    convert_list[1],convert_list[2],convert_list[3],convert_list[4],convert_list[5],\
    convert_list[6],convert_list[7],convert_list[8],convert_list[9],convert_list[10],\
    convert_list[11],convert_list[12],convert_list[13] = \
    mpii_list[8],mpii_list[12],mpii_list[11],mpii_list[10],mpii_list[13],\
    mpii_list[14],mpii_list[15],mpii_list[2],mpii_list[1],mpii_list[0],\
    mpii_list[3],mpii_list[4],mpii_list[5]
    return convert_list

def convert_coco_to_openpose(coco_list):
    convert_list = [-1 for x in range(0,18)]
    convert_list[0],convert_list[1],convert_list[2],convert_list[3],convert_list[4],convert_list[5],\
    convert_list[6],convert_list[7],convert_list[8],convert_list[9],convert_list[10],\
    convert_list[11],convert_list[12],convert_list[13],convert_list[14],convert_list[15],convert_list[16],convert_list[17] = \
    coco_list[0],(coco_list[5] + coco_list[6])//2,coco_list[6],coco_list[8],coco_list[10],coco_list[5],\
    coco_list[7],coco_list[9],coco_list[12],coco_list[14],coco_list[16],\
    coco_list[11],coco_list[13],coco_list[15],coco_list[1],coco_list[2],coco_list[3],coco_list[4]
    return convert_list

def num_keypoint_filter(row,annote,filter_num,mode='None'):
    from_obj = str(row[0])
    to_obj = str(row[1])
    from_annote = annote.loc[annote['name'] == from_obj]
    to_annote = annote.loc[annote['name'] == to_obj]
    from_y_points_list = ast.literal_eval(from_annote.keypoints_y.values[0])
    from_x_points_list = ast.literal_eval(from_annote.keypoints_x.values[0])
    to_y_points_list = ast.literal_eval(to_annote.keypoints_y.values[0])
    to_x_points_list = ast.literal_eval(to_annote.keypoints_x.values[0])
    if from_y_points_list.count(-1) > filter_num or to_y_points_list.count(-1) > filter_num:
        return True
    elif mode == 'human' and (-1 in [from_y_points_list[2],from_y_points_list[5],\
    from_y_points_list[3],from_y_points_list[6],from_y_points_list[4],from_y_points_list[7],from_y_points_list[8],from_y_points_list[11],\
    from_y_points_list[9],from_y_points_list[12],from_y_points_list[10],from_y_points_list[13]]):
        return True
    elif mode == 'human' and (-1 in [to_y_points_list[2],to_y_points_list[5],\
    to_y_points_list[3],to_y_points_list[6],to_y_points_list[4],to_y_points_list[7],to_y_points_list[8],to_y_points_list[11],\
    to_y_points_list[9],to_y_points_list[12],to_y_points_list[10],to_y_points_list[13]]):
        return True
    elif mode == 'human' and (-1 in [from_x_points_list[2],from_x_points_list[5],\
    from_x_points_list[3],from_x_points_list[6],from_x_points_list[4],from_x_points_list[7],from_x_points_list[8],from_x_points_list[11],\
    from_x_points_list[9],from_x_points_list[12],from_x_points_list[10],from_x_points_list[13]]):
        return True
    elif mode == 'human' and (-1 in [to_x_points_list[2],to_x_points_list[5],\
    to_x_points_list[3],to_x_points_list[6],to_x_points_list[4],to_x_points_list[7],to_x_points_list[8],to_x_points_list[11],\
    to_x_points_list[9],to_x_points_list[12],to_x_points_list[10],to_x_points_list[13]]):
        return True
    else:
        return False

def edge_filter(row,vid_resolution = (1280,720)):
    from_obj = row[0]
    to_obj = row[1]
    from_box = from_obj.replace('.jpg','').split('/')[1].split('_')
    to_box = to_obj.replace('.jpg','').split('/')[1].split('_')
    #from_box = from_obj.replace('.png','').split('/')[1].split('_')
    #to_box = to_obj.replace('.png','').split('/')[1].split('_')
    from_box = [int(i) for i in from_box]
    to_box = [int(i) for i in to_box]
    f_x1,f_y1,f_x2,f_y2 = from_box[0],from_box[1],from_box[2],from_box[3]
    t_x1,t_y1,t_x2,t_y2 = to_box[0],to_box[1],to_box[2],to_box[3]
    if f_x1 < 40 or f_y1 < 30 or f_x2 >  vid_resolution[0] -50 or f_y2 > vid_resolution[1] -30:
        #pairs.drop(index = [index],inplace=True)  
        return True
    elif t_x1 < 40 or t_y1 < 30 or t_x2 >  vid_resolution[0] -50 or t_y2 > vid_resolution[1] -30:
        #pairs.drop(index = [index],inplace=True)
        return True
    else:
        return False
        

def relative_points(name,y_points_list,x_points_list,target_y,target_x):
    height = int(name[3])-int(name[1])
    width = int(name[2])-int(name[0])
    height_ratio = height/target_y
    width_ratio = width/target_x
    for i in range(len(y_points_list)):
        if y_points_list[i]!= -1:
            y_points_list[i] = int(y_points_list[i] / height_ratio)
        if x_points_list[i]!= -1:
            x_points_list[i] = int(x_points_list[i] / width_ratio)
    return y_points_list,x_points_list

def resize_point(annote,target_resolution=(256,256),mode="None"):
    columns = ['name','keypoints_y','keypoints_x']
    annote.columns = ['name','keypoints_y','keypoints_x']
    annote=annote.reindex(columns=columns)
    for index,row in annote.iterrows():
        y_points_list, x_points_list = ast.literal_eval(row['keypoints_y']),ast.literal_eval(row['keypoints_x'])
        if mode == 'human':
            name = annote.iat[index,0].replace('.png','').split('/')[1].split('_')
            #name = annote.iat[index,0].replace('.jpg','').split('/')[1].split('_')
        elif mode == 'vehicle':
            name = annote.iat[index,0].replace('.jpg','').split('/')[1].split('_')
        y_points_list,x_points_list = relative_points(name,y_points_list,x_points_list,target_resolution[0],target_resolution[1])
        if mode == 'human':
            #y_points_list, x_points_list = convert_mpii_to_coco_openpose(y_points_list),convert_mpii_to_coco_openpose(x_points_list)
            y_points_list, x_points_list = convert_coco_to_openpose(y_points_list),convert_coco_to_openpose(x_points_list)
        annote.iat[index,1]= str(y_points_list)
        annote.iat[index,2]= str(x_points_list)
    return annote
    #annote.to_csv('annotation-vehicle-test-256.csv',index=False,sep = ':',quoting=csv.QUOTE_NONE)


def vehicle_clean(annote,pairs,vid_resolution):
    pairs_file_name = pairs
    annote_file_name = annote
    pairs = pd.read_csv(pairs,sep=',',encoding='utf8')
    annote = pd.read_csv(annote,sep=':',encoding='utf8')
    annote = resize_point(annote,(256,256),mode='vehicle')
    pair_save_list = []
    #print('Pair Clean')
    for index,row in pairs.iterrows():
        # if edge_filter(row,(vid_resolution[0],vid_resolution[1])):
        #     pairs.drop(index = [index],inplace=True)
        #     #continue
        # elif num_keypoint_filter(row,annote,7,'vehicle'):
        #     pairs.drop(index = [index],inplace=True)
        #     #continue
        # else:
        pair_save_list.append(row[1])
            # continue
    #print('Anote Clean')
    for index,row in annote.iterrows():
        if row[0] not in pair_save_list:
            annote.drop(index = [index],inplace=True)
        else:
            pair_save_list.remove(row[0])
    pairs.to_csv(pairs_file_name,index=False,sep=',')
    annote.to_csv(annote_file_name,index=False,sep = ':',quoting=csv.QUOTE_NONE)
    #return anote,pairs
    #print('save/drop:',save,'/',drop)
    #pairs = pd.read_csv('vehicle-pairs-test-clean.csv',sep=',',encoding='utf8')
    #pairs.to_csv('vehicle-pairs-test-clean.csv',index=False)

def human_clean(annote,pairs,vid_resolution):
    pairs_file_name = pairs
    annote_file_name = annote
    pairs = pd.read_csv(pairs,sep=',',encoding='utf8')
    annote = pd.read_csv(annote,sep=':',encoding='utf8')
    annote = resize_point(annote,(128,64),'human')
    #print('Pair Clean')
    pair_save_list = []#最后留下的图像对
    for index,row in pairs.iterrows():
        #if edge_filter(row,(1280,720)):
        #    pairs.drop(index = [index],inplace=True)
        if num_keypoint_filter(row,annote,5,'human'):
            pairs.drop(index = [index],inplace=True)
        else:
            pair_save_list.append(row[1])#保存to列的图像文件名
            continue
    #print('Anote Clean')
    for index,row in annote.iterrows():
        if row[0] not in pair_save_list:
            annote.drop(index = [index],inplace=True)
        else:
            pair_save_list.remove(row[0])
    pairs.to_csv(pairs_file_name,index=False,sep=',')
    annote.to_csv(annote_file_name,index=False,sep = ':',quoting=csv.QUOTE_NONE)
    #return anote,pairs





